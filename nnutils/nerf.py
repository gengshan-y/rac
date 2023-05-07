# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import numpy as np
import pdb
import copy
import math
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from functorch import vmap, combine_state_for_ensemble
from pytorch3d import transforms
import trimesh
from nnutils.geom_utils import fid_reindex
from nnutils.urdf_utils import angle_to_rts
from nnutils.rendering import render_rays
from nnutils.geom_utils import raycast, near_far_to_bound, chunk_rays, \
                            create_base_se3, refine_rt, rts_compose
from utils.io import load_root
from collections import defaultdict

class Embedding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True, alpha=None):
        """
        adapted from https://github.com/kwea123/nerf_pl/blob/master/models/nerf.py
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(Embedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.nfuncs = len(self.funcs)
        self.out_channels = in_channels*(len(self.funcs)*N_freqs+1)
        if alpha is None:
            self.alpha = self.N_freqs
        else: self.alpha = alpha

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...) 
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Inputs:
            x: (B, self.in_channels)

        Outputs:
            out: (B, self.out_channels)
        """
        # consine features
        if self.N_freqs>0:
            shape = x.shape
            bs = shape[0]
            input_dim = shape[-1]
            output_dim = input_dim*(1+self.N_freqs*self.nfuncs)
            out_shape = shape[:-1] + ((output_dim),)
            device = x.device

            x = x.view(-1,input_dim)
            out = []
            for freq in self.freq_bands:
                for func in self.funcs:
                    out += [func(freq*x)]
            out =  torch.cat(out, -1)

            ## Apply the window w = 0.5*( 1+cos(pi + pi clip(alpha-j)) )
            out = out.view(-1, self.N_freqs, self.nfuncs, input_dim)
            window = self.alpha - torch.arange(self.N_freqs).to(device)
            window = torch.clamp(window, 0.0, 1.0)
            window = 0.5 * (1 + torch.cos(np.pi * window + np.pi))
            window = window.view(1,-1, 1, 1)
            out = window * out
            out = out.view(-1,self.N_freqs*self.nfuncs*input_dim)

            out = torch.cat([x, out],-1)
            out = out.view(out_shape)
        else: out = x
        return out

class BaseMLP(nn.Module):
    def __init__(self,
                 D=8, W=256,
                 in_channels=63,
                 out_channels=3, 
                 skips=[4], 
                 activation=nn.ReLU(True)):
        """
        adapted from https://github.com/kwea123/nerf_pl/blob/master/models/nerf.py
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels: number of input channels 
        out_channels: number of output channels 
        skips: add skip connection in the Dth layer
        """
        super(BaseMLP, self).__init__()
        self.D = D
        self.W = W
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skips = skips

        # linear layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels, W)
            elif i in skips:
                layer = nn.Linear(W+in_channels, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, activation)
            setattr(self, f"linear_{i+1}", layer)
        self.linear_final = nn.Linear(W, out_channels)

    def reinit(self,gain=1):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if hasattr(m.weight,'data'):
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5*gain))
                if hasattr(m.bias,'data'):
                    m.bias.data.zero_()

    def forward(self, x):
        """
        Inputs:
            x: (B, self.in_channels)

        Outputs:
            out: (B, self.out_channels)
        """
        out = x
        for i in range(self.D):
            if i in self.skips:
                out = torch.cat([x, out], -1)
            out = getattr(self, f"linear_{i+1}")(out)
        out = self.linear_final(out)
        return out

class MultiMLP(nn.Module):
    def __init__(self, num_net, **kwargs):
        super(MultiMLP, self).__init__()
        nets = []
        for i in range(num_net):
            nets.append(  BaseMLP(**kwargs) )
        fnet, self.params, self.buffers = combine_state_for_ensemble(nets)
        self.fnet = [fnet] # avoid racognized as nn.module
        self.params = nn.ParameterList([nn.Parameter(p) for p in self.params])
        self.num_net = num_net

    def forward(self, x):
        """
        x: ...,-1
        out: ...,N,-1
        """
        shape = x.shape[:-1]
        in_channels = self.fnet[0].stateless_model.in_channels
        x = x.view((-1,self.num_net,in_channels)) # assumes model-batch dim is 2nd last
        
        params = [i for i in self.params]
        out = vmap(self.fnet[0], in_dims=(0,0,1))(params, self.buffers, x)
        out = out.permute(1,0,2) # bs, num_net, out_channels
        out = out.reshape(shape+(-1,))
        return out

class NeRF(nn.Module):
    def __init__(self,
                 D=8, W=256,
                 in_channels_xyz=63, in_channels_dir=27,
                 out_channels=3, 
                 skips=[4], raw_feat=False, init_beta=1./100, 
                 activation=nn.ReLU(True), in_channels_code=0, vid_code=None):
        """
        adapted from https://github.com/kwea123/nerf_pl/blob/master/models/nerf.py
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        skips: add skip connection in the Dth layer
        in_channels_code: only used for nerf_skin,
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.in_channels_code = in_channels_code
        self.skips = skips

        # video code
        self.vid_code = vid_code
        if vid_code is not None:
            self.num_vid, self.num_codedim = self.vid_code.weight.shape
            in_channels_xyz += self.num_codedim
            self.rand_ratio = 1. # 1: fully random

        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W+in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, activation)
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(W, W)

        # direction encoding layers
        self.dir_encoding = nn.Sequential(
                                nn.Linear(W+in_channels_dir, W//2),
                                activation)

        # output layers
        self.sigma = nn.Linear(W, 1)
        self.rgb = nn.Sequential(
                        nn.Linear(W//2, out_channels),
                        )

        self.raw_feat = raw_feat
        self.out_channels = out_channels

        self.beta = torch.Tensor([init_beta]) # logbeta
        self.beta = nn.Parameter(self.beta)
        self.symm_ratio = 0
        self.rand_ratio = 0
        self.use_dir = False

    def reinit(self,gain=1):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if hasattr(m.weight,'data'):
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5*gain))
                if hasattr(m.bias,'data'):
                    m.bias.data.zero_()

    def forward(self, x, vidid=None, beta=None):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
               the embedded vector of position and direction
            vidid: same size as input_xyz
            beta will overwrite vidid

        Outputs:
            out: (B, 4), rgb and sigma
        """
        if x.shape[-1] == self.in_channels_xyz:
            sigma_only = True
        else:
            sigma_only = False
        if sigma_only:
            input_xyz, input_dir = \
                torch.split(x, [self.in_channels_xyz, 0], dim=-1)
        else:
            input_xyz, input_dir = \
                torch.split(x, [self.in_channels_xyz, self.in_channels_dir], dim=-1)

        # add instance shape
        if self.vid_code is not None:
            if beta is None: 
                if vidid is None:
                    vid_code = self.vid_code.weight.mean(0).expand(input_xyz.shape[:-1] + (-1,))
                else:
                    #if hasattr(self.vid_code, "shape_code_pred"):
                    #    vid_code = self.vid_code.shape_code_pred[vidid]
                    #else:
                    #    vid_code = self.vid_code(vidid)
                    vid_code = self.vid_code(vidid)
                if self.training:
                    ##TODO 
                    vidid = torch.randint(self.num_vid, input_xyz.shape[:1])
                    vidid = vidid.to(input_xyz.device)
                    rand_code = self.vid_code(vidid)
                    rand_code = rand_code[:,None].expand(vid_code.shape)
                    rand_mask = torch.rand_like(vidid.float()) < self.rand_ratio
                    vid_code = torch.where(rand_mask[:,None,None], rand_code, vid_code)
            else:
                vid_code = beta
            input_xyz = torch.cat([input_xyz, vid_code],-1)

        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        sigma = self.sigma(xyz_)
        if sigma_only: return sigma
        
        xyz_encoding_final = self.xyz_encoding_final(xyz_)
        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir], -1)
        dir_encoding = self.dir_encoding(dir_encoding_input)
        rgb = self.rgb(dir_encoding)
        rgb = rgb.sigmoid()

        out = torch.cat([rgb, sigma], -1)
        return out

class RTHead(BaseMLP):
    """
    modify the output to be rigid transforms
    """
    def __init__(self, use_quat, pose_code=None, **kwargs):
        super(RTHead, self).__init__(**kwargs)
        # use quaternion when estimating full rotation
        # use exponential map when estimating delta rotation
        self.use_quat=use_quat
        if self.use_quat: self.num_output=7
        else: self.num_output=6
        self.scale_t = 0.1

        self.reinit(gain=1)
        
        self.pose_code = pose_code

    def forward(self, x):
        if self.pose_code is not None:
            x = self.pose_code(x)
        x = self.forward_decode(x)
        return x

    def forward_decode(self, x):
        # output: NxBx(9 rotation + 3 translation)
        x = super(RTHead, self).forward(x)
        bs = x.shape[0]
        rts = x.view(-1,self.num_output)  # bs B,x
        B = rts.shape[0]//bs

        tmat= rts[:,0:3] * self.scale_t

        if self.use_quat:
            rquat=rts[:,3:7]
            rquat=F.normalize(rquat,2,-1)
            rmat=transforms.quaternion_to_matrix(rquat) 
        else:
            rot=rts[:,3:6]
            rmat = transforms.so3_exponential_map(rot)
        rmat = rmat.view(-1,9)

        rts = torch.cat([rmat,tmat],-1)
        rts = rts.view(bs,1,-1)
        return rts

class TrnHeadIntr(BaseMLP):
    """
    translation with intrinsics
    """
    def __init__(self, **kwargs):
        super(TrnHeadIntr, self).__init__(**kwargs)
        self.reinit(gain=1)

    def forward(self, x, kvec_embedded):
        #x = kvec_embedded
        x = torch.cat([x, kvec_embedded],-1)
        x = super(TrnHeadIntr, self).forward(x)
        tvec = x.view(-1,1,3)
        return tvec
            
class SkelHead(BaseMLP):
    """
    modify the output to be rigid transforms from a kinematic chain
    """
    def __init__(self, urdf, joints, sim3, rest_angles, pose_code, rest_pose_code,
                data_offset,**kwargs):
        super(SkelHead, self).__init__(**kwargs)
        self.urdf = urdf
        self.sim3 = sim3
        self.joints = nn.Parameter(joints)
        self.num_vid = len(data_offset)-1
        self.data_offset = data_offset
        jlen_scale_z = torch.zeros(1,len(joints))
        self.jlen_scale_z = nn.Parameter(jlen_scale_z) # optimize bone length for canonical skel 
        jlen_scale = torch.zeros(self.num_vid,len(joints))
        self.jlen_scale = nn.Parameter(jlen_scale) # optimize bone length for instance-specific skel
        self.sim3_vid = nn.Parameter(torch.zeros(self.num_vid,sim3.shape[0]))
        self.rest_angles = rest_angles

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if hasattr(m.bias,'data'):
                    m.bias.data.zero_()
        
        self.pose_code = pose_code
        self.rest_pose_code = rest_pose_code

    def forward(self, x):
        # get skeleton id
        vid, _ = fid_reindex(x, self.num_vid, self.data_offset)
        vid = vid.view(-1)
        x = self.pose_code(x)
        x = self.forward_decode(x, vid)
        return x

    def forward_decode(self, x, vid, show_rest_pose=False,
                jlen_scale_in=None, sim3_in=None):
        # returns canonical bone, zero pose ->vid bone, t pose
        # vid: bs
        # input: bs, depth, nfeat
        # output: NxBx(9 rotation + 3 translation)
        bs = x.shape[0]
        device = x.device

        if show_rest_pose: 
            x = self.rest_pose_code.weight.repeat(bs,1)
        x = super(SkelHead, self).forward(x)
        angles = x.view(bs,-1)  # bs B-1: 0,1,2; 3,4,5; ...
        #if show_rest_pose: angles[:] = 0

        ##TODO debug
        #if angles.shape[0]>1:
        #    angles[:]=0
        #    angles[:,1] = -1
        angles = angles + self.rest_angles.to(device)
        angles_z = torch.zeros_like(angles) + self.rest_angles.to(device)

        # convert from angles to rts
        #sim3_can = self.sim3[None]
        #TODO do not update the trans/orient of canonical skel
        sim3_can = torch.cat([self.sim3[:7].detach(), self.sim3[7:]], 0)[None]
        if vid is None:
            jlen_scale = self.jlen_scale_z
            sim3 = sim3_can
        else:
            jlen_scale = self.jlen_scale_z + self.jlen_scale[vid]
            if show_rest_pose:
                sim3 = sim3_can + self.sim3_vid[vid]
                sim3[:,:7] = sim3_can[:,:7]
            else:
                sim3 = sim3_can + self.sim3_vid[vid]
        if jlen_scale_in is not None: jlen_scale = jlen_scale_in
        if sim3_in is not None: sim3 = sim3_in
        joints = self.joints.detach() # do not change joints original position
        # points to orignal joints, zero config
        joints_z = self.update_joints(self.urdf, joints, self.jlen_scale_z)
        fk_z = angle_to_rts(self.urdf, joints_z, angles_z, sim3_can) # 1,B,4,4
        # points to vid joints, t config
        joints = self.update_joints(self.urdf, joints, jlen_scale)
        fk   = angle_to_rts(self.urdf, joints, angles, sim3) # bs,B,4,4
        rmat = fk[...,:3,:3]
        rmat_z=fk_z[...,:3,:3]
        tmat = fk[...,:3,3]
        tmat_z=fk_z[...,:3,3]

        rmat_zi = rmat_z.permute(0,1,3,2)
        tmat_zi = -rmat_z.permute(0,1,3,2).matmul(tmat_z[...,None])[...,0]

        # world points transforms from zero to posed
        tmat = rmat.matmul(tmat_zi[...,None])[...,0] + tmat
        rmat = rmat.matmul(rmat_zi)

        rmat = rmat.reshape(-1,9)
        tmat = tmat.reshape(-1,3)

        rts = torch.cat([rmat,tmat],-1)
        rts = rts.view(bs,1,-1)
        return rts    

    def forward_abs(self, x=None,vid=None, show_rest_pose=False, 
                jlen_scale_in=None, sim3_in=None):
        # returns local points -> vid t / vid rest
        # x: time index, if none, will use rest code
        # vid: bs, if none, will return original length 
        if x is None or show_rest_pose==True:
            x = self.rest_pose_code.weight
            if vid is not None:
                vid = vid.view(-1)
                x = x.repeat(len(vid),1)
        else:
            if vid is None:
                vid, _ = fid_reindex(x, self.num_vid, self.data_offset)
            vid = vid.view(-1)
            x = self.pose_code(x)

        # absolute se3
        # points: joint coordinate to root coordinate
        # output: NxBx(9 rotation + 3 translation)
        bs = x.shape[0]
        device = x.device

        x = super(SkelHead, self).forward(x)
        angles = x.view(bs,-1)  # bs B
        angles = angles + self.rest_angles.to(device)

        # convert from angles to rts
        #sim3_can = self.sim3[None]
        #TODO do not update the trans/orient of canonical skel
        sim3_can = torch.cat([self.sim3[:7].detach(), self.sim3[7:]], 0)[None]
        if vid is None:
            jlen_scale = self.jlen_scale_z
            sim3 = sim3_can
        else:
            vid = vid.long()
            jlen_scale = self.jlen_scale_z + self.jlen_scale[vid]
            if show_rest_pose:
                sim3 = sim3_can + self.sim3_vid[vid]
                sim3[:,:7] = sim3_can[:,:7]
            else:
                sim3 = sim3_can + self.sim3_vid[vid]
        if jlen_scale_in is not None: jlen_scale = jlen_scale_in
        if sim3_in is not None: sim3 = sim3_in
        joints = self.joints.detach() # do not change joints original position
        joints = self.update_joints(self.urdf, joints, jlen_scale)
        fk = angle_to_rts(self.urdf, joints, angles, sim3) # bs,B,4,4
        rmat = fk[...,:3,:3]
        tmat = fk[...,:3,3]

        rmat = rmat.reshape(-1,9)
        tmat = tmat.reshape(-1,3)

        rts = torch.cat([rmat,tmat],-1)
        rts = rts.view(bs,1,-1)
        return rts, angles

    @staticmethod
    def update_joints(urdf, joints, jlen_scale):
        """
        scale bone length
        joints, N,3
        jlen_scale, bs, N log scales
        returns, joints, bs,N,3
        """
        joints = joints.clone()
        jlen_scale = jlen_scale.clone()
        if urdf.robot_name=='a1' or urdf.robot_name=='laikago':
            symm_idx = [3,4,5,0,1,2,9,10,11,6,7,8]
        elif urdf.robot_name=='wolf' or urdf.robot_name=='wolf_mod':
            symm_idx = [0,1,2,3, 8,9,10,11,4,5,6,7, 12,13,14,15,16, 21,22,23,24,17,18,19,20]
        elif urdf.robot_name=='human' or urdf.robot_name=='human_mod':
            symm_idx = [0,1,2,3, 8,9,10,11,4,5,6,7, 15,16,17,12,13,14]
        jlen_scale = (jlen_scale + jlen_scale[:,symm_idx]) / 2
        joints = joints[None] * jlen_scale.exp()[...,None] # 1,N,3 x bs,N,1

        # update urdf as well (may be run multiple times in a pass)
        # this only affects bone visualization
        for i,joint_name in enumerate(urdf.name2joints_idx):
            urdf.joint_map[joint_name].origin[:3,3] = joints[0,i].detach().cpu()
        return joints 


class FrameCode(nn.Module):
    """
    frame index and video index to code
    """
    def __init__(self, num_freq, embedding_dim, vid_offset, scale=1):
        super(FrameCode, self).__init__()
        self.vid_offset = vid_offset
        self.num_vids = len(vid_offset)-1
        # compute maximum frequency:64-127 frame=>10
        max_ts = (self.vid_offset[1:] - self.vid_offset[:-1]).max()
        if num_freq>0:
            self.num_freq = 2*int(np.log2(max_ts))-2
        else:
            self.num_freq = 0
#        self.num_freq = num_freq

        self.fourier_embed = Embedding(1,self.num_freq,alpha=self.num_freq)
        #self.fourier_embed = Embedding(1,num_freq,alpha=num_freq)
        self.basis_mlp = nn.Linear(self.num_vids*self.fourier_embed.out_channels,
                                embedding_dim)
        self.scale = scale # input scale factor
        self.reinit(gain=1)

    def reinit(self,gain=1):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if hasattr(m.weight,'data'):
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5*gain))
                if hasattr(m.bias,'data'):
                    m.bias.data.zero_()

    def forward(self, fid):
        """
        fid->code: N->N,embedding_dim
        """
        fid = fid.long()
        bs = fid.shape[0]
        vid, tid = fid_reindex(fid, self.num_vids, self.vid_offset)
        tid = tid*self.scale
        tid = tid.view(bs,1)
        vid = vid.view(bs,1)
        coeff = self.fourier_embed(tid) # N, n_channels
        vid = F.one_hot(vid, num_classes=self.num_vids) # N, 1, num_vids
        # pad zeros for each
        coeff = coeff[...,None] * vid # N, n_channels, num_vids
        coeff = coeff.view(bs, -1)
        code = self.basis_mlp(coeff)
        return code


class FrameCode_old(nn.Module):
    """
    frame index and video index to code
    """
    def __init__(self, num_freq, embedding_dim, vid_offset, scale=1):
        super(FrameCode, self).__init__()
        self.vid_offset = vid_offset
        self.num_vids = len(vid_offset)-1
        # compute maximum frequency:64-127 frame=>10
        max_ts = (self.vid_offset[1:] - self.vid_offset[:-1]).max()
        self.num_freq = 2*int(np.log2(max_ts))-2
#        self.num_freq = num_freq

        self.fourier_embed = Embedding(1,num_freq,alpha=num_freq)
        self.basis_mlp = nn.Linear(self.num_vids*self.fourier_embed.out_channels,
                                embedding_dim)
        self.scale = scale # input scale factor

    def forward(self, fid):
        """
        fid->code: N->N,embedding_dim
        """
        bs = fid.shape[0]
        vid, tid = fid_reindex(fid, self.num_vids, self.vid_offset)
        tid = tid*self.scale
        tid = tid.view(bs,1)
        vid = vid.view(bs,1)
        coeff = self.fourier_embed(tid) # N, n_channels
        vid = F.one_hot(vid, num_classes=self.num_vids) # N, 1, num_vids
        # pad zeros for each
        coeff = coeff[...,None] * vid # N, n_channels, num_vids
        coeff = coeff.view(bs, -1)
        code = self.basis_mlp(coeff)
        return code

class RTExplicit(nn.Module):
    """
    index rigid transforms from a dictionary
    """
    def __init__(self, max_t, delta=False, rand=True):
        super(RTExplicit, self).__init__()
        self.max_t = max_t
        self.delta = delta

        # initialize rotation
        trans = torch.zeros(max_t, 3)
        if delta:
            rot = torch.zeros(max_t, 3) 
        else:
            if rand:
                rot = torch.rand(max_t, 4) * 2 - 1
            else:
                rot = torch.zeros(max_t, 4)
                rot[:,0] = 1
        se3 = torch.cat([trans, rot],-1)

        self.se3 = nn.Parameter(se3)
        self.num_output = se3.shape[-1]


    def forward(self, x):
        # output: NxBx(9 rotation + 3 translation)
        x = x.long()
        bs = x.shape[0]
        x = self.se3[x] # bs B,x
        rts = x.view(-1,self.num_output)
        B = rts.shape[0]//bs
        
        tmat= rts[:,0:3] *0.1

        if self.delta:
            rot=rts[:,3:6]
            rmat = transforms.so3_exponential_map(rot)
        else:
            rquat=rts[:,3:7]
            rquat=F.normalize(rquat,2,-1)
            rmat=transforms.quaternion_to_matrix(rquat) 
        rmat = rmat.view(-1,9)

        rts = torch.cat([rmat,tmat],-1)
        rts = rts.view(bs,1,-1)
        return rts

class RTExpMLP(nn.Module):
    """
    index rigid transforms from a dictionary
    """
    def __init__(self, max_t, num_freqs, t_embed_dim, data_offset, delta=False):
        super(RTExpMLP, self).__init__()
        #self.root_code = nn.Embedding(max_t, t_embed_dim)
        self.root_code = FrameCode(num_freqs, t_embed_dim, data_offset, scale=0.1)
        #self.root_code = FrameCode(num_freqs, t_embed_dim, data_offset)

        self.base_rt = RTExplicit(max_t, delta=delta,rand=False)
        #self.base_rt = RTHead(use_quat=True, 
        #            D=2, W=64,
        #            in_channels_xyz=t_embed_dim,in_channels_dir=0,
        #            out_channels=7, raw_feat=True)
        #self.base_rt = nn.Sequential(self.root_code, self.base_rt)
        self.mlp_rt = RTHead(use_quat=False, 
                    in_channels=t_embed_dim,
                    out_channels=6)
        self.delta_rt = nn.Sequential(self.root_code, self.mlp_rt)

    def forward(self, x):
        # output: NxBx(9 rotation + 3 translation)
        base_rts = self.base_rt(x)
        delt_rts = self.delta_rt(x)

        # magnify gradient by 10x
        base_rts = base_rts * 10 - (base_rts*9).detach()
        
        rmat = base_rts[:,0,:9].view(-1,3,3)
        tmat = base_rts[:,0,9:12]
        
        delt_rmat = delt_rts[:,0,:9].view(-1,3,3)
        delt_tmat = delt_rts[:,0,9:12]
    
        tmat = tmat + rmat.matmul(delt_tmat[...,None])[...,0]
        #tmat = tmat + delt_tmat
        rmat = rmat.matmul(delt_rmat)
        
        rmat = rmat.view(-1,9)
        rts = torch.cat([rmat,tmat],-1)
        rts = rts.view(-1,1,12)
        return rts

class NeRFFeat(BaseMLP):
    """
    nerf feature
    """
    def __init__(self, init_beta=1./100,  **kwargs):
        super(NeRFFeat, self).__init__(**kwargs)
        self.conv3d_refine = nn.Sequential(
           Conv3dBlock(4,2,stride=(1,1,1)),
                Conv3d(2,1,3, (1,1,1),1,bias=True),
                )
        self.beta = torch.Tensor([init_beta]) # logbeta
        self.beta = nn.Parameter(self.beta)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if hasattr(m.bias,'data'):
                    m.bias.data.zero_()

    def forward(self, x):
        feat = super(NeRFFeat, self).forward(x)
        return feat

class Conv3dBlock(nn.Module):
    '''
    3d convolution block as 2 convolutions and a projection
    layer
    '''
    def __init__(self, in_planes, out_planes, stride=(1,1,1)):
        super(Conv3dBlock, self).__init__()
        if in_planes == out_planes and stride==(1,1,1):
            self.downsample = None
        else:
            self.downsample = projfeat3d(in_planes, out_planes,stride)
        self.conv1 = Conv3d(in_planes, out_planes, 3, stride, 1)
        self.conv2 = Conv3d(out_planes, out_planes, 3, (1,1,1), 1)
            

    def forward(self,x):
        out = F.relu(self.conv1(x),inplace=True)
        if self.downsample:
            x = self.downsample(x)
        out = F.relu(x + self.conv2(out),inplace=True)
        return out

def Conv3d(in_planes, out_planes, kernel_size, stride, pad,bias=False):
    if bias:
        return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride,bias=bias))
    else:
        return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride,bias=bias),
                         nn.BatchNorm3d(out_planes))

class projfeat3d(nn.Module):
    '''
    Turn 3d projection into 2d projection
    '''
    def __init__(self, in_planes, out_planes, stride):
        super(projfeat3d, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(in_planes, out_planes, (1,1), padding=(0,0), stride=stride[:2],bias=False)
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self,x):
        b,c,d,h,w = x.size()
        x = self.conv1(x.view(b,c,d,h*w))
        x = self.bn(x)
        x = x.view(b,-1,d//self.stride[0],h,w)
        return x

class NeRFBG(nn.Module):
    """
    nerf background
    """
    def __init__(self, num_freqs, data_offset, config, opts,**kwargs):
        super(NeRFBG, self).__init__()
        if opts.bgmlp=='nerf':
            self.nerf_mlp = NeRF_old(**kwargs)
        elif opts.bgmlp=='hmnerf':
            self.nerf_mlp = HMNeRF(**kwargs)
            num_freqs = 10
            env_code_dim = 64
            self.env_code = FrameCode_old(num_freqs, env_code_dim, 
                                        data_offset, scale=1)
        self.nerf_mlp.use_dir = True
        self.embedding_xyz = Embedding(3, num_freqs)
        self.embedding_dir = Embedding(3,4 )

        # pose mlp
        self.num_fr = data_offset[-1]
        self.num_vid = len(data_offset)-1
        self.data_offset = data_offset
        t_embed_dim = 128
        self.cam_mlp = RTExpMLP_old(self.num_fr, num_freqs, t_embed_dim, data_offset)
        self.bg2fg_scale = nn.Parameter(torch.ones(self.num_vid))
        bg2world = torch.zeros(7) # trans, xyzw
        bg2world[-1] = 1
        self.bg2world = nn.Parameter(bg2world[None].repeat(self.num_vid,1)) # trans, quat
       
        # vis mlp
        vid_vis_code = nn.Embedding(self.num_vid, 32)
        self.nerf_vis = NeRF_old(in_channels_xyz=kwargs['in_channels_xyz'], D=5, W=64,
                                    out_channels=1, in_channels_dir=0,
                                    raw_feat=True, vid_code=vid_vis_code)

        # nf plane 
        self.near_far = torch.zeros(self.num_fr,2)
        self.near_far = nn.Parameter(self.near_far)
            
        # others
        self.obj_bound = near_far_to_bound(self.near_far)
        self.ndepth = 128
        self.progress = 0
        self.opts = copy.deepcopy(opts)
        self.opts.use_cc = False
        self.opts.eikonal_wt = 0
        self.opts.full_mesh = False
        self.nerf_models = {'coarse': self.nerf_mlp}
        self.embeddings = {'xyz':self.embedding_xyz, 'dir':self.embedding_dir}

        # load weights
        if opts.bg_path!="":
            states = torch.load(opts.bg_path, map_location='cpu')
            #states = torch.load('tmp/bg.pth', map_location='cpu')
            nerf_states = self.rm_module_prefix(states, 
                        prefix='module.nerf_coarse')
            cam_states = self.rm_module_prefix(states, 
                        prefix='module.nerf_root_rts')
            vis_states = self.rm_module_prefix(states, 
                        prefix='module.nerf_vis')
            env_states = self.rm_module_prefix(states, 
                        prefix='module.env_code')
            self.nerf_mlp.load_state_dict(nerf_states, strict=False)
            self.cam_mlp.load_state_dict(cam_states, strict=False)
            self.nerf_vis.load_state_dict(vis_states, strict=False)
            self.env_code.load_state_dict(env_states, strict=False)
            self.embedding_xyz.alpha = int(states['module.alpha'][0].numpy())
            self.embedding_dir.alpha = int(states['module.alpha'][0].numpy())
            # load near far
            self.near_far.data = states['module.near_far']
            # vars
            var_path = opts.bg_path.replace('params', 'vars').replace('.pth', '.npy')
            self.latest_vars = np.load(var_path,allow_pickle=True)[()]

        # opts
        self.ft_bgcam = not opts.freeze_bgcam

    @staticmethod
    def rm_module_prefix(states, prefix='module'):
        new_dict = {}
        for i in states.keys():
            v = states[i]
            if i[:len(prefix)] == prefix:
                i = i[len(prefix)+1:]
                new_dict[i] = v
        return new_dict

    def get_rts(self, t):
        bs = t.shape[0]
        device = t.device
        t = t.long()
        vidid,_ = fid_reindex(t, self.num_vid, self.data_offset)

        rts = self.cam_mlp(t)
        if not self.ft_bgcam:
            rts = rts.detach() # TODO
        rts_base = create_base_se3(bs, device)
        rts = refine_rt(rts_base, rts)
        rts[:,:3,3] *= self.bg2fg_scale[vidid,None]
        return rts 

    def forward(self, xy, t, auto_reshape=True, view2x=None):
        """
        xy is normalized coords by intrinsics
        should optimize intrinsics
        """
        # compute cams
        bs = t.shape[0]
        device = t.device
        vidid,_ = fid_reindex(t, self.num_vid, self.data_offset)

        rts = self.get_rts(t)
        if view2x is not None: 
            # bg-x = view-x@bg-view
            rts = rts_compose(view2x, rts)

        Rmat = rts[:,:3,:3]
        Tmat = rts[:,:3,3]
        Tmat /= self.bg2fg_scale[vidid,None]
        Kinv = torch.eye(3,device=device)[None].repeat(bs,1,1)
        
        # raycast
        near_far = self.near_far[t].detach()
        rays = raycast(xy, Rmat, Tmat, Kinv, near_far)
        rays['vidid'] = vidid[:,None,None].repeat(1,rays['nsample'],1)
        del rays['rtk_vec']
        del rays['xys']
        del rays['xy_uncrop']

        # query mlp
        bs_rays = rays['bs'] * rays['nsample'] # over pixels
        opts = self.opts

        if opts.bgmlp=='hmnerf':
            rays['env_code'] = self.env_code(t)[:,None]
            rays['env_code'] = rays['env_code'].repeat(1,rays['nsample'],1)

        results=defaultdict(list)
        for i in range(0, bs_rays, opts.chunk):
            rays_chunk = chunk_rays(rays,i,opts.chunk)
            # decide whether to use fine samples 
            if self.progress > opts.fine_steps:
                use_fine = True
            else:
                use_fine = False
            rendered_chunks = render_rays(self.nerf_models,
                        self.embeddings,
                        rays_chunk,
                        N_samples = self.ndepth,
                        use_disp=False,
                        perturb=False,
                        noise_std=0,
                        chunk=2048, # chunk size is effective in val mode
                        obj_bound=self.obj_bound,
                        use_fine=use_fine,
                        img_size=-1, # not used
                        progress=self.progress,
                        opts=opts,
                        )
            for k, v in rendered_chunks.items():
                results[k] += [v]
        for k, v in results.items():
            if v[0].dim()==0: # loss
                v = torch.stack(v).mean()
            else:
                if isinstance(v, list):
                    v = torch.cat(v, 0)
                if self.training or not auto_reshape:
                    v = v.view(rays['bs'],rays['nsample'],-1)
                else:
                    v = v.view(bs,opts.render_size, opts.render_size, -1)
            results[k] = v
        return results['img_coarse'], results['sil_coarse'], results['depth_rnd']

class NeRFTransient(BaseMLP):
    """
    nerf transient
    """
    def __init__(self, num_freqs, tcode_dim, data_offset, **kwargs):
        super(NeRFTransient, self).__init__(**kwargs)
        self.xyembed = Embedding(2,num_freqs)
        self.tcode = FrameCode(num_freqs, tcode_dim, data_offset)
        # nf plane, placeholder
        self.num_fr = data_offset[-1]
        self.near_far = torch.stack([torch.zeros(self.num_fr), 
                                     torch.ones(self.num_fr)],1)

    def forward(self, xy, t, auto_reshape=True):
        """
        auto_reshape is never used, consider removing it for nerfbg as well
        xy: bs,N,2
        rgb: bs,N,3
        """
        xy = xy.clone()
        t = t.clone()
        #xy[...,0] = xy[...,0] / xy[...,0].max() * 2 -1
        #xy[...,1] = xy[...,1] / xy[...,1].max() * 2 -1
        xy_embedded = self.xyembed(xy)
        t_embedded = self.tcode(t)[:,None].repeat(1,xy.shape[1],1)
        xyt_code = torch.cat([xy_embedded, t_embedded],-1)

        rgb = super(NeRFTransient, self).forward(xyt_code)
        rgb = rgb.sigmoid()
        return rgb,rgb[...,:1],rgb[...,:1]

class ResNetConv(nn.Module):
    """
    adapted from https://github.com/shubhtuls/factored3d/blob/master/nnutils/net_blocks.py
    """
    def __init__(self, in_channels):
        super(ResNetConv, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        if in_channels!=3:
            self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), 
                                    stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc=None

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        return x

class ConvProj(nn.Module):
    def __init__(self, num_feat):
        super(ConvProj, self).__init__()
        n_hidden = 128
        self.proj1 = conv2d(True, num_feat, n_hidden, kernel_size=1)
        self.proj2 = conv2d(True, n_hidden, n_hidden, kernel_size=1)
        self.proj3 = nn.Conv2d(n_hidden, num_feat, 
                        kernel_size=1, stride=1, padding=0, bias=True)
        self.proj = nn.Sequential(self.proj1,self.proj2, 
                                 self.proj3)

    def forward(self, feat):
        feat = self.proj(feat)
        return feat

class Encoder(nn.Module):
    """
    adapted from https://github.com/shubhtuls/factored3d/blob/master/nnutils/net_blocks.py
    Current:
    Resnet with 4 blocks (x32 spatial dim reduction)
    Another conv with stride 2 (x64)
    This is sent to 2 fc layers with final output nz_feat.
    """

    def __init__(self, input_shape, in_channels=3,out_channels=128, batch_norm=True):
        super(Encoder, self).__init__()
        self.resnet_conv = ResNetConv(in_channels=in_channels)
        self.conv1 = conv2d(batch_norm, 512, out_channels, stride=1, kernel_size=3)
        #net_init(self.conv1)

    def forward(self, img):
        feat = self.resnet_conv.forward(img) # 512,4,4
        feat = self.conv1(feat) # 128,4,4
        feat = F.max_pool2d(feat, 4, 4)
        feat = feat.view(img.size(0), -1)
        return feat

## 2D convolution layers
def conv2d(batch_norm, in_planes, out_planes, kernel_size=3, stride=1):
    """
    adapted from https://github.com/shubhtuls/factored3d/blob/master/nnutils/net_blocks.py
    """
    if batch_norm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.2,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.2,inplace=True)
        )

def grab_xyz_weights(nerf_model, clone=False):
    """
    zero grad for coarse component connected to inputs, 
    and return intermediate params
    """
    param_list = []
    input_layers=[0]+nerf_model.skips

    input_wt_names = []
    for layer in input_layers:
        input_wt_names.append(f"xyz_encoding_{layer+1}.0.weight")

    for name,p in nerf_model.named_parameters():
        if name in input_wt_names:
            # equiv since the wt after pos_dim does not change
            if clone:
                param_list.append(p.detach().clone()) 
            else:
                param_list.append(p) 
            ## get the weights according to coarse posec
            ## 63 = 3 + 60
            ## 60 = (num_freqs, 2, 3)
            #out_dim = p.shape[0]
            #pos_dim = nerf_model.in_channels_xyz-nerf_model.in_channels_code
            #param_list.append(p[:,:pos_dim]) # 
    return param_list

class BANMoCNN(nn.Module):
    def __init__(self, cnn_in_channels):
        super(BANMoCNN, self).__init__()
        self.encoder = Encoder((112,112), in_channels=cnn_in_channels,
                        out_channels=32)
        self.resnet_transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

        self.decoder_rot = RTHead(use_quat=True, D=1,
                    in_channels=32*4,
                    out_channels=7)
       
        self.encoder_app = nn.Sequential(
               Encoder((112,112), in_channels=cnn_in_channels, out_channels=32),
               nn.Linear(128, 64))
        self.encoder_shape = nn.Sequential(
               Encoder((112,112), in_channels=cnn_in_channels, out_channels=32),
               nn.Linear(128, 32))

        #self.embedding_kvec = Embedding(4, 6)
        #in_channels_xyz=4+4*6*2
        #self.decoder_trn = TrnHeadIntr(D=4,
        #            in_channels_xyz=4*32+in_channels_xyz,in_channels_dir=0,
        #            #in_channels_xyz=32+in_channels_xyz,in_channels_dir=0,
        #            out_channels=3, raw_feat=True)
        
        #from nnutils.mono import mono 
        #self.decoder_depth = mono()
        #self.decoder_depth = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        #self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

    def forward(self, imgs):
        #dp_feats = F.interpolate(dp_feats, (112,112), mode='bilinear')
        #feats = self.encoder(dp_feats)
        imgs = F.interpolate(imgs, (256,256), mode='bilinear')
        imgs_prsd = torch.stack([self.resnet_transform(x) for x in imgs])
        feats = self.encoder(imgs_prsd)
        rot_pred = self.decoder_rot(feats)
        rot_pred[...,9:] = 0 # zero translation

        # translation branch
        #kvec_embedded = self.embedding_kvec(kvec)
        #trn_pred = self.decoder_trn(feats, kvec_embedded)
        #root_pred = torch.cat([rot_pred, trn_pred],-1)

        # depth branch
        #imgs = self.midas_transform(imgs)
        #depth = 1. / self.decoder_depth(imgs)
        #depth = self.decoder_depth(imgs)
        return rot_pred

    def forward_with_code(self, imgs):
        imgs = F.interpolate(imgs, (256,256), mode='bilinear')
        imgs_prsd = torch.stack([self.resnet_transform(x) for x in imgs])
        feats = self.encoder(imgs_prsd)

        rot_pred = self.decoder_rot(feats)
        rot_pred[...,9:] = 0 # zero translation

        app_code = self.encoder_app(imgs_prsd)
        shape_code = self.encoder_shape(imgs_prsd)*0.1
        return rot_pred, app_code, shape_code

class NeRF_old(nn.Module):
    def __init__(self,
                 D=8, W=256,
                 in_channels_xyz=63, in_channels_dir=27,
                 out_channels=3, 
                 skips=[4], raw_feat=False, init_beta=1./100, 
                 activation=nn.ReLU(True), in_channels_code=0, vid_code=None,
                 color_act=True):
        """
        adapted from https://github.com/kwea123/nerf_pl/blob/master/models/nerf.py
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        skips: add skip connection in the Dth layer
        in_channels_code: only used for nerf_skin,
        """
        super(NeRF_old, self).__init__()
        self.D = D
        self.W = W
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.in_channels_code = in_channels_code
        self.skips = skips
        self.out_channels = out_channels
        self.raw_feat = raw_feat
        self.color_act = color_act

        # video code
        self.vid_code = vid_code
        if vid_code is not None:
            self.num_vid, self.num_codedim = self.vid_code.weight.shape
            in_channels_xyz += self.num_codedim
            self.rand_ratio = 1. # 1: fully random

        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W+in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, activation)
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(W, W)

        # direction encoding layers
        self.dir_encoding = nn.Sequential(
                                nn.Linear(W+in_channels_dir, W//2),
                                activation)

        # output layers
        self.sigma = nn.Linear(W, 1)
        self.rgb = nn.Sequential(
                        nn.Linear(W//2, self.out_channels),
                        )


        self.beta = torch.Tensor([init_beta]) # logbeta
        self.beta = nn.Parameter(self.beta)
        self.symm_ratio = 0
        self.rand_ratio = 0
        self.use_dir = False # use app code instead of view dir

    def reinit(self,gain=1):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if hasattr(m.weight,'data'):
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5*gain))
                if hasattr(m.bias,'data'):
                    m.bias.data.zero_()

    def forward(self, x, vidid=None, beta=None):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
               the embedded vector of position and direction
            vidid: same size as input_xyz

        Outputs:
            out: (B, 4), rgb and sigma
        """
        if x.shape[-1] == self.in_channels_xyz and not self.raw_feat:
            sigma_only = True
        else:
            sigma_only = False
        if x.shape[-1] == self.in_channels_xyz:
            input_xyz, input_dir = \
                torch.split(x, [self.in_channels_xyz, 0], dim=-1)
        else:
            input_xyz, input_dir = \
                torch.split(x, [self.in_channels_xyz, self.in_channels_dir], dim=-1)

        # add instance shape
        if self.vid_code is not None:
            if vidid is None:
                vid_code = self.vid_code.weight.mean(0).expand(input_xyz.shape[:-1] + (-1,))
            else:
                vid_code = self.vid_code(vidid)
            if self.training:
                vidid = torch.randint(self.num_vid, input_xyz.shape[:1])
                vidid = vidid.to(input_xyz.device)
                rand_code = self.vid_code(vidid)
                rand_code = rand_code[:,None].expand(vid_code.shape)
                rand_mask = torch.rand_like(vidid.float()) < self.rand_ratio
                vid_code = torch.where(rand_mask[:,None,None], rand_code, vid_code)
            input_xyz = torch.cat([input_xyz, vid_code],-1)

        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        sigma = self.sigma(xyz_)
        if sigma_only:
            return sigma

        xyz_encoding_final = self.xyz_encoding_final(xyz_)

        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir], -1)
        dir_encoding = self.dir_encoding(dir_encoding_input)
        rgb = self.rgb(dir_encoding)
        if self.raw_feat:
            out = rgb
        else:
            if self.color_act:
                rgb = rgb.sigmoid()
            out = torch.cat([rgb, sigma], -1)
        return out

class HMNeRF(nn.Module):
    def __init__(self, num_vid, **kwargs):
        super(HMNeRF, self).__init__()
        self.nerf = NeRF_old(**kwargs)
        kwargs.pop('vid_code')
        kwargs['D']=2
        kwargs['W']=64
        self.mnerf = MNeRF(num_vid, **kwargs)
        self.num_vid = num_vid
        #TODO replace with per video beta
        self.beta = torch.Tensor([0.1]) # logbeta
        self.beta = nn.Parameter(self.beta)

    def forward(self, x, vidid=None, beta=None):
        out2 = self.nerf(x, vidid=vidid)
        sigma_only = out2.shape[-1]==1
        out1 = self.mnerf(x, vidid=vidid, sigma_only=sigma_only)
        out = out1+out2
        if not sigma_only and not self.nerf.raw_feat:
            out[...,:3] = out[...,:3].sigmoid()
        return out

class MNeRF(nn.Module):
    def __init__(self, num_vid, **kwargs):
        super(MNeRF, self).__init__()
        self.num_vid = num_vid
        nets = []
        for i in range(num_vid):
            nets.append(  NeRF_old(**kwargs) )
        fnet, self.params, self.buffers = combine_state_for_ensemble(nets)
        self.fnet = [fnet] # avoid racognized as nn.module
        self.params = nn.ParameterList([nn.Parameter(p) for p in self.params])

        #TODO replace with per video beta
        self.beta = torch.Tensor([0.1]) # logbeta
        self.beta = nn.Parameter(self.beta)

    def forward(self, x, vidid=None, sigma_only=False):
        #out_test = self.forward_parallel(x, vidid=vidid, sigma_only=sigma_only)
        #if not self.training:        return out_test
        shape = x.shape[:-1]
        # bs, -1
        out = torch.zeros_like(x[...,:1]).view(-1,1)
        nelem = out.shape[0]
        x = x.view(nelem,-1)
        if sigma_only:
            out_last = (1,)
        else:
            out_last = self.fnet[0].stateless_model.out_channels
            if not self.fnet[0].stateless_model.raw_feat: out_last += 1
            out = torch.cat([out]*out_last,-1)
            out_last = (out_last,)
            
        if vidid is None:
            vidid = torch.randint(self.num_vid, (nelem, ))
            vidid = vidid.to(x.device)
        else:
            vidid = vidid.view(-1)

        unique_ids = torch.unique(vidid)
        for it in unique_ids:
            id_sel = vidid == it
            x_sel = x[id_sel][None]
            params = [i[it: it+1] for i in self.params]
            out_sel = vmap(self.fnet[0])(params, self.buffers, 
                                        x_sel)
            out[id_sel] = out_sel[0]
        out = out.view(shape+out_last)
        return out

    def forward_parallel(self, x, vidid=None, sigma_only=False):
        shape = x.shape[:-1]

        # bs, -1
        out = torch.zeros_like(x[...,:1]).view(-1,1)
        nelem = out.shape[0]
        x = x.view(nelem,-1)
        if sigma_only:
            out_last = (1,)
        else:
            out_last = self.fnet[0].stateless_model.out_channels
            if not self.fnet[0].stateless_model.raw_feat: out_last += 1
            out = torch.cat([out]*out_last,-1)
            out_last = (out_last,)
            
        if vidid is None:
            vidid = torch.randint(self.num_vid, (nelem, ))
            vidid = vidid.to(x.device)
        else:
            vidid = vidid.view(-1)

        unique_ids = torch.unique(vidid)
        if len(unique_ids)==1:
            uni_id = unique_ids[0]
            params = [i[uni_id: uni_id+1] for i in self.params]
            out = vmap(self.fnet[0])(params, self.buffers, 
                                        x[None])
            out = out[0]
        else:
            x_rearranged = []        
            id_rearranged =[]
            for it in range( self.num_vid ):
                id_sel = vidid == it
                x_sel = x[id_sel]
                x_rearranged.append( x_sel )
                id_rearranged.append( id_sel )

            max_bs = max([i.shape[0] for i in x_rearranged])
            x_padded = []
            for it in range( self.num_vid ):
                x_sel = x_rearranged[it]
                x_pad = F.pad( x_sel, (0,0,0, max_bs - x_sel.shape[0]))
                x_padded.append( x_pad )
            x_padded = torch.stack(x_padded,0)
            
            params = [i for i in self.params]
            out_padded = vmap(self.fnet[0])(params, self.buffers, 
                                            x_padded)
            
            for it in range( self.num_vid ):
                id_sel = id_rearranged[it]
                out_rearranged = out_padded[it][:id_sel.sum()]
                out[id_sel] = out_rearranged 

        # original shape
        out = out.view(shape+out_last)
        return out

class RTHead_old(NeRF_old):
    """
    modify the output to be rigid transforms
    """
    def __init__(self, use_quat, **kwargs):
        super(RTHead_old, self).__init__(**kwargs)
        # use quaternion when estimating full rotation
        # use exponential map when estimating delta rotation
        self.use_quat=use_quat
        if self.use_quat: self.num_output=7
        else: self.num_output=6
        self.scale_t = 0.1

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if hasattr(m.bias,'data'):
                    m.bias.data.zero_()
        self.reinit(gain=1)

    def reinit(self, gain=1):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if hasattr(m.weight,'data'):
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5*gain))
                if hasattr(m.bias,'data'):
                    m.bias.data.zero_()

    def forward(self, x):
        # output: NxBx(9 rotation + 3 translation)
        x = super(RTHead_old, self).forward(x)
        bs = x.shape[0]
        rts = x.view(-1,self.num_output)  # bs B,x
        B = rts.shape[0]//bs

        tmat= rts[:,0:3] * self.scale_t

        if self.use_quat:
            rquat=rts[:,3:7]
            rquat=F.normalize(rquat,2,-1)
            rmat=transforms.quaternion_to_matrix(rquat)
        else:
            rot=rts[:,3:6]
            rmat = transforms.so3_exponential_map(rot)
        rmat = rmat.view(-1,9)

        rts = torch.cat([rmat,tmat],-1)
        rts = rts.view(bs,1,-1)
        return rts

class RTExpMLP_old(nn.Module):
    """
    index rigid transforms from a dictionary
    """
    def __init__(self, max_t, num_freqs, t_embed_dim, data_offset, delta=False):
        super(RTExpMLP_old, self).__init__()
        self.root_code = FrameCode_old(num_freqs, t_embed_dim, data_offset, scale=0.1)

        self.base_rt = RTExplicit(max_t, delta=delta,rand=False)
        self.mlp_rt = RTHead_old(use_quat=False, 
                    in_channels_xyz=t_embed_dim, in_channels_dir=0,
                    out_channels=6, raw_feat=True)
        self.delta_rt = nn.Sequential(self.root_code, self.mlp_rt)

    def forward(self, x):
        # output: NxBx(9 rotation + 3 translation)
        base_rts = self.base_rt(x)
        delt_rts = self.delta_rt(x)

        # magnify gradient by 10x
        base_rts = base_rts * 10 - (base_rts*9).detach()
        
        rmat = base_rts[:,0,:9].view(-1,3,3)
        tmat = base_rts[:,0,9:12]
        
        delt_rmat = delt_rts[:,0,:9].view(-1,3,3)
        delt_tmat = delt_rts[:,0,9:12]
    
        tmat = tmat + rmat.matmul(delt_tmat[...,None])[...,0]
        rmat = rmat.matmul(delt_rmat)
        
        rmat = rmat.view(-1,9)
        rts = torch.cat([rmat,tmat],-1)
        rts = rts.view(-1,1,12)
        return rts

class FrameCode_old(nn.Module):
    """
    frame index and video index to code
    """
    def __init__(self, num_freq, embedding_dim, vid_offset, scale=1):
        super(FrameCode_old, self).__init__()
        self.vid_offset = vid_offset
        self.num_vids = len(vid_offset)-1
        # compute maximum frequency:64-127 frame=>10
        max_ts = (self.vid_offset[1:] - self.vid_offset[:-1]).max()
        self.num_freq = 2*int(np.log2(max_ts))-2
#        self.num_freq = num_freq

        self.fourier_embed = Embedding(1,num_freq,alpha=num_freq)
        self.basis_mlp = nn.Linear(self.num_vids*self.fourier_embed.out_channels,
                                embedding_dim)
        self.scale = scale # input scale factor

    def forward(self, fid):
        """
        fid->code: N->N,embedding_dim
        """
        bs = fid.shape[0]
        vid, tid = fid_reindex(fid, self.num_vids, self.vid_offset)
        tid = tid*self.scale
        tid = tid.view(bs,1)
        vid = vid.view(bs,1)
        coeff = self.fourier_embed(tid) # N, n_channels
        vid = F.one_hot(vid, num_classes=self.num_vids) # N, 1, num_vids
        # pad zeros for each
        coeff = coeff[...,None] * vid # N, n_channels, num_vids
        coeff = coeff.view(bs, -1)
        code = self.basis_mlp(coeff)
        return code
