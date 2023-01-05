# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import pdb
import time
import cv2
import numpy as np
import trimesh
from pytorch3d import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
import mcubes
from nnutils.transform import quaternion_mul, dual_quaternion_apply

import sys
sys.path.insert(0, 'third_party')
from ext_utils.flowlib import warp_flow, cat_imgflo 
import chamfer3D.dist_chamfer_3D

def evaluate_mlp(model, xyz_embedded, embed_xyz=None, dir_embedded=None,
                chunk=32*1024, 
                xyz=None,
                code=None, vidid=None):
    """
    embed_xyz: embedding function
    chunk is the point-level chunk divided by number of bins
    """
    B,nbins,_ = xyz_embedded.shape
    out_chunks = []
    for i in range(0, B, chunk):
        embedded = xyz_embedded[i:i+chunk]
        if embed_xyz is not None:
            embedded = embed_xyz(embedded)
        if dir_embedded is not None:
            embedded = torch.cat([embedded,
                       dir_embedded[i:i+chunk]], -1)
        if code is not None:
            code_chunk = code[i:i+chunk]
            if code_chunk.dim() == 2: 
                code_chunk = code_chunk[:,None]
            code_chunk = code_chunk.repeat(1,nbins,1)
            embedded = torch.cat([embedded,code_chunk], -1)
        if xyz is not None:
            xyz_chunk = xyz[i:i+chunk]
        else: xyz_chunk = None
        
        if vidid is not None:
            vidid_chunk = vidid[i:i+chunk] # Npix,1
            vidid_chunk = vidid_chunk.repeat(1,nbins)
            out_chunks += [model(embedded, vidid=vidid_chunk)]
        else:
            out_chunks += [model(embedded)]

    out = torch.cat(out_chunks, 0)
    return out


def bone_transform(bones_in, rts, is_vec=False):
    """ 
    bones_in: 1,B,10  - B gaussian ellipsoids of bone coordinates
    rts: ...,B,3,4    - B ririd transforms
    rts are applied to bone coordinate transforms (left multiply)
    is_vec:     whether rts are stored as r1...9,t1...3 vector form
    """
    B = bones_in.shape[-2]
    bones = bones_in.view(-1,B,10).clone()
    if is_vec:
        rts = rts.view(-1,B,12)
    else:
        rts = rts.view(-1,B,3,4)
    bs = rts.shape[0] 

    center = bones[:,:,:3]
    orient = bones[:,:,3:7] # real first
    scale =  bones[:,:,7:10]
    if is_vec:
        Rmat = rts[:,:,:9].view(-1,B,3,3)
        Tmat = rts[:,:,9:12].view(-1,B,3,1)
    else:
        Rmat = rts[:,:,:3,:3]   
        Tmat = rts[:,:,:3,3:4]   

    # move bone coordinates (left multiply)
    center = Rmat.matmul(center[...,None])[...,0]+Tmat[...,0]
    Rquat = transforms.matrix_to_quaternion(Rmat)
    orient = _quaternion_mul(Rquat, orient)

    scale = scale.repeat(bs,1,1)
    bones = torch.cat([center,orient,scale],-1)
    return bones 

def rtmat_invert(Rmat, Tmat):
    """
    Rmat: ...,3,3   - rotations
    Tmat: ...,3   - translations
    """
    rts = torch.cat([Rmat, Tmat[...,None]],-1)
    rts_i = rts_invert(rts)
    Rmat_i = rts_i[...,:3,:3] # bs, B, 3,3
    Tmat_i = rts_i[...,:3,3]
    return Rmat_i, Tmat_i

def rtk_invert(rtk_in, B):
    """
    rtk_in: ... (rot 1...9, trans 1...3)
    """
    rtk_shape = rtk_in.shape
    rtk_in = rtk_in.view(-1,B,12)# B,12
    rmat=rtk_in[:,:,:9]
    rmat=rmat.view(-1,B,3,3)
    tmat= rtk_in[:,:,9:12]
    rts_fw = torch.cat([rmat,tmat[...,None]],-1)
    rts_fw = rts_fw.view(-1,B,3,4)
    rts_bw = rts_invert(rts_fw)

    rvec = rts_bw[...,:3,:3].reshape(-1,9)
    tvec = rts_bw[...,:3,3] .reshape(-1,3)
    rtk = torch.cat([rvec,tvec],-1).view(rtk_shape)
    return rtk

def rts_invert(rts_in):
    """
    rts: ...,3,4   - B ririd transforms
    """
    rts = rts_in.view(-1,3,4).clone()
    Rmat = rts[:,:3,:3] # bs, B, 3,3
    Tmat = rts[:,:3,3:]
    Rmat_i=Rmat.permute(0,2,1)
    Tmat_i=-Rmat_i.matmul(Tmat)
    rts_i = torch.cat([Rmat_i, Tmat_i],-1)
    rts_i = rts_i.view(rts_in.shape)
    return rts_i

def rtk_to_4x4(rtk):
    """
    rtk: ...,12
    """
    device = rtk.device
    bs = rtk.shape[0]
    zero_one = torch.Tensor([[0,0,0,1]]).to(device).repeat(bs,1)

    rmat=rtk[:,:9]
    rmat=rmat.view(-1,3,3)
    tmat=rtk[:,9:12]
    rts = torch.cat([rmat,tmat[...,None]],-1)
    rts = torch.cat([rts,zero_one[:,None]],1)
    return rts

def rts_compose(rts1, rts2):
    """
    rts ...
    """
    rts_shape = rts1.shape
    device = rts1.device
    rts1 = rts1.view(-1,3,4)# ...,12
    rts2 = rts2.view(-1,3,4)# ...,12

    bs = rts1.shape[0]
    zero_one = torch.Tensor([[0,0,0,1]]).to(device).repeat(bs,1)
    rts1 = torch.cat([rts1,zero_one[:,None]],1)
    rts2 = torch.cat([rts2,zero_one[:,None]],1)
    rts = rts1.matmul(rts2)

    rts = rts[:,:3].view(rts_shape)
    return rts

def rtk_compose(rtk1, rtk2):
    """
    rtk ...
    """
    rtk_shape = rtk1.shape
    rtk1 = rtk1.view(-1,12)# ...,12
    rtk2 = rtk2.view(-1,12)# ...,12

    rts1 = rtk_to_4x4(rtk1)
    rts2 = rtk_to_4x4(rtk2)

    rts = rts1.matmul(rts2)
    rvec = rts[...,:3,:3].reshape(-1,9)
    tvec = rts[...,:3,3].reshape(-1,3)
    rtk = torch.cat([rvec,tvec],-1).view(rtk_shape)
    return rtk

def se3exp_to_vec(se3exp):
    """
    se3exp: B,4,4
    vec: Bx10
    """
    num_bones = se3exp.shape[0]
    device = se3exp.device

    center = se3exp[:,:3,3]
    orient =  se3exp[:,:3,:3]
    orient = transforms.matrix_to_quaternion(orient)
    scale = torch.zeros(num_bones,3).to(device)
    vec = torch.cat([center, orient, scale],-1)
    return vec

def vec_to_sim3(vec):
    """
    vec:      ...,10 / ...,8
    center:   ...,3
    orient:   ...,3,3
    scale:    ...,3 / ...,1
    """
    center = vec[...,:3]
    orient = vec[...,3:7] # real first
    orient = F.normalize(orient, 2,-1)
    orient = transforms.quaternion_to_matrix(orient) # real first
    scale =  vec[...,7:].exp()
    return center, orient, scale

def skinning(xyz, embedding_xyz, bones, 
                    pose_code,  nerf_skin):
    """
    skin: bs,N,B   - skinning matrix
    """
    if nerf_skin is not None and nerf_skin.skin_type == 'cmlp':
        skin = center_mlp_skinning(xyz, embedding_xyz, bones, 
                    pose_code,  nerf_skin)
        dskin=None
    else: 
        skin,dskin = gauss_mlp_skinning(xyz, embedding_xyz, bones, 
                    pose_code,  nerf_skin)

    # truncated softmax
    max_bone=min(skin.shape[2],3)
    topk, indices = skin.topk(max_bone, 2, largest=True)
    skin = torch.zeros_like(skin).fill_(-np.inf)
    skin = skin.scatter(2, indices, topk)

    skin = skin.softmax(2)
    return skin, dskin

def gauss_mlp_skinning(xyz, embedding_xyz, bones, 
                    pose_code,  nerf_skin):
    """
    xyz:        N_rays, ndepth, 3
    bones:      ... nbones, 10
    pose_code:  ...,1, nchannel
    """
    skin = gauss_skinning(bones, xyz) # bs, N, B

    if nerf_skin is not None:
        #TODO hacky way to make code compaitible with noqueryfw
        N_rays = xyz.shape[0]
        if pose_code.dim() == 2 and pose_code.shape[0]!=N_rays: 
            pose_code = pose_code[None].repeat(N_rays, 1,1)
        xyz_embedded = embedding_xyz(xyz)
        dskin = mlp_skinning(nerf_skin, pose_code, xyz_embedded)
        skin += dskin
    return skin, dskin

def center_mlp_skinning(xyz, embedding_xyz, bones, 
                    pose_code,  nerf_skin):
    """
    xyz:        N_rays, ndepth, 3
    bones:      ... nbones, 10
    pose_code:  ...,1, nchannel
    """
    bs,N,_ = xyz.shape
    B = bones.shape[-2]
    if bones.dim()==2: bones = bones[None].repeat(bs,1,1)
    bones = bones.view(-1,B,10)
   
    center, orient, scale = vec_to_sim3(bones) 
    orient = orient.permute(0,1,3,2) # transpose R

    # mahalanobis distance [(p-v)^TR^T]S[R(p-v)]
    # transform a vector to the local coordinate
    mdis = center.view(bs,1,B,3) - xyz.view(bs,N,1,3) # bs,N,B,3
    mdis = axis_rotate(orient.view(bs,1,B,3,3), mdis[...,None])
    mdis = mdis[...,0]
    mdis = mdis / scale.view(bs,1,B,3)
    skin_init = -mdis.pow(2).sum(3) # initial skinning weights

    # embedding input to network
    mdis = mdis.view(-1,3)
    mdis_embedded = mdis
    #mdis_embedded = embedding_xyz(mdis) # bs, N, B, 3

    #TODO hacky way to make code compaitible with noqueryfw
    N_rays = xyz.shape[0]
    if pose_code.dim() == 2 and pose_code.shape[0]!=N_rays: 
        pose_code = pose_code[None].repeat(N_rays, 1,1) # bs, nmlp, 63

    # mlp
    mdis_embedded = mdis_embedded.reshape(bs, N,-1) 
    skin = evaluate_mlp(nerf_skin, mdis_embedded, code = pose_code) # bs,N,B
    ## mmlp
    #mdis_embedded = mdis_embedded.view(bs, N*B, -1) 
    #skin = evaluate_mlp(nerf_skin, mdis_embedded, code = pose_code) # bs,N,B
    #skin = skin.view(bs, N, B)

    skin = skin + skin_init
    return skin

def mlp_skinning(mlp, code, pts_embed):
    """
    code: bs, D          - N D-dimensional pose code
    pts_embed: bs,N,x    - N point positional embeddings
    dskin: bs,N,B        - delta skinning matrix
    """
    if mlp is None:
        dskin = None
    else:
        dskin = evaluate_mlp(mlp, pts_embed, code=code, chunk=8*1024)
    return dskin

def axis_rotate(orient, mdis):
    bs,N,B,_,_ = mdis.shape
    mdis = (orient * mdis.view(bs,N,B,1,3)).sum(4)[...,None] # faster 
    #mdis = orient.matmul(mdis) # bs,N,B,3,1 # slower
    return mdis

def extract_bone_sdf(bones_rst, skin_aux, pts):
    """
    pts: ...,3
    sdf: ...,1
    """
    rescale = skin_aux[1].exp()*0.2
    out_rescale = skin_aux[0].exp()
    shape = pts.shape[:-1]
    pts = pts.view(1,-1,3)
    mdis = pts_bone_distance(bones_rst, pts)
    mdis = mdis[0].sqrt() / rescale
    mdis = mdis.min(-1)[0] # select the closest bone => make this differentiable
    sdf = mdis - 1 # outside: positive
    sdf = sdf.view(shape+(1,))
    sdf = sdf * out_rescale
    return sdf

def pts_bone_distance(bones, pts):
    """
    bone: bs,B,10  - B gaussian ellipsoids
    pts: bs,N,3    - N 3d points, usually N=num points per ray, b~=2034
    mdis: bs,N,B   - mahalanobis distance to bone center
    """
    device = pts.device
    bs,N,_ = pts.shape
    B = bones.shape[-2]
    if bones.dim()==2: bones = bones[None].repeat(bs,1,1)
    bones = bones.view(-1,B,10)
   
    center, orient, scale = vec_to_sim3(bones) 
    orient = orient.permute(0,1,3,2) # transpose R

    # mahalanobis distance [(p-v)^TR^T]S[R(p-v)]
    # transform a vector to the local coordinate
    mdis = center.view(bs,1,B,3) - pts.view(bs,N,1,3) # bs,N,B,3
    mdis = axis_rotate(orient.view(bs,1,B,3,3), mdis[...,None])
    mdis = mdis[...,0]
    mdis = mdis / scale.view(bs,1,B,3)
    mdis = mdis.pow(2)
    mdis = mdis.sum(3) # bs,N,B
    return mdis

def gauss_skinning_chunk(bones, pts):
    """
    bone: bs,B,10  - B gaussian ellipsoids
    pts: bs,N,3    - N 3d points, usually N=num points per ray, b~=2034
    skin: bs,N,B   - skinning matrix
    """
    mdis = pts_bone_distance(bones, pts)
    skin = -mdis
    return skin
    

def gauss_skinning(bones, pts):
    """
    bone: ...,B,10  - B gaussian ellipsoids
    pts: bs,N,3    - N 3d points
    skin: bs,N,B   - skinning matrix
    """
    chunk=4096
    bs,N,_ = pts.shape
    B = bones.shape[-2]
    if bones.dim()==2: bones = bones[None].repeat(bs,1,1)
    bones = bones.view(-1,B,10)

    skin = []
    for i in range(0,bs,chunk):
        skin_chunk = gauss_skinning_chunk(bones[i:i+chunk], pts[i:i+chunk])
        skin.append( skin_chunk )
    skin = torch.cat(skin,0)
    return skin
   
@torch.jit.script
def _quaternion_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.
    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.
    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)
 
def quat_lbs(rts, pts, skin):
    bs = rts.shape[0]
    B = rts.shape[-3]
    N = pts.shape[-2]
    pts = pts.view(-1,N,3)
    rts = rts.view(-1,B,3,4)
    Rmat = rts[:,:,:3,:3] # bs, B, 3,3
    Tmat = rts[:,:,:3,3]
    device = Tmat.device
    #TODO do dual quat
    #TODO ensure in the same hemisphere (+ real)
    qr = transforms.matrix_to_quaternion(Rmat)
    qr = qr[:,None].repeat(1,N,1,1).reshape(-1,B,4)
    pivot = skin.argmax(-1).view(-1,1,1).repeat(1,1,4)
    sign = (torch.gather(qr, 1, pivot) * qr).sum(-1) > 0
    sign = sign[...,None].float()*2-1
    qr = sign * qr
    qr = qr.reshape(bs,N,B,4)
    #TODO make sure blending quats on the same hemisphere
    qd = torch.cat([torch.zeros_like(Tmat[...,:1]), Tmat],-1)[:,None] #TODO remove tmat
    qd = 0.5* _quaternion_mul(qd, qr) # failed prev due to quat norm
    #TODO einsum
    qr_w = (skin[...,None] * qr).sum(2) # bnk1,bnk4->bnk4->bn4
    qd_w = (skin[...,None] * qd).sum(2) 
    #TODO merge with dual_quat_apply()
    qr_mag_inv = qr_w.norm(p=2, dim=-1, keepdim=True).reciprocal()
    qr_w = qr_w * qr_mag_inv
    qd_w = qd_w * qr_mag_inv
    Rmat_w = transforms.quaternion_to_matrix(qr_w)
    conj_qr_w = torch.cat((qr_w[..., :1], -qr_w[..., 1:]), -1)
    Tmat_w = 2*_quaternion_mul(qd_w, conj_qr_w)[..., 1:]
    return Rmat_w, Tmat_w

def quat_lbs_v2(rts, pts, skin):
    """
    bone: bs,B,10   - B gaussian ellipsoids
    rts: bs,B,3,4   - B ririd transforms, applied to bone coordinates (points attached to bones in world coords)
    pts: bs,N,3     - N 3d points
    skin: bs,N,B   - skinning matrix
    apply rts to bone coordinates, while computing blending globally
    """
    bs = rts.shape[0]
    B = rts.shape[-3]
    N = pts.shape[-2]
    pts = pts.view(-1,N,3)
    rts = rts.view(-1,B,3,4)
    Rmat = rts[:,:,:3,:3] # bs, B, 3,3
    Tmat = rts[:,:,:3,3]
    device = Tmat.device
    
    # convert
    qr = transforms.matrix_to_quaternion(Rmat)

    ## sign1
    #sign = (qr[...,:1]>0).float()*2-1
    #qr = sign * qr # make sure blending quats on the same hemisphere
    #qd = 0.5* quaternion_mul(Tmat, qr)
    #qr_w = torch.einsum('bnk,bkl->bnl', skin, qr)
    #qd_w = torch.einsum('bnk,bkl->bnl', skin, qd)
    
    # sign2
    qr = qr[:,None].repeat(1,N,1,1).reshape(-1,B,4)
    pivot = skin.argmax(-1).view(-1,1,1).repeat(1,1,4)
    sign = (torch.gather(qr, 1, pivot) * qr).sum(-1) > 0
    sign = sign[...,None].float()*2-1
    qr = sign * qr # make sure blending quats on the same hemisphere
    qr = qr.reshape(bs,N,B,4)
    Tmat = Tmat[:,None].repeat(1,N,1,1)
    qd = 0.5* quaternion_mul(Tmat, qr)
    qr_w = torch.einsum('bnk,bnkl->bnl', skin, qr)
    qd_w = torch.einsum('bnk,bnkl->bnl', skin, qd)

    qr_mag_inv = qr_w.norm(p=2, dim=-1, keepdim=True).reciprocal()
    qr_w = qr_w * qr_mag_inv
    qd_w = qd_w * qr_mag_inv
    # apply
    pts = dual_quaternion_apply((qr_w, qd_w), pts)
    return pts

def blend_skinning_chunk(bones, rts, skin, pts):
#def blend_skinning(bones, rts, skin, pts):
    """
    bone: bs,B,10   - B gaussian ellipsoids
    rts: bs,B,3,4   - B ririd transforms, applied to bone coordinates (points attached to bones in world coords)
    pts: bs,N,3     - N 3d points
    skin: bs,N,B   - skinning matrix
    apply rts to bone coordinates, while computing blending globally
    """
    pts = quat_lbs_v2(rts, pts, skin)

    #bs = rts.shape[0]
    #B = rts.shape[-3]
    #N = pts.shape[-2]
    #pts = pts.view(-1,N,3)
    #rts = rts.view(-1,B,3,4)
    #Rmat = rts[:,:,:3,:3] # bs, B, 3,3
    #Tmat = rts[:,:,:3,3]
    #device = Tmat.device

    ### Gi=sum(wbGb), V=RV+T
    ##Rmat_w, Tmat_w = quat_lbs(rts, pts, skin)
    ##pts = Rmat_w.matmul(pts[...,None]) + Tmat_w[...,None] 
    ##pts = pts[...,0]
    #
    #Rmat_w = (skin[...,None,None] * Rmat[:,None]).sum(2) # bs,N,B,3
    #Tmat_w = (skin[...,None] * Tmat[:,None]).sum(2) # bs,N,B,3
    #pts = Rmat_w.matmul(pts[...,None]) + Tmat_w[...,None] 
    #pts = pts[...,0]
    return pts

def blend_skinning(bones, rts, skin, pts):
    """
    bone: bs,B,10   - B gaussian ellipsoids
    rts: bs,B,3,4   - B ririd transforms, applied to bone coordinates
    pts: bs,N,3     - N 3d points
    skin: bs,N,B   - skinning matrix
    apply rts to bone coordinates, while computing blending globally
    """
    chunk=4096
    B = rts.shape[-3]
    N = pts.shape[-2]
    bones = bones.view(-1,B,10)
    pts = pts.view(-1,N,3)
    rts = rts.view(-1,B,3,4)
    bs = pts.shape[0]

    pts_out = []
    for i in range(0,bs,chunk):
        pts_chunk = blend_skinning_chunk(bones[i:i+chunk], rts[i:i+chunk], 
                                          skin[i:i+chunk], pts[i:i+chunk])
        pts_out.append(pts_chunk)
    pts = torch.cat(pts_out,0)
    return pts

def lbs(bones, rts_fw, skin, xyz_in, backward=True):
    """
    bones: bs,B,10       - B gaussian ellipsoids indicating rest bone coordinates
    rts_fw: bs,B,12       - B rigid transforms, applied to the rest bones
    xyz_in: bs,N,3       - N 3d points after transforms in the root coordinates
    """
    B = bones.shape[-2]
    N = xyz_in.shape[-2]
    bs = rts_fw.shape[0]
    bones = bones.view(-1,B,10)
    xyz_in = xyz_in.view(-1,N,3)
    rts_fw = rts_fw.view(-1,B,12)# B,12
    rmat=rts_fw[:,:,:9]
    rmat=rmat.view(bs,B,3,3)
    tmat= rts_fw[:,:,9:12]
    rts_fw = torch.cat([rmat,tmat[...,None]],-1)
    rts_fw = rts_fw.view(-1,B,3,4)

    if backward:
        bones_dfm = bone_transform(bones, rts_fw) # bone coordinates after deform
        rts_bw = rts_invert(rts_fw)
        xyz = blend_skinning(bones_dfm, rts_bw, skin, xyz_in)
    else:
        xyz = blend_skinning(bones.repeat(bs,1,1), rts_fw, skin, xyz_in)
        bones_dfm = bone_transform(bones, rts_fw) # bone coordinates after deform
    return xyz, bones_dfm

def obj_to_cam(in_verts, Rmat, Tmat):
    """
    verts: ...,N,3
    Rmat:  ...,3,3
    Tmat:  ...,3 
    """
    verts = in_verts.clone()
    if verts.dim()==2: verts=verts[None]
    verts = verts.view(-1,verts.shape[1],3)
    Rmat = Rmat.view(-1,3,3).permute(0,2,1) # left multiply
    Tmat = Tmat.view(-1,1,3)
    
    verts =  verts.matmul(Rmat) + Tmat 
    verts = verts.reshape(in_verts.shape)
    return verts

def obj2cam_np(pts, Rmat, Tmat):
    """
    a wrapper for numpy array
    pts: ..., 3
    Rmat: 1,3,3
    Tmat: 1,3,3
    """
    pts_shape = pts.shape
    pts = torch.Tensor(pts).cuda().reshape(1,-1,3)
    pts = obj_to_cam(pts, Rmat,Tmat)
    return pts.view(pts_shape).cpu().numpy()

    
def K2mat(K):
    """
    K: ...,4
    """
    K = K.view(-1,4)
    device = K.device
    bs = K.shape[0]

    Kmat = torch.zeros(bs, 3, 3, device=device)
    Kmat[:,0,0] = K[:,0]
    Kmat[:,1,1] = K[:,1]
    Kmat[:,0,2] = K[:,2]
    Kmat[:,1,2] = K[:,3]
    Kmat[:,2,2] = 1
    return Kmat

def mat2K(Kmat):
    """
    Kmat: ...,3,3
    """
    shape=Kmat.shape[:-2]
    Kmat = Kmat.view(-1,3,3)
    device = Kmat.device
    bs = Kmat.shape[0]

    K = torch.zeros(bs, 4, device=device)
    K[:,0] = Kmat[:,0,0]
    K[:,1] = Kmat[:,1,1]
    K[:,2] = Kmat[:,0,2]
    K[:,3] = Kmat[:,1,2]
    K = K.view(shape+(4,))
    return K

def Kmatinv(Kmat):
    """
    Kmat: ...,3,3
    """
    K = mat2K(Kmat)
    Kmatinv = K2inv(K)
    Kmatinv = Kmatinv.view(Kmat.shape)
    return Kmatinv

def K2inv(K):
    """
    K: ...,4
    """
    K = K.view(-1,4)
    device = K.device
    bs = K.shape[0]

    Kmat = torch.zeros(bs, 3, 3, device=device)
    Kmat[:,0,0] = 1./K[:,0]
    Kmat[:,1,1] = 1./K[:,1]
    Kmat[:,0,2] = -K[:,2]/K[:,0]
    Kmat[:,1,2] = -K[:,3]/K[:,1]
    Kmat[:,2,2] = 1
    return Kmat

def pinhole_cam(in_verts, K):
    """
    in_verts: ...,N,3
    K:        ...,4
    verts:    ...,N,3 in (x,y,Z)
    """
    verts = in_verts.clone()
    if verts.dim()==2: verts=verts[None]
    verts = verts.view(-1,verts.shape[1],3)
    K = K.view(-1,4)

    Kmat = K2mat(K)
    Kmat = Kmat.permute(0,2,1)

    verts = verts.matmul(Kmat)
    verts_z = verts[:,:,2:3]
    verts_xy = verts[:,:,:2] / (1e-6+verts_z) # deal with neg z
    
    verts = torch.cat([verts_xy,verts_z],-1)
    verts = verts.reshape(in_verts.shape)
    return verts

def render_color(renderer, in_verts, faces, colors, texture_type='vertex'):
    """
    verts in ndc
    in_verts: ...,N,3/4
    faces: ...,N,3
    rendered: ...,4,...
    """
    import soft_renderer as sr
    verts = in_verts.clone()
    verts = verts.view(-1,verts.shape[-2],3)
    faces = faces.view(-1,faces.shape[-2],3)
    if texture_type=='vertex':  colors = colors.view(-1,colors.shape[-2],3)
    elif texture_type=='surface': colors = colors.view(-1,colors.shape[1],colors.shape[2],3)
    device=verts.device

    offset = torch.Tensor( renderer.transform.transformer._eye).to(device)[np.newaxis,np.newaxis]
    verts_pre = verts[:,:,:3]-offset
    verts_pre[:,:,1] = -1*verts_pre[:,:,1]  # pre-flip
    rendered = renderer.render_mesh(sr.Mesh(verts_pre,faces,textures=colors,texture_type=texture_type))
    return rendered

def render_flow(renderer, verts, faces, verts_n):
    """
    rasterization
    verts in ndc
    verts: ...,N,3/4
    verts_n: ...,N,3/4
    faces: ...,N,3
    """
    verts = verts.view(-1,verts.shape[1],3)
    verts_n = verts_n.view(-1,verts_n.shape[1],3)
    faces = faces.view(-1,faces.shape[1],3)
    device=verts.device

    rendered_ndc_n = render_color(renderer, verts, faces, verts_n)
    _,_,h,w = rendered_ndc_n.shape
    rendered_sil = rendered_ndc_n[:,-1]

    ndc = np.meshgrid(range(w), range(h))
    ndc = torch.Tensor(ndc).to(device)[None]
    ndc[:,0] = ndc[:,0]*2 / (w-1) - 1
    ndc[:,1] = ndc[:,1]*2 / (h-1) - 1

    flow = rendered_ndc_n[:,:2] - ndc
    flow = flow.permute(0,2,3,1) # x,h,w,2
    flow = torch.cat([flow, rendered_sil[...,None]],-1)

    flow[rendered_sil<1]=0.
    flow[...,-1]=0. # discard the last channel
    return flow

def force_type(varlist):
    for i in range(len(varlist)):
        varlist[i] = varlist[i].type(varlist[0].dtype)
    return varlist

def tensor2array(tdict):
    adict={}
    for k,v in tdict.items():
        adict[k] = v.detach().cpu().numpy()
    return adict

def array2tensor(adict, device='cpu'):
    tdict={}
    for k,v in adict.items():
        try: 
            tdict[k] = torch.Tensor(v)
            if device != 'cpu': tdict[k] = tdict[k].to(device)
        except: pass # trimesh object
    return tdict

def raycast(xys, Rmat, Tmat, Kinv, near_far):
    """
    assuming xys and Rmat have same num of bs
    xys: bs, N, 3
    Rmat:bs, ...,3,3 
    Tmat:bs, ...,3, camera to root coord transform 
    Kinv:bs, ...,3,3 
    near_far:bs,2
    """
    Rmat, Tmat, Kinv, xys = force_type([Rmat, Tmat, Kinv, xys])
    Rmat = Rmat.view(-1,3,3)
    Tmat = Tmat.view(-1,1,3)
    Kinv = Kinv.view(-1,3,3)
    bs,nsample,_ = xys.shape
    device = Rmat.device

    xy1s = torch.cat([xys, torch.ones_like(xys[:,:,:1])],2)
    xyz3d = xy1s.matmul(Kinv.permute(0,2,1))
    ray_directions = xyz3d.matmul(Rmat)  # transpose -> right multiply
    ray_origins = -Tmat.matmul(Rmat) # transpose -> right multiply

    if near_far is not None:
        znear= (torch.ones(bs,nsample,1).to(device) * near_far[:,0,None,None]) 
        zfar = (torch.ones(bs,nsample,1).to(device) * near_far[:,1,None,None]) 
    else:
        lbound, ubound=[-1.5,1.5]

        znear= Tmat[:,:,-1:].repeat(1,nsample,1)+lbound
        zfar = Tmat[:,:,-1:].repeat(1,nsample,1)+ubound
        znear[znear<1e-5]=1e-5

    ray_origins = ray_origins.repeat(1,nsample,1)

    rmat_vec = Rmat.reshape(-1,1,9)
    tmat_vec = Tmat.reshape(-1,1,3)
    kinv_vec = Kinv.reshape(-1,1,9)
    rtk_vec = torch.cat([rmat_vec, tmat_vec, kinv_vec],-1) # x,21
    rtk_vec = rtk_vec.repeat(1,nsample,1)

    rays={'rays_o': ray_origins, 
          'rays_d': ray_directions,
          'near': znear,
          'far': zfar,
          'rtk_vec': rtk_vec,
          'xys': xys,
          'nsample': nsample,
          'bs': bs,
          'xy_uncrop': xyz3d[...,:2],
          }
    return rays

def sample_xy(img_size, bs, nsample, device, return_all=False, lineid=None):
    """
    rand_inds:  bs, ns
    xys:        bs, ns, 2
    """
    xygrid = np.meshgrid(range(img_size), range(img_size))  # w,h->hxw
    xygrid = np.array(xygrid)
    xygrid = torch.Tensor(xygrid).to(device)  # (x,y)
    xygrid = xygrid.permute(1,2,0).reshape(1,-1,2)  # 1,..., 2
    
    if return_all:
        xygrid = xygrid.repeat(bs,1,1)                  # bs,..., 2
        nsample = xygrid.shape[1]
        rand_inds=torch.Tensor(range(nsample))
        rand_inds=rand_inds[None].repeat(bs,1)
        xys = xygrid
    else:
        if lineid is None:
            probs = torch.ones(img_size**2).to(device) # 512*512 vs 128*64
            rand_inds = torch.multinomial(probs, bs*nsample, replacement=False)
            rand_inds = rand_inds.view(bs,nsample)
            xys = torch.stack([xygrid[0][rand_inds[i]] for i in range(bs)],0) # bs,ns,2
        else:
            probs = torch.ones(img_size).to(device) # 512*512 vs 128*64
            rand_inds = torch.multinomial(probs, bs*nsample, replacement=True)
            rand_inds = rand_inds.view(bs,nsample)
            xys = torch.stack([xygrid[0][rand_inds[i]] for i in range(bs)],0) # bs,ns,2
            xys[...,1] = xys[...,1] + lineid[:,None]
   
    rand_inds = rand_inds.long()
    return rand_inds, xys

def chunk_rays(rays,start,delta):
    """
    rays: a dictionary
    """
    rays_chunk = {}
    for k,v in rays.items():
        if torch.is_tensor(v):
            v = v.view(-1, v.shape[-1])
            rays_chunk[k] = v[start:start+delta]
    return rays_chunk
        

def generate_bones(num_bones_x, num_bones, bound, device):
    """
    num_bones_x: bones along one direction
    bones: x**3,9
    """
    center =  torch.linspace(-bound, bound, num_bones_x).to(device)
    center =torch.meshgrid(center, center, center)
    center = torch.stack(center,0).permute(1,2,3,0).reshape(-1,3)
    center = center[:num_bones]
    
    orient =  torch.Tensor([[1,0,0,0]]).to(device)
    orient = orient.repeat(num_bones,1)
    scale = torch.zeros(num_bones,3).to(device)
    bones = torch.cat([center, orient, scale],-1)
    return bones

def reinit_bones(model, mesh, num_bones):
    """
    update the data of bones and nerf_body_rts.rgb without add new parameters
    num_bones: number of bones on the surface
    mesh: trimesh
    warning: ddp does not support adding/deleting parameters after construction
    """
    #TODO find another way to add/delete bones
    from kmeans_pytorch import kmeans
    device = model.nerf_body_rts.linear_final.weight.device
    points = torch.Tensor(mesh.vertices).to(device)
    
    if points.shape[0]<100:
        bound = model.latest_vars['obj_bound']
        bound = torch.Tensor(bound)[None]
        center = torch.rand(num_bones, 3) *  bound*2 - bound
    else:
        _, center = kmeans(X=points, num_clusters=num_bones, iter_limit=100,
                        tqdm_flag=False, distance='euclidean', device=device)
    center=center.to(device)
    orient =  torch.Tensor([[1,0,0,0]]).to(device)
    orient = orient.repeat(num_bones,1)
    scale = -3.5*torch.ones(num_bones,3).to(device)
    bones = torch.cat([center, orient, scale],-1)

    # reinit
    rthead = model.nerf_body_rts.linear_final
    num_in = rthead.weight.shape[1]
    num_out_channels = model.nerf_body_rts.out_channels

    rthead = nn.Linear(num_in, num_out_channels).to(device)
    torch.nn.init.xavier_uniform_(rthead.weight, gain=0.5)
    torch.nn.init.zeros_(rthead.bias)

    bias_reinit =   rthead.bias.data
    weight_reinit=rthead.weight.data
    model.nerf_body_rts.linear_final.bias.data[:num_out_channels] = bias_reinit
    model.nerf_body_rts.linear_final.weight.data[:num_out_channels] = weight_reinit
    
    bones,_ = zero_to_rest_bone(model, bones, inverse=True)
    model.bones.data = bones
    model.nerf_models['bones'] = model.bones
    return

def zero_to_rest_bone(model, bones_rst, inverse=False, vid=None):
    """
    Returns 
        bones at rest position for vid/canonical bone,
        transformation from canonical bone/0 pose to vid bone/rest pose
    apply rest pose code to 0 configuration bones 
    vid: bs, if none, return canonical skeleton
    inverse, whether from 0 to rest or from rest to 0, not used by skeleton
    """
    if model.opts.pre_skel!="":
        # here bone centers are dependent on joints
        # rest bones are queried
        from nnutils.urdf_utils import compute_bone_from_joint
        bones = compute_bone_from_joint(model, is_init=False, vid=vid)
        body_code = model.rest_pose_code.weight
        if vid is not None:
            vid = vid.view(-1)
            bs = len(vid)
            body_code = body_code.repeat(bs,1)
        joints = model.nerf_body_rts.forward_decode(body_code, vid)
        return bones, joints
    # bones=>bones_rst
    bones_rst = bones_rst.clone()
    rest_pose_code =  model.rest_pose_code
    rest_pose_code = rest_pose_code(torch.Tensor([0]).long().to(bones_rst.device))
    rts_head = model.nerf_body_rts
    bone_rts_rst = rts_head.forward_decode(rest_pose_code)[0] # 1,B*12
    if inverse:
        bone_rts_rst = rtk_invert(bone_rts_rst, model.opts.num_bones)
    bones_rst = bone_transform(bones_rst, bone_rts_rst, is_vec=True)[0] 
    return bones_rst, bone_rts_rst

def zero_to_rest_dpose(opts, bone_rts_fw, bone_rts_rst):
    """
    bone_rts_fw: canonical/0 -> vid bone/t
    bone_rts_rst: canonical/0 -> vid /raw bone/rest
    output transforms: canonical/rest -> vid bone/t
    # delta rts
    bone_rts_fw: bs, N, 12: 0->t, None->vid
    bone_rts_rst:1/bs, 1, xxx : 0->rest, None
    """
    bone_rts_fw = bone_rts_fw.clone()
    rts_shape = bone_rts_fw.shape
    bone_shape = bone_rts_rst.shape
    bone_rts_rst_inv = rtk_invert(bone_rts_rst, opts.num_bones)
    
    bone_rts_rst_inv = bone_rts_rst_inv.repeat(rts_shape[0]//bone_shape[0],rts_shape[1],1)
    bone_rts_fw =     rtk_compose(bone_rts_fw, bone_rts_rst_inv) # shoule be rest->0->t
    return bone_rts_fw

def warp_bw(opts, model, rt_dict, query_xyz_chunk, embedid):
    """
    only used in mesh extraction
    embedid: embedding id
    """
    chunk = query_xyz_chunk.shape[0]
    query_time = torch.ones(chunk,1).to(model.device)*embedid
    query_time = query_time.long()
    if opts.lbs:
        # backward skinning
        bones_rst = model.bones
        bone_rts_fw = model.nerf_body_rts(query_time)
        # update bones
        bones_rst, bone_rts_rst = zero_to_rest_bone(model, bones_rst)
        #vidid,_ = fid_reindex(embedid, model.num_vid, model.data_offset)
        #bones_rst, bone_rts_rst = zero_to_rest_bone(model, bones_rst, vid=vidid)
        bone_rts_fw = zero_to_rest_dpose(opts, bone_rts_fw, bone_rts_rst)

        query_xyz_chunk = query_xyz_chunk[:,None]

        if opts.nerf_skin:
            nerf_skin = model.nerf_skin
        else:
            nerf_skin = None
        time_embedded = model.pose_code(query_time)
        bones_dfm = bone_transform(bones_rst, bone_rts_fw, is_vec=True)

        skin_backward,_ = skinning(query_xyz_chunk, model.embedding_xyz,
                   bones_dfm, time_embedded, nerf_skin)

        query_xyz_chunk,bones_dfm = lbs(bones_rst, 
                                      bone_rts_fw,
                                      skin_backward,
                                      query_xyz_chunk)

        query_xyz_chunk = query_xyz_chunk[:,0]
        rt_dict['bones'] = bones_dfm 
    return query_xyz_chunk, rt_dict
        
def warp_fw(opts, model, rt_dict, vertices, embedid, robot_render=True):
    """
    vertices: -1,3
    embedid: int or n,1
    only used in mesh extraction
    """
    num_pts = vertices.shape[0]
    query_time = torch.ones(num_pts,1).long().to(model.device)*embedid
    pts_can=torch.Tensor(vertices).to(model.device)
    # make sure rts and pts have same shape
    if num_pts==1: pts_can = pts_can.repeat(query_time.shape[0],1) 
    if opts.lbs:
        # forward skinning
        pts_can = pts_can[:,None]
        bones_rst = model.bones
        bone_rts_fw = model.nerf_body_rts(query_time)
        bones_rst, bone_rts_rst = zero_to_rest_bone(model, bones_rst)
        #vidid,_ = fid_reindex(embedid, model.num_vid, model.data_offset)
        #bones_rst, bone_rts_rst = zero_to_rest_bone(model, bones_rst, vid=vidid)
        bone_rts_fw = zero_to_rest_dpose(opts, bone_rts_fw, bone_rts_rst)
        
        if opts.nerf_skin:
            nerf_skin = model.nerf_skin
        else:
            nerf_skin = None
        rest_pose_code =  model.rest_pose_code
        rest_pose_code = rest_pose_code(torch.Tensor([0]).long().to(bones_rst.device))
        if opts.mlp_deform:
            # N,1,C vs N,d,1,3
            pts_can = model.mlp_deform.forward(model.dfm_code(query_time)[:,None], 
                                        pts_can[:,:,None])[:,:,0]
        skin_forward,_ = skinning(pts_can, model.embedding_xyz, bones_rst, 
                            rest_pose_code, nerf_skin)

        pts_dfm,bones_dfm = lbs(bones_rst, bone_rts_fw, skin_forward, 
                pts_can,backward=False)
        #pts_dfm = pts_can
        pts_dfm = pts_dfm[:,0]
        rt_dict['bones'] = bones_dfm
        rt_dict['se3'] = bone_rts_fw

        if opts.pre_skel!="" and robot_render:
            from nnutils.urdf_utils import visualize_joints
            # save articulated joints
            robot_save_path = 'tmp/robot-%05d.jpg'%(int(query_time[0,0]))
            joints, angles, skel_rendered, robot_mesh, kps = visualize_joints(model,
                    query_time=query_time[:1], robot_save_path=robot_save_path)
            rt_dict['skel_render'] = skel_rendered
            rt_dict['joints'] = joints
            rt_dict['angles'] = angles
            rt_dict['kps'] = kps

    vertices = pts_dfm.cpu().numpy()
    return vertices, rt_dict
    
def canonical2ndc(model, dp_canonical_pts, rtk, kaug, embedid):
    """
    dp_canonical_pts: 5004,3, pts in the canonical space of each video
    dp_px: bs, 5004, 3
    """
    Rmat = rtk[:,:3,:3]
    Tmat = rtk[:,:3,3]
    Kmat = K2mat(rtk[:,3,:])
    Kaug = K2inv(kaug) # p = Kaug Kmat P
    Kinv = Kmatinv(Kaug.matmul(Kmat))
    K = mat2K(Kmatinv(Kinv))
    bs = Kinv.shape[0]
    npts = dp_canonical_pts.shape[0]

    # projection
    dp_canonical_pts = dp_canonical_pts[None]
    dp_deformed_pts = dp_canonical_pts.repeat(bs,1,1)
    dp_cam_pts = obj_to_cam(dp_deformed_pts, Rmat, Tmat) 
    dp_px = pinhole_cam(dp_cam_pts,K)
    return dp_px 

def get_near_far(near_far, vars_np, tol_fac=1.5, pts=None):
    """
    pts:        point coordinate N,3
    near_far:   near and far plane M,2
    rtk:        object to camera transform, M,4,4
    idk:        indicator of obsered or not M
    tol_fac     tolerance factor
    """
    if pts is None:
        #pts = vars_np['mesh_rest'].vertices
        # turn points to bounding box
        pts = trimesh.bounds.corners(vars_np['mesh_rest'].bounds)

    device = near_far.device
    rtk = torch.Tensor(vars_np['rtk']).to(device)
    idk = torch.Tensor(vars_np['idk']).to(device)

    pts = pts_to_view(pts, rtk, device)

    pmax = pts[...,-1].max(-1)[0]
    pmin = pts[...,-1].min(-1)[0]
    delta = (pmax - pmin)*(tol_fac-1)

    near= pmin-delta
    far = pmax+delta

    near_far[idk==1,0] = torch.clamp(near[idk==1], min=1e-3)
    near_far[idk==1,1] = torch.clamp( far[idk==1], min=1e-3)
    return near_far

def pts_to_view(pts, rtk, device):
    """
    object to camera coordinates
    pts:        point coordinate N,3
    rtk:        object to camera transform, M,4,4
    idk:        indicator of obsered or not M
    """
    M = rtk.shape[0]
    out_pts = []
    chunk=100
    for i in range(0,M,chunk):
        rtk_sub = rtk[i:i+chunk]
        pts_sub = torch.Tensor(np.tile(pts[None],
                        (len(rtk_sub),1,1))).to(device) # M,N,3
        pts_sub = obj_to_cam(pts_sub,  rtk_sub[:,:3,:3], 
                                       rtk_sub[:,:3,3])
        pts_sub = pinhole_cam(pts_sub, rtk_sub[:,3])
        out_pts.append(pts_sub)
    out_pts = torch.cat(out_pts, 0)
    return out_pts

def compute_point_visibility(pts, vars_np, device):
    """
    pts:        point coordinate N,3
    rtk:        object to camera transform, M,4,4
    idk:        indicator of obsered or not M
    **deprecated** due to K vars_tensor['rtk'] may not be consistent
    """
    vars_tensor = array2tensor(vars_np, device=device)
    rtk = vars_tensor['rtk']
    idk = vars_tensor['idk']
    vis = vars_tensor['vis']
    
    pts = pts_to_view(pts, rtk, device) # T, N, 3
    h,w = vis.shape[1:]

    vis = vis[:,None]
    xy = pts[:,None,:,:2] 
    xy[...,0] = xy[...,0]/w*2 - 1
    xy[...,1] = xy[...,1]/h*2 - 1

    # grab the visibility value in the mask and sum over frames
    vis = F.grid_sample(vis, xy)[:,0,0]
    vis = (idk[:,None]*vis).sum(0)
    vis = (vis>0).float() # at least seen in one view
    return vis


def near_far_to_bound(near_far):
    """
    near_far: T, 2 on cuda
    bound: float
    this can only be used for a single video (and for approximation)
    """
    bound=(near_far[:,1]-near_far[:,0]).mean() / 2
    bound = bound.detach().cpu().numpy()
    return bound


def rot_angle(mat):
    """
    rotation angle of rotation matrix 
    rmat: ..., 3,3
    """
    eps=1e-4
    cos = (  mat[...,0,0] + mat[...,1,1] + mat[...,2,2] - 1 )/2
    cos = cos.clamp(-1+eps,1-eps)
    angle = torch.acos(cos)
    return angle

def match2coords(match, w_rszd):
    tar_coord = torch.cat([match[:,None]%w_rszd, match[:,None]//w_rszd],-1)
    tar_coord = tar_coord.float()
    return tar_coord
    
def match2flo(match, w_rszd, img_size, warp_r, warp_t, device):
    ref_coord = sample_xy(w_rszd, 1, 0, device, return_all=True)[1].view(-1,2)
    ref_coord = ref_coord.matmul(warp_r[:2,:2]) + warp_r[None,:2,2]
    tar_coord = match2coords(match, w_rszd)
    tar_coord = tar_coord.matmul(warp_t[:2,:2]) + warp_t[None,:2,2]

    flo_dp = (tar_coord - ref_coord) / img_size * 2 # [-2,2]
    flo_dp = flo_dp.view(w_rszd, w_rszd, 2)
    flo_dp = flo_dp.permute(2,0,1)

    xygrid = sample_xy(w_rszd, 1, 0, device, return_all=True)[1] # scale to img_size
    xygrid = xygrid * float(img_size/w_rszd)
    warp_r_inv = Kmatinv(warp_r)
    xygrid = xygrid.matmul(warp_r_inv[:2,:2]) + warp_r_inv[None,:2,2]
    xygrid = xygrid / w_rszd * 2 - 1 
    flo_dp = F.grid_sample(flo_dp[None], xygrid.view(1,w_rszd,w_rszd,2))[0]
    return flo_dp

def compute_flow_cse(cse_a,cse_b, warp_a, warp_b, img_size):
    """
    compute the flow between two frames under cse feature matching
    assuming two feature images have the same dimension (also rectangular)
    cse:        16,h,w, feature image
    flo_dp:     2,h,w
    """
    _,_,w_rszd = cse_a.shape
    hw_rszd = w_rszd*w_rszd
    device = cse_a.device

    cost = (cse_b[:,None,None] * cse_a[...,None,None]).sum(0)
    _,match_a = cost.view(hw_rszd, hw_rszd).max(1)
    _,match_b = cost.view(hw_rszd, hw_rszd).max(0)

    flo_a = match2flo(match_a, w_rszd, img_size, warp_a, warp_b, device)
    flo_b = match2flo(match_b, w_rszd, img_size, warp_b, warp_a, device)
    return flo_a, flo_b

def compute_flow_geodist(dp_refr,dp_targ, geodists):
    """
    compute the flow between two frames under geodesic distance matching
    dps:        h,w, canonical surface mapping index
    geodists    N,N, distance matrix
    flo_dp:     2,h,w
    """
    h_rszd,w_rszd = dp_refr.shape
    hw_rszd = h_rszd*w_rszd
    device = dp_refr.device
    chunk = 1024

    # match: hw**2
    match = torch.zeros(hw_rszd).to(device)
    for i in range(0,hw_rszd,chunk):
        chunk_size = len(dp_refr.view(-1,1)[i:i+chunk] )
        dp_refr_sub = dp_refr.view(-1,1)[i:i+chunk].repeat(1,hw_rszd).view(-1,1)
        dp_targ_sub = dp_targ.view(1,-1)        .repeat(chunk_size,1).view(-1,1)
        match_sub = geodists[dp_refr_sub, dp_targ_sub]
        dis_geo_sub,match_sub = match_sub.view(-1, hw_rszd).min(1)
        #match_sub[dis_geo_sub>0.1] = 0
        match[i:i+chunk] = match_sub

    # cx,cy
    tar_coord = match2coords(match, w_rszd)
    ref_coord = sample_xy(w_rszd, 1, 0, device, return_all=True)[1].view(-1,2)
    ref_coord = ref_coord.view(h_rszd, w_rszd, 2)
    tar_coord = tar_coord.view(h_rszd, w_rszd, 2)
    flo_dp = (tar_coord - ref_coord) / w_rszd * 2 # [-2,2]
    match = match.view(h_rszd, w_rszd)
    flo_dp[match==0] = 0
    flo_dp = flo_dp.permute(2,0,1)
    return flo_dp

def compute_flow_geodist_old(dp_refr,dp_targ, geodists):
    """
    compute the flow between two frames under geodesic distance matching
    dps:        h,w, canonical surface mapping index
    geodists    N,N, distance matrix
    flo_dp:     2,h,w
    """
    h_rszd,w_rszd = dp_refr.shape
    hw_rszd = h_rszd*w_rszd
    device = dp_refr.device
    dp_refr = dp_refr.view(-1,1).repeat(1,hw_rszd).view(-1,1)
    dp_targ = dp_targ.view(1,-1).repeat(hw_rszd,1).view(-1,1)

    match = geodists[dp_refr, dp_targ]
    dis_geo,match = match.view(hw_rszd, hw_rszd).min(1)
    #match[dis_geo>0.1] = 0

    # cx,cy
    tar_coord = match2coords(match, w_rszd)
    ref_coord = sample_xy(w_rszd, 1, 0, device, return_all=True)[1].view(-1,2)
    ref_coord = ref_coord.view(h_rszd, w_rszd, 2)
    tar_coord = tar_coord.view(h_rszd, w_rszd, 2)
    flo_dp = (tar_coord - ref_coord) / w_rszd * 2 # [-2,2]
    match = match.view(h_rszd, w_rszd)
    flo_dp[match==0] = 0
    flo_dp = flo_dp.permute(2,0,1)
    return flo_dp



def fb_flow_check(flo_refr, flo_targ, img_refr, img_targ, dp_thrd, 
                    save_path=None):
    """
    apply forward backward consistency check on flow fields
    flo_refr: 2,h,w forward flow
    flo_targ: 2,h,w backward flow
    fberr:    h,w forward backward error
    """
    h_rszd, w_rszd = flo_refr.shape[1:]
    # clean up flow
    flo_refr = flo_refr.permute(1,2,0).cpu().numpy()
    flo_targ = flo_targ.permute(1,2,0).cpu().numpy()
    flo_refr_mask = np.linalg.norm(flo_refr,2,-1)>0 # this also removes 0 flows
    flo_targ_mask = np.linalg.norm(flo_targ,2,-1)>0
    flo_refr_px = flo_refr * w_rszd / 2
    flo_targ_px = flo_targ * w_rszd / 2

    #fb check
    x0,y0  =np.meshgrid(range(w_rszd),range(h_rszd))
    hp0 = np.stack([x0,y0],-1) # screen coord

    flo_fb = warp_flow(hp0 + flo_targ_px, flo_refr_px) - hp0
    flo_fb = 2*flo_fb/w_rszd
    fberr_fw = np.linalg.norm(flo_fb, 2,-1)
    fberr_fw[~flo_refr_mask] = 0

    flo_bf = warp_flow(hp0 + flo_refr_px, flo_targ_px) - hp0
    flo_bf = 2*flo_bf/w_rszd
    fberr_bw = np.linalg.norm(flo_bf, 2,-1)
    fberr_bw[~flo_targ_mask] = 0

    if save_path is not None:
        # vis
        thrd_vis = 0.01
        img_refr = F.interpolate(img_refr, (h_rszd, w_rszd), mode='bilinear')[0]
        img_refr = img_refr.permute(1,2,0).cpu().numpy()[:,:,::-1]
        img_targ = F.interpolate(img_targ, (h_rszd, w_rszd), mode='bilinear')[0]
        img_targ = img_targ.permute(1,2,0).cpu().numpy()[:,:,::-1]
        flo_refr[:,:,0] = (flo_refr[:,:,0] + 2)/2
        flo_targ[:,:,0] = (flo_targ[:,:,0] - 2)/2
        flo_refr[fberr_fw>thrd_vis]=0.
        flo_targ[fberr_bw>thrd_vis]=0.
        flo_refr[~flo_refr_mask]=0.
        flo_targ[~flo_targ_mask]=0.
        img = np.concatenate([img_refr, img_targ], 1)
        flo = np.concatenate([flo_refr, flo_targ], 1)
        imgflo = cat_imgflo(img, flo)
        imgcnf = np.concatenate([fberr_fw, fberr_bw],1)
        imgcnf = np.clip(imgcnf, 0, dp_thrd)*(255/dp_thrd)
        imgcnf = np.repeat(imgcnf[...,None],3,-1)
        imgcnf = cv2.resize(imgcnf, imgflo.shape[::-1][1:])
        imgflo_cnf = np.concatenate([imgflo, imgcnf],0)
        cv2.imwrite(save_path, imgflo_cnf)
    return fberr_fw, fberr_bw


def mask_aug(rendered):
    lb = 0.1;    ub = 0.3
    _,h,w=rendered.shape
    if np.random.binomial(1,0.5):
        sx = int(np.random.uniform(lb*w,ub*w))
        sy = int(np.random.uniform(lb*h,ub*h))
        cx = int(np.random.uniform(sx,w-sx))
        cy = int(np.random.uniform(sy,h-sy))
        feat_mean = rendered.mean(-1).mean(-1)[:,None,None]
        rendered[:,cx-sx:cx+sx,cy-sy:cy+sy] = feat_mean
    return rendered

def process_so3_seq(rtk_seq, vis=False, smooth=True):
    """
    rtk_seq, bs, N, 13 including
    {scoresx1, rotationsx9, translationsx3}
    """
    from utils.io import draw_cams
    scores =rtk_seq[...,0]
    bs,N = scores.shape
    rmat =  rtk_seq[...,1:10]
    tmat = rtk_seq[:,0,10:13]
    rtk_raw = rtk_seq[:,0,13:29].reshape((-1,4,4))
   
    distribution = torch.Tensor(scores).softmax(1)
    entropy = (-distribution.log() * distribution).sum(1)

    if vis:
        # draw distribution
        obj_scale = 3
        cam_space = obj_scale * 0.2
        tmat_raw = np.tile(rtk_raw[:,None,:3,3], (1,N,1))
        scale_factor = obj_scale/tmat_raw[...,-1].mean()
        tmat_raw *= scale_factor
        tmat_raw = tmat_raw.reshape((bs,12,-1,3))
        tmat_raw[...,-1] += np.linspace(-cam_space, cam_space,12)[None,:,None]
        tmat_raw = tmat_raw.reshape((bs,-1,3))
        # bs, tiltxae
        all_rts = np.concatenate([rmat, tmat_raw],-1)
        all_rts = np.transpose(all_rts.reshape(bs,N,4,3), [0,1,3,2])
    
        for i in range(bs):
            top_idx = scores[i].argsort()[-30:]
            top_rt = all_rts[i][top_idx]
            top_score = scores[i][top_idx]
            top_score = (top_score - top_score.min())/(top_score.max()-top_score.min())
            mesh = draw_cams(top_rt, color_list = top_score)
            mesh.export('tmp/%d.obj'%(i))
   
    if smooth:
        # graph cut scores, bsxN
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_softmax
        graph = dcrf.DenseCRF2D(bs, 1, N)  # width, height, nlabels
        unary = unary_from_softmax(distribution.numpy().T.copy())
        graph.setUnaryEnergy(unary)
        grid = rmat[0].reshape((N,3,3))
        drot = np.matmul(grid[None], np.transpose(grid[:,None], (0,1,3,2)))
        drot = rot_angle(torch.Tensor(drot))
        compat = (-2*(drot).pow(2)).exp()*10
        compat = compat.numpy()
        graph.addPairwiseGaussian(sxy=10, compat=compat)

        Q = graph.inference(100)
        scores = np.asarray(Q).T

    # argmax
    idx_max = scores.argmax(-1)
    rmat = rmat[0][idx_max]

    rmat = rmat.reshape((-1,9))
    rts = np.concatenate([rmat, tmat],-1)
    rts = rts.reshape((bs,1,-1))

    # post-process se3
    root_rmat = rts[:,0,:9].reshape((-1,3,3))
    root_tmat = rts[:,0,9:12]
    
    rmat = rtk_raw[:,:3,:3]
    tmat = rtk_raw[:,:3,3]
    tmat = tmat + np.matmul(rmat, root_tmat[...,None])[...,0]
    rmat = np.matmul(rmat, root_rmat)
    rtk_raw[:,:3,:3] = rmat
    rtk_raw[:,:3,3] = tmat
   
    if vis:
        # draw again
        pdb.set_trace()
        rtk_vis = rtk_raw.copy()
        rtk_vis[:,:3,3] *= scale_factor
        mesh = draw_cams(rtk_vis)
        mesh.export('tmp/final.obj')
    return rtk_raw

def align_sim3(rootlist_a, rootlist_b, is_inlier=None, err_valid=None):
    """
    nx4x4 matrices
    is_inlier: n
    """
#    ta = np.matmul(-np.transpose(rootlist_a[:,:3,:3],[0,2,1]), 
#                                 rootlist_a[:,:3,3:4])
#    ta = ta[...,0].T
#    tb = np.matmul(-np.transpose(rootlist_b[:,:3,:3],[0,2,1]), 
#                                 rootlist_b[:,:3,3:4])
#    tb = tb[...,0].T
#    dso3,dtrn,dscale=umeyama_alignment(tb, ta,with_scale=False)
#    
#    dscale = np.linalg.norm(rootlist_a[0,:3,3],2,-1) /\
#             np.linalg.norm(rootlist_b[0,:3,3],2,-1)
#    rootlist_b[:,:3,:3] = np.matmul(rootlist_b[:,:3,:3], dso3.T[None])
#    rootlist_b[:,:3,3:4] = rootlist_b[:,:3,3:4] - \
#            np.matmul(rootlist_b[:,:3,:3], dtrn[None,:,None]) 

    dso3 = np.matmul(np.transpose(rootlist_b[:,:3,:3],(0,2,1)),
                        rootlist_a[:,:3,:3])
    dscale = np.linalg.norm(rootlist_a[:,:3,3],2,-1)/\
            np.linalg.norm(rootlist_b[:,:3,3],2,-1)

    # select inliers to fit 
    if is_inlier is not None:
        if is_inlier.sum() == 0:
            is_inlier[np.argmin(err_valid)] = True
        dso3 = dso3[is_inlier]
        dscale = dscale[is_inlier]

    dso3 = R.from_matrix(dso3).mean().as_matrix()
    rootlist_b[:,:3,:3] = np.matmul(rootlist_b[:,:3,:3], dso3[None])

    dscale = dscale.mean()
    rootlist_b[:,:3,3] = rootlist_b[:,:3,3] * dscale

    so3_err = np.matmul(rootlist_a[:,:3,:3], 
            np.transpose(rootlist_b[:,:3,:3],[0,2,1]))
    so3_err = rot_angle(torch.Tensor(so3_err))
    so3_err = so3_err / np.pi*180
    so3_err_max = so3_err.max()
    so3_err_mean = so3_err.mean()
    so3_err_med = np.median(so3_err)
    so3_err_std = np.asarray(so3_err.std())
    print(so3_err)
    print('max  so3 error (deg): %.1f'%(so3_err_max))
    print('med  so3 error (deg): %.1f'%(so3_err_med))
    print('mean so3 error (deg): %.1f'%(so3_err_mean))
    print('std  so3 error (deg): %.1f'%(so3_err_std))

    return rootlist_b

def align_sfm_sim3(aux_seq, datasets):
    from utils.io import draw_cams, load_root
    for dataset in datasets:
        seqname = dataset.imglist[0].split('/')[-2]

        # only process dataset with rtk_path input
        if dataset.has_prior_cam:
            root_dir = dataset.rtklist[0][:-9]
            root_sfm = load_root(root_dir, 0)[:-1] # excluding the last

            # split predicted root into multiple sequences
            seq_idx = [seqname == i.split('/')[-2] for i in aux_seq['impath']]
            root_pred = aux_seq['rtk'][seq_idx]
            is_inlier = aux_seq['is_valid'][seq_idx]
            err_valid = aux_seq['err_valid'][seq_idx]
            # only use certain ones to match
            #pdb.set_trace()
            #mesh = draw_cams(root_sfm, color='gray')
            #mesh.export('0.obj')
            
            # pre-align the center according to cat mask
            root_sfm = visual_hull_align(root_sfm, 
                    aux_seq['kaug'][seq_idx],
                    aux_seq['masks'][seq_idx])

            root_sfm = align_sim3(root_pred, root_sfm, 
                    is_inlier=is_inlier, err_valid=err_valid)
            # only modify rotation
            #root_pred[:,:3,:3] = root_sfm[:,:3,:3]
            root_pred = root_sfm
            
            aux_seq['rtk'][seq_idx] = root_pred
            aux_seq['is_valid'][seq_idx] = True
        else:
            print('not aligning %s, no rtk path in config file'%seqname)

def visual_hull_align(rtk, kaug, masks):
    """
    input: array
    output: array
    """
    rtk = torch.Tensor(rtk)
    kaug = torch.Tensor(kaug)
    masks = torch.Tensor(masks)
    num_view,h,w = masks.shape
    grid_size = 64
   
    if rtk.shape[0]!=num_view:
        print('rtk size mismtach: %d vs %d'%(rtk.shape[0], num_view))
        rtk = rtk[:num_view]
        
    rmat = rtk[:,:3,:3]
    tmat = rtk[:,:3,3:]

    Kmat = K2mat(rtk[:,3])
    Kaug = K2inv(kaug) # p = Kaug Kmat P
    kmat = mat2K(Kaug.matmul(Kmat))

    rmatc = rmat.permute((0,2,1))
    tmatc = -rmatc.matmul(tmat)

    bound = tmatc.norm(2,-1).mean()
    pts = np.linspace(-bound, bound, grid_size).astype(np.float32)
    query_yxz = np.stack(np.meshgrid(pts, pts, pts), -1)  # (y,x,z)
    query_yxz = torch.Tensor(query_yxz).view(-1, 3)
    query_xyz = torch.cat([query_yxz[:,1:2], query_yxz[:,0:1], query_yxz[:,2:3]],-1)

    score_xyz = []
    chunk = 1000
    for i in range(0,len(query_xyz),chunk):
        query_xyz_chunk = query_xyz[None, i:i+chunk].repeat(num_view, 1,1)
        query_xyz_chunk = obj_to_cam(query_xyz_chunk, rmat, tmat)
        query_xyz_chunk = pinhole_cam(query_xyz_chunk, kmat)

        query_xy = query_xyz_chunk[...,:2]
        query_xy[...,0] = query_xy[...,0]/w*2-1
        query_xy[...,1] = query_xy[...,1]/h*2-1

        # sum over time
        score = F.grid_sample(masks[:,None], query_xy[:,None])[:,0,0]
        score = score.sum(0)
        score_xyz.append(score)

    # align the center
    score_xyz = torch.cat(score_xyz)
    center = query_xyz[score_xyz>0.8*num_view]
    print('%d points used to align center'% (len(center)) )
    center = center.mean(0)
    tmatc = tmatc - center[None,:,None]
    tmat = np.matmul(-rmat, tmatc)
    rtk[:,:3,3:] = tmat

    return rtk

def ood_check_cse(dp_feats, dp_embed, dp_idx):
    """
    dp_feats: bs,16,h,w
    dp_idx:   bs, h,w
    dp_embed: N,16
    valid_list bs
    """
    bs,_,h,w = dp_feats.shape
    N,_ = dp_embed.shape
    device = dp_feats.device
    dp_idx = F.interpolate(dp_idx.float()[None], (h,w), mode='nearest').long()[0]
    
    ## dot product 
    #pdb.set_trace()
    #err_list = []
    #err_threshold = 0.05
    #for i in range(bs):
    #    err = 1- (dp_embed[dp_idx[i]]*dp_feats[i].permute(1,2,0)).sum(-1)
    #    err_list.append(err)

    # fb check
    err_list = []
    err_threshold = 12
    # TODO no fb check
    #err_threshold = 100
    for i in range(bs):
        # use chunk
        chunk = 5000
        max_idx = torch.zeros(N).to(device)
        for j in range(0,N,chunk):
            costmap = (dp_embed.view(N,16,1)[j:j+chunk]*\
                    dp_feats[i].view(1,16,h*w)).sum(-2)
            max_idx[j:j+chunk] = costmap.argmax(-1)  #  N
    
        rpj_idx = max_idx[dp_idx[i]]
        rpj_coord = torch.stack([rpj_idx % w, rpj_idx//w],-1)
        ref_coord = sample_xy(w, 1, 0, device, return_all=True)[1].view(h,w,2)
        err = (rpj_coord - ref_coord).norm(2,-1) 
        err_list.append(err)

    valid_list = []
    error_list = []
    for i in range(bs):
        err = err_list[i]
        mean_error = err[dp_idx[i]!=0].mean()
        is_valid = mean_error < err_threshold
        error_list.append( mean_error)
        valid_list.append( is_valid  )
        #cv2.imwrite('tmp/%05d.png'%i, (err/mean_error).cpu().numpy()*100)
        #print(i); print(mean_error)
    error_list = torch.stack(error_list,0)
    valid_list = torch.stack(valid_list,0)

    return valid_list, error_list

def bbox_dp2rnd(bbox, kaug):
    """
    bbox: bs, 4
    kaug: bs, 4
    cropab2: bs, 3,3, transformation from dp bbox to rendered bbox coords
    """
    cropa2im = torch.cat([(bbox[:,2:] - bbox[:,:2]) / 112., 
                           bbox[:,:2]],-1)
    cropa2im = K2mat(cropa2im)
    im2cropb = K2inv(kaug) 
    cropa2b = im2cropb.matmul(cropa2im)
    return cropa2b
            



def resample_dp(dp_feats, dp_bbox, kaug, target_size, rt_grid=False):
    """
    dp_feats: bs, 16, h,w
    dp_bbox:  bs, 4
    kaug:     bs, 4
    """
    # if dp_bbox are all zeros, just do the resizing
    if dp_bbox.abs().sum()==0:
        dp_feats_rsmp = F.interpolate(dp_feats, (target_size, target_size),
                                                            mode='bilinear')
        xygrid=None
    else:
        dp_size = dp_feats.shape[-1]
        device = dp_feats.device

        dp2rnd = bbox_dp2rnd(dp_bbox, kaug)
        rnd2dp = Kmatinv(dp2rnd)
        xygrid = sample_xy(target_size, 1, 0, device, return_all=True)[1] 
        xygrid = xygrid.matmul(rnd2dp[:,:2,:2]) + rnd2dp[:,None,:2,2]
        xygrid_norm = xygrid / dp_size * 2 - 1 
        if rt_grid:
            dp_feats_rsmp = None
        else:
            dp_feats_rsmp = F.grid_sample(dp_feats, xygrid_norm.view(-1,target_size,target_size,2))
    return dp_feats_rsmp, xygrid


def vrender_flo(weights_coarse, xyz_coarse_target, xys, img_size):
    """
    weights_coarse:     ..., ndepth
    xyz_coarse_target:  ..., ndepth, 3
    flo_coarse:         ..., 2
    flo_valid:          ..., 1
    """
    # render flow 
    weights_coarse = weights_coarse.clone()
    xyz_coarse_target = xyz_coarse_target.clone()

    # bs, nsamp, -1, x
    weights_shape = weights_coarse.shape
    xyz_coarse_target = xyz_coarse_target.view(weights_shape+(3,))
    xy_coarse_target = xyz_coarse_target[...,:2]

    # deal with negative z
    invalid_ind = torch.logical_or(xyz_coarse_target[...,-1]<1e-5,
                           xy_coarse_target.norm(2,-1).abs()>2*img_size)
    weights_coarse[invalid_ind] = 0.
    xy_coarse_target[invalid_ind] = 0.

    # renormalize
    weights_coarse = weights_coarse/(1e-9+weights_coarse.sum(-1)[...,None])

    # candidate motion vector
    xys_unsq = xys.view(weights_shape[:-1]+(1,2))
    flo_coarse = xy_coarse_target - xys_unsq
    flo_coarse =  weights_coarse[...,None] * flo_coarse
    flo_coarse = flo_coarse.sum(-2)

    ## candidate target point
    #xys_unsq = xys.view(weights_shape[:-1]+(2,))
    #xy_coarse_target = weights_coarse[...,None] * xy_coarse_target
    #xy_coarse_target = xy_coarse_target.sum(-2)
    #flo_coarse = xy_coarse_target - xys_unsq

    flo_coarse = flo_coarse/img_size * 2
    flo_valid = ((1-invalid_ind.float()).sum(-1)>0).float()[...,None]
    return flo_coarse, flo_valid

def diff_flo(pts_target, xys, img_size):
    """
    pts_target:         ..., 1, 2
    xys:                ..., 2
    flo_coarse:         ..., 2
    flo_valid:          ..., 1
    """

    # candidate motion vector
    pts_target = pts_target.view(xys.shape)
    flo_coarse = pts_target - xys
    flo_coarse = flo_coarse/img_size * 2
    return flo_coarse

def fid_reindex(fid, num_vids, vid_offset):
    """
    re-index absolute frameid {0,....N} to subsets of video id and relative frameid
    fid: N absolution id
    vid: N video id
    tid: N relative id
    """
    tid = torch.zeros_like(fid).float()
    vid = torch.zeros_like(fid)
    max_ts = (vid_offset[1:] - vid_offset[:-1]).max()
    for i in range(num_vids):
        assign = torch.logical_and(fid>=vid_offset[i],
                                    fid<vid_offset[i+1])
        vid[assign] = i
        tid[assign] = fid[assign].float() - vid_offset[i]
        doffset = vid_offset[i+1] - vid_offset[i]
        tid[assign] = (tid[assign] - doffset/2)/max_ts*2
        #tid[assign] = 2*(tid[assign] / doffset)-1
        #tid[assign] = (tid[assign] - doffset/2)/1000.
    return vid, tid

def create_base_se3(bs, device):
    """
    create a base se3 based on near-far plane
    """
    rt = torch.zeros(bs,3,4).to(device)
    rt[:,:3,:3] = torch.eye(3)[None].repeat(bs,1,1).to(device)
    rt[:,:2,3] = 0.
    rt[:,2,3] = 0.3
    return rt

def refine_rt(rt_raw, root_rts):
    """
    input:  rt_raw representing the initial root poses (after scaling)
    input:  root_rts representing delta se3
    output: current estimate of rtks for all frames
    """
    rt_raw = rt_raw.clone()
    root_rmat = root_rts[:,0,:9].view(-1,3,3)
    root_tmat = root_rts[:,0,9:12]

    rmat = rt_raw[:,:3,:3].clone()
    tmat = rt_raw[:,:3,3].clone()
    tmat = tmat + rmat.matmul(root_tmat[...,None])[...,0]
    rmat = rmat.matmul(root_rmat)
    rt_raw[:,:3,:3] = rmat
    rt_raw[:,:3,3] = tmat
    return rt_raw

def se3_vec2mat(vec):
    """
    torch/numpy function
    vec: ...,7, quaternion real last
    or vec: ...,6, axis angle
    mat: ...,4,4
    """
    shape = vec.shape[:-1]
    if torch.is_tensor(vec):
        mat = torch.zeros(shape+(4,4)).to(vec.device)
        if vec.shape[-1] == 6:
            rmat = transforms.axis_angle_to_matrix(vec[...,3:6])
        else:
            vec = vec[...,[0,1,2,6,3,4,5]] # xyzw => wxyz
            rmat = transforms.quaternion_to_matrix(vec[...,3:7]) 
        tmat = vec[...,:3]
    else:
        mat = np.zeros(shape+(4,4))
        vec = vec.reshape((-1,vec.shape[-1]))
        if vec.shape[-1]==6:
            rmat = R.from_axis_angle(vec[...,3:6]).as_matrix() # xyzw
        else:
            rmat = R.from_quat(vec[...,3:7]).as_matrix() # xyzw
        tmat = np.asarray(vec[...,:3])
        rmat = rmat.reshape(shape+(3,3))
        tmat = tmat.reshape(shape+(3,))
    mat[...,:3,:3] = rmat
    mat[...,:3,3] = tmat
    mat[...,3,3] = 1
    return mat

def se3_mat2rt(mat):
    """
    numpy function
    mat: ...,4,4
    rmat: ...,3,3
    tmat: ...,3
    """
    rmat = mat[...,:3,:3]
    tmat = mat[...,:3,3]
    return rmat, tmat

def se3_mat2vec(mat, outdim=7):
    """
    mat: ...,4,4
    vec: ...,7
    """
    shape = mat.shape[:-2]
    assert( torch.is_tensor(mat) )
    tmat = mat[...,:3,3]
    quat = transforms.matrix_to_quaternion(mat[...,:3,:3]) 
    if outdim==7:
        rot = quat[...,[1,2,3,0]] # xyzw <= wxyz
    elif outdim==6:
        rot = transforms.quaternion_to_axis_angle(quat)
    else: print('error'); exit()
    vec = torch.cat([tmat, rot], -1)
    return vec

def fit_plane_contact(can_to_env, kps, high_pt=None, meshbg=None):
    """
    numpy inputs
    high_pt: high point used to resolve normal direction
    """
    import pyransac3d as pyrsc
    # fit plane with foot
    kps_prev = can_to_env @ kps
    kps_prev = kps_prev[:,:3]
    kps_prev = kps_prev.transpose([0,2,1]).reshape(-1,3)
    if high_pt is not None: 
        high_pt = (can_to_env @ high_pt)[...,0]

    plane1 = pyrsc.Plane()
    if meshbg is not None:
        # reduce the mesh to be part around the foot
        reduce_mesh_around_kp(meshbg, kps_prev, high_pt)
   
        ## smooth mesh
        #import pymeshlab
        #ms = pymeshlab.MeshSet()
        #ms.add_mesh(pymeshlab.Mesh(meshbg.vertices, meshbg.faces), "name")
        #p = pymeshlab.Percentage(10) # use larger values to get smoother mesh
        #mesh_isotropic = ms.meshing_isotropic_explicit_remeshing(targetlen=p)
        #meshbg = trimesh.Trimesh( ms.current_mesh().vertex_matrix(),
        #                        ms.current_mesh().face_matrix())
        num_verts = len(meshbg.vertices)
        kp_dis = np.linalg.norm((meshbg.vertices[None] - kps_prev[:,None]),2,-1)
        idx=np.unique(np.argsort(kp_dis, 1)[:,:num_verts//20].reshape(-1))
        ground_pts = meshbg.vertices[idx]

        #meshbg.export('tmp/0.obj')
        #trimesh.Trimesh(ground_pts).export('tmp/1.obj')
        #trimesh.Trimesh(kps_prev).export('tmp/2.obj')
        #pdb.set_trace()

        #npt = kps.shape[0]
        #_,dis,triangle_id = trimesh.proximity.closest_point(meshbg, kps_prev)
        #best_inliers = np.argsort(dis)[:10]
        ##best_inliers = np.argsort(dis)[:npt//2]
        #triangle_id = triangle_id[best_inliers]
        ## find normal
        #plane_n = meshbg.face_normals[triangle_id].mean(0)
        #plane_n = plane_n / np.linalg.norm(plane_n,2,0)
        #center = kps_prev[best_inliers].mean(0)
        #plane_d = -(center * plane_n).sum()
        #best_eq = np.concatenate([plane_n, plane_d[None]])
    
        best_eq, best_inliers = plane1.fit(ground_pts, 0.001)
    else:
        ground_pts = kps_prev
        best_eq, best_inliers = plane1.fit(ground_pts, 0.01)
    best_eq = np.asarray(best_eq)

    if high_pt is None:
        if best_eq[1]<0: best_eq = -1*best_eq
    else:
        if (high_pt * best_eq[None]).sum()>0: # should be on the neg side
            best_eq = -1*best_eq
    plane_n = np.asarray(best_eq[:3])
    center = ground_pts[best_inliers].mean(0)
    dist = (center * plane_n).sum() + best_eq[3]
    plane_o = center - plane_n * dist
    plane = np.concatenate([plane_o, plane_n])
    T = trimesh.geometry.plane_transform(origin=plane[:3], normal=plane[3:6]) # align with 00-1
    xy2xz = np.eye(4)
    xy2xz[:3,:3] = cv2.Rodrigues(np.asarray([-np.pi/2,0,0]))[0]
    xy2xz[:3,:3] = cv2.Rodrigues(np.asarray([0,-np.pi/2,0]))[0]@xy2xz[:3,:3]
    bg2world = xy2xz@T # coplanar with xy->xz plane
    return bg2world, xy2xz
        
def eval_fgbg_scale(bg2fg_scale, cams, bgcams, kps, verts):
    nframe, _, nkp = kps.shape
    chamLoss = chamfer3D.dist_chamfer_3D.chamfer_3DDist()

    scale = (-bg2fg_scale).exp() # fg2bg
    view_kps = cams @ kps # view space -1,4,k
    view_kps = scale * view_kps
    view_kps[:,-1] = 1.
    kps_prev = bgcams.inverse() @ view_kps
    kps_prev = kps_prev[:,:3]
    kps_prev = kps_prev.permute(0,2,1).reshape(-1,3)
    
    raw_cd,raw_cd_back,_,_ = chamLoss(kps_prev[None],verts[None])  # this returns distance squared
    #TODO this rule failed when half points penetrates the wall
    #raw_cd = raw_cd.view(-1)
    #loss = raw_cd.topk(nframe*nkp*2//4, largest=False)[0].mean() # make sure top 50% kps are on ground
    #TODO top half point at each time?
    assert(nkp>1)
    raw_cd = raw_cd.reshape(nframe, nkp)
    loss = raw_cd.topk(nkp//2,1,largest=False)[0].mean() # make sure top 50% kps are on ground
    return loss, kps_prev
    

def reduce_mesh_around_kp(mesh, kps_out, high_pt):
    """
    process the mesh a bit, because the mesh was too big, only use the in contact part
    """
    n_fr = high_pt.shape[0]
    verts_to_kp = np.linalg.norm((mesh.vertices[None] - kps_out[:,None]),2,-1)
    verts_dis_min = verts_to_kp.min(0)
    kps_reshaped = kps_out.reshape((n_fr, -1, 3))
    kp_dis_max = np.linalg.norm(kps_reshaped[:,None] - kps_reshaped[:,:,None],2,-1).max() # within a frame
    #kp_dis_max = np.linalg.norm(kps_out[None] - kps_out[:,None],2,-1).max()
    verts_mask = verts_dis_min < kp_dis_max*2
    #TODO remove verts with a different direction than kp-high_pt
    low_pt = kps_out.reshape((high_pt.shape[0],-1,3)).mean(1)
    up_vec = (high_pt[:,:3] - low_pt).mean(0)
    up_vec /= np.linalg.norm(up_vec)
    dir_mask = (mesh.vertex_normals * up_vec[None]).sum(-1)>0.5 # 0: 90 deg, 0.5:60 deg
    verts_mask = np.logical_and(verts_mask,  dir_mask)
    
    faces_mask = verts_mask[mesh.faces.reshape((-1))].reshape((-1,3)).sum(1)==3
    mesh.update_faces(faces_mask)
    mesh.remove_unreferenced_vertices()

def optimize_scale(bgcams_np, cams_np, kps_np, mesh, 
                init_scale=0, debug=False, high_pt=None, use_mesh_plane=True):
    mesh = mesh.copy()
    # ransac scheme?
    bg2fg_scale = nn.Parameter(torch.Tensor([init_scale]).cuda())
    optimizer = torch.optim.AdamW([{'params': [bg2fg_scale]}], lr=1e-2)

    bgcams = torch.Tensor(bgcams_np).cuda()
    cams = torch.Tensor(cams_np).cuda()
    kps = torch.Tensor(kps_np).cuda()
    verts = torch.Tensor(mesh.vertices).cuda()

    # normalize 
    verts_scale = 1./verts.std() # this makes sure object starts in front of the bg
    verts *= verts_scale
    bgcams[:,:3,3] *= verts_scale
    cams[:,:3,3] *= verts_scale
    kps[:,:3] *= verts_scale # -1,4,K

    # do a discrete search first
    loss_queried = []
    val_input = []
    kps_out = []
    for it,val in enumerate(np.linspace(-3,3,200)): # 5: 148x, 3: 20x
        bg2fg_scale.data[:] = val
        loss,kps_out_sub = eval_fgbg_scale(bg2fg_scale, cams, bgcams, kps, verts)
        if it%10==0:
            trimesh.Trimesh(kps_out_sub.detach().cpu()).export('tmp/d-%04d.obj'%it)
        val_input.append(val)
        loss_queried.append(loss)
        kps_out.append(kps_out_sub)
    val_idx = torch.stack(loss_queried,0).argmin()
    bg2fg_scale.data[:] = val_input[val_idx]
    kps_out = kps_out[val_idx]
    
    # reduce after first init
    scale = bg2fg_scale.exp().detach().cpu().numpy() # fg2bg
    bgcams_np_tmp = bgcams_np.numpy().copy()
    bgcams_np_tmp[:,:3,3] *= scale
    bg_high_pt = np.linalg.inv(bgcams_np_tmp) @ cams_np.numpy() @ high_pt
    reduce_mesh_around_kp(mesh, kps_out.detach().cpu().numpy()/verts_scale.cpu().numpy(), bg_high_pt[...,0])
    verts = torch.Tensor(mesh.vertices).cuda() * verts_scale


    if debug: trimesh.Trimesh(verts.cpu(), mesh.faces).export('tmp/bg.obj')
    for i in range(2000): 
        loss,kps_prev = eval_fgbg_scale(bg2fg_scale, cams, bgcams, kps, verts)
        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()
        if debug:
            if i==0 or (i+1)%100==0:
                trimesh.Trimesh(kps_prev.detach().cpu()).export('tmp/%04d.obj'%i)
                print(loss)
    scale = (-bg2fg_scale).exp() # fg2bg
    scale = 1./scale
    scale = scale.detach().cpu().numpy()
    
    # fit plane
    if torch.is_tensor(bgcams_np): bgcams_np = bgcams_np.cpu().numpy()
    if torch.is_tensor(cams_np): cams_np = cams_np.cpu().numpy()
    if torch.is_tensor(kps_np): kps_np = kps_np.cpu().numpy()
    bgcams_np[:,:3,3] *= scale
    can_to_env = np.linalg.inv(bgcams_np) @ cams_np
    if use_mesh_plane:
        meshbg_scaled = mesh.copy()
        meshbg_scaled.vertices = meshbg_scaled.vertices * scale
        bg2world,xy2xz = fit_plane_contact(can_to_env, kps_np, high_pt=high_pt,
                                            meshbg=meshbg_scaled)
    else:
        # use foot plane for quadruqed since 4 => plane more robust
        bg2world,xy2xz = fit_plane_contact(can_to_env, kps_np, high_pt=high_pt)

    kps = bg2world @ can_to_env @ kps_np
    #bg2world[1,3] -= kps[:,1,:].max()
    #bg2world[1,3] -= 0.01
    if debug:
        from utils.io import vis_kps
        kps = bg2world @ can_to_env @ kps_np
        vis_kps(kps, 'tmp/kps.obj')
    return scale, bgcams_np, bg2world, xy2xz

def extract_mesh_simp(model,chunk,grid_size,
                  threshold = -0.002,
                  embedid=None,
                  vidid=None,
                  beta=None,
                  is_eval=False):
    opts = model.opts
    device=model.near_far.device
    bound = model.latest_vars['obj_bound']

    # sample grid
    ptx = np.linspace(-bound[0], bound[0], grid_size).astype(np.float32)
    pty = np.linspace(-bound[1], bound[1], grid_size).astype(np.float32)
    ptz = np.linspace(-bound[2], bound[2], grid_size).astype(np.float32)
    query_yxz = np.stack(np.meshgrid(pty, ptx, ptz), -1)  # (y,x,z)
    query_yxz = torch.Tensor(query_yxz).to(device).view(-1, 3)
    query_xyz = torch.cat([query_yxz[:,1:2], query_yxz[:,0:1], query_yxz[:,2:3]],-1)

    bs_pts = query_xyz.shape[0]
    out_chunks = []
    if beta is not None:
        beta_geo = beta[None].repeat(chunk,1)
    else:
        beta_geo = beta
    for i in range(0, bs_pts, chunk):
        query_xyz_chunk = query_xyz[i:i+chunk]

        xyz_embedded = model.embedding_xyz(query_xyz_chunk) # (N, embed_xyz_channels)
        if vidid is not None:
            # expand video id
            vidid_chunk = vidid * torch.ones(xyz_embedded.shape[:-1]).to(device)
            vidid_chunk = vidid_chunk.long()
        else: vidid_chunk = None

        sdf = model.nerf_models['coarse'](xyz_embedded, vidid=vidid_chunk, beta=beta_geo)
        out_chunks += [sdf]

    vol_o = torch.cat(out_chunks, 0)
    vol_o = vol_o.view(grid_size, grid_size, grid_size)

    print('fraction occupied:', (vol_o > threshold).float().mean())
    vertices, triangles = mcubes.marching_cubes(vol_o.cpu().numpy(), threshold)
    vertices = (vertices - grid_size/2)/grid_size*2*bound[None, :]

    mesh = trimesh.Trimesh(vertices, triangles)

    # mesh post-processing 
    if len(mesh.vertices)>0:
        if opts.use_cc:
            # keep the largest mesh
            mesh = [i for i in mesh.split(only_watertight=False)]
            mesh = sorted(mesh, key=lambda x:x.vertices.shape[0])
            mesh = mesh[-1]

        if is_eval:
            print('simplifying the mesh')
            import subprocess
            if vidid is None: 
                suffix='999'
            else:
                suffix='%02d'%(vidid)
    
            # fix non watertight meshes
            import pymeshfix
            meshfix = pymeshfix.MeshFix(mesh.vertices, mesh.faces)
            meshfix.repair()
            mesh = trimesh.Trimesh(meshfix.v, meshfix.f)

            mesh.export('tmp/input-%s.obj'%suffix)
            try:
                print(subprocess.check_output(['./Manifold/build/simplify', '-i', 'tmp/input-%s.obj'%suffix, '-o', 'tmp/simple-%s.obj'%suffix, '-m', '-f', '20000']))
                mesh = trimesh.load('tmp/simple-%s.obj'%suffix)
            except:
                print('simplification failed')
                # smooth the mesh (slow)
                import pymeshlab
                ms = pymeshlab.MeshSet()
                ms.add_mesh(pymeshlab.Mesh(mesh.vertices, mesh.faces), "name")
                p = pymeshlab.Percentage(1) # use larger values to get smoother mesh
                mesh_isotropic = ms.meshing_isotropic_explicit_remeshing(targetlen=p)
                mesh = trimesh.Trimesh( ms.current_mesh().vertex_matrix(),
                                        ms.current_mesh().face_matrix())

        # assign color based on canonical location
        vis = mesh.vertices
        try:
            model.module.vis_min = vis.min(0)[None]
            model.module.vis_len = vis.max(0)[None] - vis.min(0)[None]
        except: # test time
            model.vis_min = vis.min(0)[None]
            model.vis_len = vis.max(0)[None] - vis.min(0)[None]
        vis = vis - model.vis_min
        vis = vis / model.vis_len
        if not opts.ce_color:
            if beta is not None:
                beta_col = beta[None].repeat(vis.shape[0],1)
            else:
                beta_col = beta
            vis = get_vertex_colors(model, mesh, frame_idx=-1, vidid=vidid, beta=beta_col)
        mesh.visual.vertex_colors[:,:3] = vis*255
    return mesh

def extract_mesh(model,chunk,grid_size,
                  #threshold = -0.005,
                  threshold = -0.002,
                  #threshold = 0.,
                  embedid=None,
                  vidid=None,
                  mesh_dict_in=None,
                  is_eval=False):
    opts = model.opts
    device=model.near_far.device
    mesh_dict = {}
    #if model.near_far is not None: 
    if vidid is not None and 'obj_bounds' in model.latest_vars.keys():
        bound = model.latest_vars['obj_bounds'][int(vidid)]
    else:
        bound = model.latest_vars['obj_bound']
    #else: bound=1.5*np.asarray([1,1,1])

    if mesh_dict_in is None:
        ptx = np.linspace(-bound[0], bound[0], grid_size).astype(np.float32)
        pty = np.linspace(-bound[1], bound[1], grid_size).astype(np.float32)
        ptz = np.linspace(-bound[2], bound[2], grid_size).astype(np.float32)
        query_yxz = np.stack(np.meshgrid(pty, ptx, ptz), -1)  # (y,x,z)
        #pts = np.linspace(-bound, bound, grid_size).astype(np.float32)
        #query_yxz = np.stack(np.meshgrid(pts, pts, pts), -1)  # (y,x,z)
        query_yxz = torch.Tensor(query_yxz).to(device).view(-1, 3)
        query_xyz = torch.cat([query_yxz[:,1:2], query_yxz[:,0:1], query_yxz[:,2:3]],-1)
        query_dir = torch.zeros_like(query_xyz)

        bs_pts = query_xyz.shape[0]
        out_chunks = []
        for i in range(0, bs_pts, chunk):
            query_xyz_chunk = query_xyz[i:i+chunk]
            query_dir_chunk = query_dir[i:i+chunk]

            # backward warping 
            if embedid is not None and not opts.queryfw:
                query_xyz_chunk, mesh_dict = warp_bw(opts, model, mesh_dict, 
                                               query_xyz_chunk, embedid)
            #if opts.symm_shape: 
            #    #TODO set to x-symmetric
            #    query_xyz_chunk[...,0] = query_xyz_chunk[...,0].abs()
            xyz_embedded = model.embedding_xyz(query_xyz_chunk) # (N, embed_xyz_channels)
            if vidid is not None:
                # expand video id
                vidid_chunk = vidid * torch.ones(xyz_embedded.shape[:-1]).to(device)
                vidid_chunk = vidid_chunk.long()
            else: vidid_chunk = None

            sdf = model.nerf_models['coarse'](xyz_embedded, vidid=vidid_chunk)
            #if 'bones' in model.nerf_models.keys():
            #    bones_rst, _ = zero_to_rest_bone(model, model.bones)
            #    #sdf = -extract_bone_sdf(bones_rst, model.skin_aux, query_xyz_chunk) 
            #    sdf = sdf - extract_bone_sdf(bones_rst, model.skin_aux, query_xyz_chunk) # out: negative
            out_chunks += [sdf]

        vol_o = torch.cat(out_chunks, 0)
        vol_o = vol_o.view(grid_size, grid_size, grid_size)
        #vol_o = F.softplus(vol_o)

        if not opts.full_mesh:
            #TODO set density of non-observable points to small value
            if model.latest_vars['idk'].sum()>0:
                vis_chunks = []
                for i in range(0, bs_pts, chunk):
                    query_xyz_chunk = query_xyz[i:i+chunk]
                    if opts.nerf_vis:
                        #if opts.symm_shape:
                        #    query_xyz_chunk[...,0] = query_xyz_chunk[...,0].abs()
                        # this leave no room for halucination and is not what we want
                        xyz_embedded = model.embedding_xyz(query_xyz_chunk) # (N, embed_xyz_channels)
                        if vidid is not None and model.nerf_vis.__class__.__name__=='NeRF_old':
                            # expand video id
                            vidid_chunk = vidid * torch.ones(xyz_embedded.shape[:-1]).to(device)
                            vidid_chunk = vidid_chunk.long()
                            vis_chunk_nerf = model.nerf_vis(xyz_embedded, vidid=vidid_chunk)
                        else: 
                            vis_chunk_nerf = model.nerf_vis(xyz_embedded)
                        vis_chunk = vis_chunk_nerf[...,0].sigmoid()
                    else:
                        #TODO deprecated!
                        vis_chunk = compute_point_visibility(query_xyz_chunk.cpu(),
                                         model.latest_vars, device)[None]
                    vis_chunks += [vis_chunk]
                vol_visi = torch.cat(vis_chunks, 0)
                vol_visi = vol_visi.view(grid_size, grid_size, grid_size)
                vol_o[vol_visi<0.2] = -1

        ## save color of sampled points 
        #cmap = cm.get_cmap('cool')
        ##pts_col = cmap(vol_visi.float().view(-1).cpu())
        #pts_col = cmap(vol_o.sigmoid().view(-1).cpu())
        #mesh = trimesh.Trimesh(query_xyz.view(-1,3).cpu(), vertex_colors=pts_col)
        #mesh.export('0.obj')
        #pdb.set_trace()

        print('fraction occupied:', (vol_o > threshold).float().mean())
        vertices, triangles = mcubes.marching_cubes(vol_o.cpu().numpy(), threshold)
        vertices = (vertices - grid_size/2)/grid_size*2*bound[None, :]

        mesh = trimesh.Trimesh(vertices, triangles)

        # mesh post-processing 
        if len(mesh.vertices)>0:
            if opts.use_cc:
                # keep the largest mesh
                mesh = [i for i in mesh.split(only_watertight=False)]
                mesh = sorted(mesh, key=lambda x:x.vertices.shape[0])
                mesh = mesh[-1]

            if is_eval:
                print('simplifying the mesh')
                import subprocess
                #import optimesh
                #mesh.vertices, mesh.faces = optimesh.optimize_points_cells(
                #mesh.vertices, mesh.faces, "CVT (block-diagonal)", 1.0e-5, 100)
                if vidid is None: 
                    suffix='999'
                else:
                    suffix='%02d'%(vidid)

                # fix non watertight meshes
                import pymeshfix
                meshfix = pymeshfix.MeshFix(mesh.vertices, mesh.faces)
                meshfix.repair()
                mesh = trimesh.Trimesh(meshfix.v, meshfix.f)

                mesh.export('tmp/input-%s.obj'%suffix)
                try:
                    print(subprocess.check_output(['./Manifold/build/simplify', '-i', 'tmp/input-%s.obj'%suffix, '-o', 'tmp/simple-%s.obj'%suffix, '-m', '-f', '20000']))
                    mesh = trimesh.load('tmp/simple-%s.obj'%suffix)
                except:
                    print('simplification failed')
                    # smooth the mesh (slow)
                    import pymeshlab
                    ms = pymeshlab.MeshSet()
                    ms.add_mesh(pymeshlab.Mesh(mesh.vertices, mesh.faces), "name")
                    p = pymeshlab.Percentage(1) # use larger values to get smoother mesh
                    mesh_isotropic = ms.meshing_isotropic_explicit_remeshing(targetlen=p)
                    mesh = trimesh.Trimesh( ms.current_mesh().vertex_matrix(),
                                            ms.current_mesh().face_matrix())
                    #print(subprocess.check_output(['./Manifold/build/manifold', 'tmp/input.obj', 'tmp/output.obj', '100000']))
                    #print(subprocess.check_output(['./Manifold/build/simplify', '-i', 'tmp/output.obj', '-o', 'tmp/simple.obj', '-m', '-f', '50000']))
                    #mesh = trimesh.load('tmp/simple.obj')

            # assign color based on canonical location
            vis = mesh.vertices
            try:
                model.module.vis_min = vis.min(0)[None]
                model.module.vis_len = vis.max(0)[None] - vis.min(0)[None]
            except: # test time
                model.vis_min = vis.min(0)[None]
                model.vis_len = vis.max(0)[None] - vis.min(0)[None]
            vis = vis - model.vis_min
            vis = vis / model.vis_len
            if not opts.ce_color:
                vis = get_vertex_colors(model, mesh, frame_idx=-1, vidid=vidid)
            mesh.visual.vertex_colors[:,:3] = vis*255
        
        # account for bg scaling
        if hasattr(model, "bg2fg_scale"):
            mesh.vertices *= model.bg2fg_scale[int(vidid)].cpu().numpy()
            mesh = mesh.copy()

    # forward warping
    if embedid is not None and opts.queryfw:
        mesh = mesh_dict_in['mesh'].copy()
        vertices = mesh.vertices
        vertices, mesh_dict = warp_fw(opts, model, mesh_dict, 
                                       vertices, embedid)
        mesh.vertices = vertices
           
    mesh_dict['mesh'] = mesh
    return mesh_dict

def get_vertex_colors(model, mesh, frame_idx=0, view_dir=None, vidid=None, beta=None):
    device = model.near_far.device
    # assign color to mesh verts according to current frame
    xyz_query = torch.cuda.FloatTensor(mesh.vertices, device=device)
    xyz_embedded = model.embedding_xyz(xyz_query) # (N, embed_xyz_channels)

    if model.nerf_models['coarse'].use_dir:
        # view dir
        if view_dir is None:
            # use view direction of (0,0,-1)
            dir_query = torch.zeros_like(xyz_query) 
            dir_query[:,2] = -1
        else:
            dir_query = F.normalize(view_dir, 2,-1)
        dir_embedded = model.embedding_dir(dir_query) # (N, embed_xyz_channels)
        xyz_embedded = torch.cat([xyz_embedded, dir_embedded],-1)

    if hasattr(model, 'env_code'):
        # env code
        if frame_idx>-1:
            # use env code of the first frame
            env_code = model.env_code(torch.Tensor([frame_idx]).long().to(device))
        else:
            # use average env code
            vid_offset = model.data_offset
            if vidid is None:
                all_ids = range(vid_offset[-1]-1)
            else:
                all_ids = range(vid_offset[int(vidid)], vid_offset[int(vidid)+1])
            all_ids = torch.Tensor(all_ids).long().to(device)
            #env_code = model.env_code(all_ids).mean(0)[None]
            env_code = model.env_code(all_ids)[:1]
        env_code = env_code.expand(xyz_query.shape[0],-1)
        xyz_embedded = torch.cat([xyz_embedded, env_code],-1)

    if vidid is not None:
        # expand video id
        vidid = vidid * torch.ones(xyz_embedded.shape[:-1]).to(device)
        vidid = vidid.long()
    vis = model.nerf_models['coarse'](xyz_embedded, vidid=vidid, beta=beta)[:,:3].cpu().numpy()
    vis = np.clip(vis, 0, 1)
    return vis

def transform_bg_to_cam(cam, bgcam, mesh, mesh_bg):
    # invert fg motion
    Rfg = torch.Tensor(cam[:3,:3]   )
    Tfg = torch.Tensor(cam[:3 ,3]   )
    Rfgi=Rfg.T
    Tfgi=-Rfg.T.matmul(Tfg[:,None])[:,0]

    # compose
    Rbg = torch.Tensor(bgcam[:3,:3]   )
    Tbg = torch.Tensor(bgcam[:3 ,3]   )
    Rmat = torch.Tensor(Rfgi.matmul(Rbg))
    Tmat = torch.Tensor(Rfgi.matmul(Tbg[:,None])[:,0] + Tfgi)
    
    # bg points
    mesh_bg_cam = mesh_bg.copy()
    verts = torch.Tensor(mesh_bg_cam.vertices)
    
    mesh_bg_cam.vertices = obj_to_cam(verts, Rmat, Tmat).numpy()
    mesh = trimesh.util.concatenate([mesh, mesh_bg_cam])
    return mesh
