# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import app
from absl import flags
from collections import defaultdict
import os
import os.path as osp
import pickle
import sys
sys.path.insert(0, 'third_party')
import cv2, numpy as np, time, torch, torchvision
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import trimesh, pytorch3d, pytorch3d.loss, pdb
from pytorch3d import transforms
import configparser

from nnutils.nerf import Embedding, NeRF, RTHead, RTHead_old, RTExplicit, Encoder,\
                    MultiMLP, BaseMLP, ConvProj, NeRFFeat, NeRFTransient, \
                    grab_xyz_weights, FrameCode, RTExpMLP, SkelHead, BANMoCNN, NeRFBG
from nnutils.robot import URDFRobot
from nnutils.geom_utils import K2mat, mat2K, Kmatinv, K2inv, raycast, sample_xy,\
                                chunk_rays, generate_bones,\
                                canonical2ndc, obj_to_cam, vec_to_sim3, \
                                near_far_to_bound, compute_flow_geodist, \
                                compute_flow_cse, fb_flow_check, pinhole_cam, \
                                render_color, mask_aug, bbox_dp2rnd, resample_dp, \
                                vrender_flo, get_near_far, array2tensor, rot_angle, \
                                rtk_invert, rtk_compose, bone_transform, zero_to_rest_bone,\
                                zero_to_rest_dpose, fid_reindex, create_base_se3, \
                                refine_rt, evaluate_mlp, center_mlp_skinning, gauss_skinning, \
                                reinit_bones, warp_fw, extract_mesh
from nnutils.urdf_utils import compute_bone_from_joint, visualize_joints
from nnutils.rendering import render_rays
from nnutils.loss_utils import rtk_loss, \
                            feat_match_loss, kp_reproj_loss, \
                            loss_filter, loss_filter_line, compute_xyz_wt_loss,\
                            compute_root_sm_2nd_loss, shape_init_loss, compute_cnn_loss
from utils.io import draw_pts

# distributed data parallel
flags.DEFINE_integer('local_rank', 0, 'for distributed training')
flags.DEFINE_integer('ngpu', 1, 'number of gpus to use')

# data io
flags.DEFINE_integer('accu_steps', 1, 'how many steps to do gradient accumulation')
flags.DEFINE_string('seqname', 'syn-spot-40', 'name of the sequence')
flags.DEFINE_string('logname', 'exp_name', 'Experiment Name')
flags.DEFINE_string('checkpoint_dir', 'logdir/', 'Root directory for output files')
flags.DEFINE_string('model_path', '', 'load model path')
flags.DEFINE_string('pose_cnn_path', '', 'path to pre-trained pose cnn')
flags.DEFINE_string('rtk_path', '', 'path to rtk files')
flags.DEFINE_bool('lineload',False,'whether to use pre-computed data per line')
flags.DEFINE_integer('n_data_workers', 1, 'Number of data loading workers')
flags.DEFINE_boolean('use_rtk_file', False, 'whether to use input rtk files')
flags.DEFINE_boolean('debug', False, 'deubg')

# for img2line
flags.DEFINE_boolean('save_to_img', False, 'save img to img npy instead of lines')
flags.DEFINE_string('save_prefix', '', 'save img npy with prefix$seqname')
flags.DEFINE_string('load_prefix', '', 'load img npy with prefix$seqname')

# model: shape, appearance, and feature
flags.DEFINE_bool('use_human', False, 'whether to use human cse model')
flags.DEFINE_bool('symm_shape', False, 'whether to set geometry to x-symmetry')
flags.DEFINE_float('symm_bgn', 0, 'prob of sampling symm points')
flags.DEFINE_float('symm_end', 0, 'prob of sampling symm points')
flags.DEFINE_bool('env_code', True, 'whether to use environment code for each video')
flags.DEFINE_bool('env_fourier', True, 'whether to use fourier basis for env')
flags.DEFINE_bool('use_unc',False, 'whether to use uncertainty sampling')
flags.DEFINE_bool('nerf_vis', True, 'use visibility volume')
flags.DEFINE_bool('anneal_freq', False, 'whether to use frequency annealing')
flags.DEFINE_integer('num_freq', 10, 'frequency for fourier features')
flags.DEFINE_integer('alpha', 10, 'maximum frequency for fourier features')
flags.DEFINE_bool('use_cc', True, 'whether to use connected component for mesh')
flags.DEFINE_bool('use_category',False, 'whether to use video shape code')
flags.DEFINE_float('catep_bgn', 0, 'prob of sampling random shape code')
flags.DEFINE_float('catep_end', 0, 'prob of sampling random shape code')
flags.DEFINE_string('bgmlp','', 'whether to train background mlp, {mlp, nerf}')
flags.DEFINE_string('bg_path','', 'path to pre-trained bg model')
flags.DEFINE_bool('copy_bgfl',False, 'whether to copy focal length from bg model')
flags.DEFINE_bool('ft_bgcam',False, 'whether to finetune bg cameras')
flags.DEFINE_bool('fit_scale',False, 'whether to fit the scale between fg/bg')
flags.DEFINE_bool('phys_opt',False, 'whether to optimize with physics loss')
flags.DEFINE_string('phys_vid', '', 'whether to optimize selected videos, e.g., 0,1,2')
flags.DEFINE_integer('phys_wdw_len', 8, 'window length')
flags.DEFINE_integer('cyc_len', 20, 'total cycle length')
flags.DEFINE_integer('phys_cyc_len', 10, 'phys cycle length')
flags.DEFINE_float('phys_wt',0.1, 'weight of physics loss')
flags.DEFINE_float('crop_factor',1.2, 'crop factor for image')
flags.DEFINE_bool('use_compose',False, 'whether to use background compositing')
flags.DEFINE_bool('dp_proj',False, 'whether to learn a projection of dp feats')
flags.DEFINE_string('pre_skel', '', 'whether to use predefined skeleton')

# model: motion
flags.DEFINE_bool('lbs', True, 'use lbs for backward warping 3d flow')
flags.DEFINE_bool('mlp_deform', False, 'use invertible 3d deformation over frame')
flags.DEFINE_integer('num_bones', 25, 'maximum number of bones')
flags.DEFINE_string('nerf_skin', 'mlp', 'which mlp skinning function')
flags.DEFINE_integer('t_embed_dim', 16, 'dimension of the pose code')
flags.DEFINE_bool('frame_code', True, 'whether to use frame code')

# model: cameras
flags.DEFINE_bool('use_cam', False, 'whether to use pre-defined camera pose')
flags.DEFINE_string('root_basis', 'expmlp', 'which root pose basis to use {mlp, cnn, exp}')
flags.DEFINE_bool('root_opt', True, 'whether to optimize root body poses')
flags.DEFINE_bool('ks_opt', True,   'whether to optimize camera intrinsics')
flags.DEFINE_float('rand_rep', 0, 'percentage of iters to repeat random noise')
flags.DEFINE_float('rand_std', 30, 'std of gaussian rotation noise in degree')
flags.DEFINE_string('depth_init', 'h', 'init depth based on bbox, use {h, hw}')

# optimization: hyperparams
flags.DEFINE_integer('num_epochs', 1000, 'Number of epochs to train')
flags.DEFINE_float('learning_rate', 5e-4, 'learning rate')
flags.DEFINE_integer('batch_size', 2, 'size of minibatches')
flags.DEFINE_integer('img_size', 256, 'image size for optimization')
flags.DEFINE_integer('nsample', 6, 'num of samples per image at optimization time')
flags.DEFINE_float('perturb',   1.0, 'factor to perturb depth sampling points')
flags.DEFINE_float('noise_std', 0., 'std dev of noise added to regularize sigma')
flags.DEFINE_float('nactive', 0.5, 'num of samples per image at optimization time')
flags.DEFINE_integer('ndepth', 128, 'num of depth samples per px at optimization time')
flags.DEFINE_float('clip_scale', 100, 'grad clip scale')
flags.DEFINE_float('warmup_steps', 0.4, 'steps used to increase sil loss')
flags.DEFINE_float('reinit_bone_steps', 0.667, 'steps to initialize bones')
flags.DEFINE_float('dskin_steps', 0.8, 'steps to add delta skinning weights')
flags.DEFINE_float('init_beta', 0.1, 'initial value for transparency beta')
flags.DEFINE_bool('reset_beta', False, 'reset volsdf beta to 0.1')
flags.DEFINE_float('fine_steps', 1.1, 'by default, not using fine samples')
flags.DEFINE_float('nf_reset', 0.1, 'by default, start reseting near-far plane at 50%')
flags.DEFINE_float('bound_reset', 0.1, 'by default, start reseting bound from 50%')
flags.DEFINE_float('bound_factor', 2, 'by default, use a loose bound')

# optimization: initialization 
flags.DEFINE_bool('init_ellips', False, 'whether to init shape as ellips')
flags.DEFINE_integer('warmup_pose_ep', 0, 'epochs to pre-train cnn pose predictor')
flags.DEFINE_integer('warmup_shape_ep', 0, 'epochs to pre-train nerf')
flags.DEFINE_integer('warmup_skin_ep', 0, 'epochs to pre-train nerf skin')
flags.DEFINE_bool('warmup_rootmlp', False, 'whether to preset base root pose (compatible with expmlp root basis only)')
flags.DEFINE_bool('unc_filter', True, 'whether to filter root poses init with low uncertainty')

# optimization: fine-tuning
flags.DEFINE_bool('keep_pose_basis', True, 'keep pose basis when loading models at train time')
flags.DEFINE_bool('freeze_root', False, 'whether to freeze root body pose')
flags.DEFINE_bool('freeze_bg', False, 'whether to freeze background geometry (besides cam)')
flags.DEFINE_bool('freeze_bgcam',False, 'whether to freeze bg cameras')
flags.DEFINE_bool('root_stab', True, 'whether to stablize root at ft')
flags.DEFINE_bool('freeze_cvf',  False, 'whether to freeze canonical features')
flags.DEFINE_bool('freeze_shape',False, 'whether to freeze canonical shape')
flags.DEFINE_bool('freeze_skin',False, 'whether to freeze skinning weights')
flags.DEFINE_bool('freeze_proj', False, 'whether to freeze some params w/ proj loss')
flags.DEFINE_bool('freeze_body_mlp', False, 'whether to freeze body pose mlp')
flags.DEFINE_float('proj_start', 0.0, 'steps to strat projection opt')
flags.DEFINE_float('frzroot_start', 0.0, 'steps to strat fixing root pose')
flags.DEFINE_float('frzbody_end', 0.0,   'steps to end fixing body pose')
flags.DEFINE_float('proj_end', 0.2,  'steps to end projection opt')

# CSE fine-tuning (turned off by default)
flags.DEFINE_bool('ft_cse', False, 'whether to fine-tune cse features')
flags.DEFINE_bool('mt_cse', True,  'whether to maintain cse features')
flags.DEFINE_float('mtcse_steps', 0.0, 'only distill cse before several epochs')
flags.DEFINE_float('ftcse_steps', 0.0, 'finetune cse after several epochs')
flags.DEFINE_bool('train_cnn', False, 'whether to train feedforward cnns')
flags.DEFINE_bool('use_cnn', False, 'whether to use CNN basis for opt')
flags.DEFINE_bool('cnn_code', False, 'whether to use CNN basis for opt')

# render / eval
flags.DEFINE_integer('render_size', 64, 'size used for eval visualizations')
flags.DEFINE_integer('frame_chunk', 20, 'chunk size to split the input frames')
flags.DEFINE_integer('chunk', 32*1024, 'chunk size to split the input to avoid OOM')
flags.DEFINE_integer('rnd_frame_chunk', 3, 'chunk size to render eval images')
flags.DEFINE_bool('queryfw', True, 'use forward warping to query deformed shape')
flags.DEFINE_float('mc_threshold', -0.002, 'marching cubes threshold')
flags.DEFINE_bool('full_mesh', False, 'extract surface without visibility check')
flags.DEFINE_bool('ce_color', True, 'assign mesh color as canonical surface mapping or radiance')
flags.DEFINE_integer('sample_grid3d', 64, 'resolution for mesh extraction from nerf')
flags.DEFINE_string('test_frames', '9', 'a list of video index or num of frames, {0,1,2}, 30')

# losses
flags.DEFINE_bool('use_embed', True, 'whether to use feature consistency losses')
flags.DEFINE_bool('use_proj', True, 'whether to use reprojection loss')
flags.DEFINE_bool('use_corresp', True, 'whether to render and compare correspondence')
flags.DEFINE_bool('dist_corresp', True, 'whether to render distributed corresp')
flags.DEFINE_float('total_wt', 1, 'by default, multiple total loss by 1')
flags.DEFINE_float('sil_wt', 0.1, 'weight for silhouette loss')
flags.DEFINE_float('img_wt',  0.1, 'weight for silhouette loss')
flags.DEFINE_float('feat_wt', 0., 'by default, multiple feat loss by 1')
flags.DEFINE_float('frnd_wt', 1., 'by default, multiple feat loss by 1')
flags.DEFINE_float('proj_wt', 0.02, 'by default, multiple proj loss by 1')
flags.DEFINE_float('flow_wt', 1, 'by default, multiple flow loss by 1')
flags.DEFINE_float('cyc_wt', 1, 'by default, multiple cyc loss by 1')
flags.DEFINE_float('dfm_wt', 1, 'by default, multiple deform delta loss by 50')
flags.DEFINE_bool('rig_loss', False,'whether to use globally rigid loss')
flags.DEFINE_bool('root_sm', True, 'whether to use smooth loss for root pose')
flags.DEFINE_float('eikonal_wt', 0., 'weight of eikonal loss')
flags.DEFINE_bool('eik_init', False, 'whether to use a relaxed version of eik')
flags.DEFINE_float('arap_wt', 0, 'weight of arap loss')
flags.DEFINE_float('bone_loc_reg', 0.1, 'use bone location regularization')
flags.DEFINE_float('rsdf_wt', 0.2, 'use sdf loss')
flags.DEFINE_float('bone_len_reg', 0., 'use bone length regularization')
flags.DEFINE_float('rest_angle_wt', 0.01, 'use rest joint angle reg')
flags.DEFINE_bool('loss_flt', True, 'whether to use loss filter')
flags.DEFINE_bool('rm_novp', True,'whether to remove loss on non-overlapping pxs')

# for scripts/visualize/match.py
flags.DEFINE_string('match_frames', '0 1', 'a list of frame index')

# for compatibility with vidyn
flags.DEFINE_boolean('rollout', False, 'rollout stage')


class banmo(nn.Module):
    def __init__(self, opts, data_info):
        super(banmo, self).__init__()
        self.opts = opts
        self.device = torch.device("cuda:%d"%opts.local_rank)
        self.config = configparser.RawConfigParser()
        self.config.read('configs/%s.config'%opts.seqname)
        self.alpha=torch.Tensor([opts.alpha])
        self.alpha=nn.Parameter(self.alpha)
        self.loss_select = 1 # by default,  use all losses
        self.root_update = 1 # by default, update root pose
        self.body_update = 1 # by default, update body pose
        self.shape_update = 1 # by default, update shape
        self.cvf_update = 1 # by default, update feat
        self.skin_update = 1 # by default, update skin
        self.progress = 0. # also reseted in optimizer
        self.counter_frz_rebone = 0. # counter to freeze params for reinit bones
        self.counter_frz_randrt = 0. # counter to freeze params for rand root noise
        self.use_fine = False # by default not using fine samples
        #self.ndepth_bk = opts.ndepth # original ndepth
        self.root_basis = opts.root_basis
        self.use_cam = opts.use_cam
        self.is_warmup_pose = False # by default not warming up
        self.img_size = opts.img_size # current rendering size, 
                                      # have to be consistent with dataloader, 
                                      # eval/train has different size
        embed_net = nn.Embedding
        
        # multi-video mode
        self.num_vid =  len(self.config.sections())-1
        self.data_offset = data_info['offset']
        self.num_fr=self.data_offset[-1]  
        self.max_ts = (self.data_offset[1:] - self.data_offset[:-1]).max()
        self.impath      = data_info['impath']
        self.latest_vars = {}
        # only used in get_near_far: rtk, idk
        # only used in visibility: rtk, vis, idx (deprecated)
        # raw rot/trans estimated by pose net
        self.latest_vars['rt_raw'] = np.zeros((self.data_offset[-1], 3,4)) # from data
        # rtk raw scaled and refined
        self.latest_vars['rtk'] = np.zeros((self.data_offset[-1], 4,4))
        self.latest_vars['idk'] = np.zeros((self.data_offset[-1],))
        self.latest_vars['mesh_rest'] = trimesh.Trimesh()
        if opts.lineload:
            #TODO todo, this should be idx512,-1
            self.latest_vars['fp_err'] =       np.zeros((self.data_offset[-1]*opts.img_size,2)) # feat, proj
            self.latest_vars['flo_err'] =      np.zeros((self.data_offset[-1]*opts.img_size,6)) 
            self.latest_vars['sil_err'] =      np.zeros((self.data_offset[-1]*opts.img_size,)) 
            self.latest_vars['flo_err_hist'] = np.zeros((self.data_offset[-1]*opts.img_size,6,10))
        else:
            self.latest_vars['fp_err'] =       np.zeros((self.data_offset[-1],2)) # feat, proj
            self.latest_vars['flo_err'] =      np.zeros((self.data_offset[-1],6)) 
            self.latest_vars['sil_err'] =      np.zeros((self.data_offset[-1],)) 
            self.latest_vars['flo_err_hist'] = np.zeros((self.data_offset[-1],6,10))

        # get near-far plane
        self.near_far = np.zeros((self.data_offset[-1],2))
        self.near_far[...,1] = 6.
        self.near_far = self.near_far.astype(np.float32)
        self.near_far = torch.Tensor(self.near_far).to(self.device)
        self.obj_scale = float(near_far_to_bound(self.near_far)) / 0.3 # to 0.3
        self.near_far = self.near_far / self.obj_scale
        self.near_far_base = self.near_far.clone() # used for create_base_se3()
        self.near_far = nn.Parameter(self.near_far)
    
        # object bound
        self.latest_vars['obj_bound'] = np.asarray([1.,1.,1.])
        self.latest_vars['obj_bound'] *= near_far_to_bound(self.near_far)

        self.vis_min=np.asarray([[0,0,0]])
        self.vis_len=self.latest_vars['obj_bound']/2
        
        # set shape/appearancce model
        self.num_freqs = opts.num_freq
        in_channels_xyz=3+3*self.num_freqs*2
        in_channels_dir=27
        if opts.env_code:
            # add video-speficit environment lighting embedding
            env_code_dim = 64
            if opts.env_fourier and not opts.cnn_code:
                self.env_code = FrameCode(self.num_freqs, env_code_dim, self.data_offset, scale=1)
            else:
                self.env_code = embed_net(self.num_fr, env_code_dim)
        else:
            env_code_dim = 0
        if opts.use_category:
            # add video-specific shape code
            vid_code_dim=32  
            vid_shape_code = embed_net(self.num_vid, vid_code_dim)
            vid_shape_code.weight.data[:] = 0
        else:
            vid_shape_code = None 
        self.nerf_coarse = NeRF(in_channels_xyz=in_channels_xyz, 
                                in_channels_dir=env_code_dim,
                                init_beta=opts.init_beta, vid_code=vid_shape_code)
        self.embedding_xyz = Embedding(3,self.num_freqs,alpha=self.alpha.data[0])
        self.embedding_dir = Embedding(3,4,             alpha=self.alpha.data[0])
        self.embeddings = {'xyz':self.embedding_xyz, 'dir':self.embedding_dir}
        self.nerf_models= {'coarse':self.nerf_coarse}

        if opts.bgmlp == 'nerf':
            # 3d nerf
            vid_code_dim=32  
            vid_code_bg = embed_net(self.num_vid, vid_code_dim)
            vid_code_bg.weight.data[:] = 0
            self.trsi_2d = NeRFBG(num_freqs=self.num_freqs, 
                                data_offset = self.data_offset,
                                config = self.config,
                                opts = opts,
                                in_channels_xyz=in_channels_xyz, 
                                in_channels_dir=in_channels_dir,
                                init_beta=opts.init_beta,
                                vid_code=vid_code_bg)
        elif opts.bgmlp == 'hmnerf':
            # 3d nerf
            vid_code_dim=32  
            vid_code_bg = embed_net(self.num_vid, vid_code_dim)
            vid_code_bg.weight.data[:] = 0
            self.trsi_2d = NeRFBG(num_freqs=self.num_freqs, 
                                data_offset = self.data_offset,
                                config = self.config,
                                opts = opts,
                                num_vid = self.num_vid,D=6,color_act=False,
                                in_channels_xyz=in_channels_xyz, 
                                in_channels_dir=in_channels_dir+env_code_dim,
                                init_beta=opts.init_beta,
                                vid_code=vid_code_bg)
        elif opts.bgmlp == 'mlp':
            # 2d nerf
            trsi_tcode_dim = 21
            self.trsi_2d = NeRFTransient(num_freqs=6, 
                tcode_dim = trsi_tcode_dim, data_offset = self.data_offset,
                in_channels=26+trsi_tcode_dim)

        # set motion model
        t_embed_dim = opts.t_embed_dim
        if opts.frame_code:
            self.pose_code = FrameCode(self.num_freqs, t_embed_dim, self.data_offset)
        else:
            self.pose_code = embed_net(self.num_fr, t_embed_dim)
        self.rest_pose_code = embed_net(1, t_embed_dim)
        self.nerf_models['rest_pose_code'] = self.rest_pose_code

        if opts.lbs:
            # pre-defined skeleton
            if opts.pre_skel!="":
                if opts.pre_skel=="a1":
                    urdf_path='mesh_material/a1/urdf/a1.urdf'
                elif opts.pre_skel=="wolf":
                    urdf_path='mesh_material/wolf.urdf'
                elif opts.pre_skel=="wolf_mod":
                    urdf_path='mesh_material/wolf_mod.urdf'
                elif opts.pre_skel=="laikago":
                    urdf_path='mesh_material/laikago/laikago.urdf'
                elif opts.pre_skel=="human":
                    urdf_path='mesh_material/human.urdf' 
                elif opts.pre_skel=="human_mod":
                    urdf_path='mesh_material/human_mod.urdf'
                robot = URDFRobot(urdf_path=urdf_path)
                self.sim3 = nn.Parameter(robot.sim3) # used in two places for grad (1) sinkhorn (2) fk
                opts.num_bones = robot.num_bones
                self.nerf_body_rts = SkelHead(urdf=robot.urdf,joints=robot.joints,
                      sim3=self.sim3, rest_angles=robot.rest_angles, 
                                pose_code = self.pose_code,
                           rest_pose_code = self.rest_pose_code,
                                data_offset=self.data_offset,
                                in_channels=t_embed_dim,
                                out_channels=robot.num_dofs)
                self.robot = self.nerf_body_rts
                bones = compute_bone_from_joint(self, is_init=True)
                self.bones = nn.Parameter(bones)
                self.nerf_models['bones'] = self.bones
                #self.name2bpos = {
                #        'FR_hip_joint': [-0.018853, -0.000279, 0.004995],
                #        'FR_thigh_joint': [-0.020761, -0.012507, 0.010323], 
                #        'FR_calf_joint': [-0.020137, -0.035019, 0.007742],
                #        'FL_hip_joint': [0.008887, -0.001229, 0.002356], 
                #        'FL_thigh_joint': [0.008884, -0.012507, 0.010323], 
                #        'FL_calf_joint': [0.007275, -0.032517, 0.007742], 
                #        'RR_hip_joint': [-0.018666, -0.002523, -0.035547], 
                #        'RR_thigh_joint': [-0.020184, -0.017509, -0.036129], 
                #        'RR_calf_joint': [-0.019990, -0.042523, -0.036129], 
                #        'RL_hip_joint': [0.007119, -0.004024, -0.033229], 
                #        'RL_thigh_joint': [0.006982, -0.020011, -0.036129], 
                #        'RL_calf_joint':[0.005201, -0.040022, -0.033548]}
                self.name2bpos = {
                'FR_hip_joint': [-0.015739, -0.003684, 0.009900],
                'FR_thigh_joint': [-0.014420, -0.018653, 0.011545], 
                'FR_calf_joint': [-0.014562, -0.040038, 0.010863],
                'FL_hip_joint': [0.011860, -0.003634, 0.009793], 
                'FL_thigh_joint': [0.009794, -0.018945, 0.011812], 
                'FL_calf_joint': [0.009838, -0.040435, 0.010889], 
                'RR_hip_joint': [-0.016750, -0.004093, -0.034878], 
                'RR_thigh_joint': [-0.015498, -0.018688, -0.036901], 
                'RR_calf_joint': [-0.014790, -0.037194, -0.037433], 
                'RL_hip_joint': [0.009597, -0.004193, -0.033975], 
                'RL_thigh_joint': [0.009639, -0.018240, -0.035562], 
                'RL_calf_joint': [0.009476, -0.041826, -0.033747]}
            else:
                self.nerf_body_rts = RTHead(use_quat=False, 
                                pose_code = self.pose_code,
                                in_channels=t_embed_dim,
                                out_channels=6*opts.num_bones)
                self.bones = nn.Parameter(torch.zeros(opts.num_bones,10))
                sphere_mesh = trimesh.creation.uv_sphere(radius=0.03)
                reinit_bones(self, sphere_mesh, opts.num_bones)

            #TODO scale+constant parameters
            skin_aux = torch.Tensor([0,0]) 
            self.skin_aux = nn.Parameter(skin_aux)
            self.nerf_models['skin_aux'] = self.skin_aux

            if opts.nerf_skin != "":
                if opts.nerf_skin == 'mmlp':
                    self.nerf_skin = MultiMLP(num_net=opts.num_bones,
                                        in_channels=3+t_embed_dim,
                                        D=2,W=16,out_channels=1) 
                elif opts.nerf_skin == 'cmlp':
                    self.nerf_skin = BaseMLP(in_channels=3*opts.num_bones+t_embed_dim,
                                        D=5,W=64,out_channels=opts.num_bones)
                elif opts.nerf_skin == 'mlp':
                    self.nerf_skin = BaseMLP(in_channels=in_channels_xyz+t_embed_dim,
                                        D=5,W=64,out_channels=opts.num_bones)
                self.nerf_skin.skin_type = opts.nerf_skin
                self.nerf_models['nerf_skin'] = self.nerf_skin

        if opts.mlp_deform:
            from nnutils.nvp import NVP
            dfm_code_dim=16
            self.mlp_deform = NVP(
            n_layers=2,
            feature_dims=dfm_code_dim,
            hidden_size=[32, 16, 16, 8, 8],
            proj_dims=32,
            code_proj_hidden_size=[32,32,32],
            proj_type='simple',
            block_normalize=False,
            normalization=False)
            self.nerf_models['mlp_deform'] = self.mlp_deform

            #self.dfm_code = embed_net(self.num_vid, dfm_code_dim)
            #self.dfm_code.weight.data[:] = 0
            self.dfm_code = FrameCode(self.num_freqs, dfm_code_dim, self.data_offset)
            self.nerf_models['dfm_code'] = self.dfm_code

        # set visibility nerf
        if opts.nerf_vis:
            self.nerf_vis = BaseMLP(in_channels=in_channels_xyz, D=5, W=64, 
                                    out_channels=1)
            self.nerf_models['nerf_vis'] = self.nerf_vis
        
        # optimize camera
        if opts.root_opt:
            if self.use_cam: 
                use_quat=False
                out_channels=6
            else:
                use_quat=True
                out_channels=7
            # train a cnn pose predictor for warmup
            cnn_in_channels = 16

            cnn_head = RTHead_old(use_quat=True, D=1,
                        in_channels_xyz=128,in_channels_dir=0,
                        out_channels=7, raw_feat=True)
            self.dp_root_rts = nn.Sequential(
                            Encoder((112,112), in_channels=cnn_in_channels,
                                out_channels=128), cnn_head)
            if self.root_basis == 'cnn':
                self.nerf_root_rts = nn.Sequential(
                                Encoder((112,112), in_channels=cnn_in_channels,
                                out_channels=128),
                                RTHead(use_quat=use_quat, D=1,
                                in_channelsi=128,
                                out_channels=out_channels))
            elif self.root_basis == 'exp':
                self.nerf_root_rts = RTExplicit(self.num_fr, delta=self.use_cam)
            elif self.root_basis == 'expmlp':
                self.nerf_root_rts = RTExpMLP(self.num_fr, 
                                  self.num_freqs,t_embed_dim,self.data_offset,
                                  delta=self.use_cam)
            elif self.root_basis == 'mlp':
                self.root_code = embed_net(self.num_fr, t_embed_dim)
                output_head =   RTHead(use_quat=use_quat, 
                            in_channels=t_embed_dim,
                            out_channels=out_channels)
                self.nerf_root_rts = nn.Sequential(self.root_code, output_head)
            else: print('error'); exit()

        # intrinsics
        ks_list = []
        for i in range(self.num_vid):
            fx,fy,px,py=[float(i) for i in \
                    self.config.get('data_%d'%i, 'ks').split(' ')]
            ks_list.append([fx,fy,px,py])
        self.ks_param = torch.Tensor(ks_list).to(self.device)
        if opts.ks_opt:
            self.ks_param = nn.Parameter(self.ks_param)
            

        # densepose
        detbase='./third_party/detectron2/'
        if opts.use_human:
            canonical_mesh_name = 'smpl_27554'
            config_path = '%s/projects/DensePose/configs/cse/densepose_rcnn_R_101_FPN_DL_soft_s1x.yaml'%(detbase)
            weight_path = 'https://dl.fbaipublicfiles.com/densepose/cse/densepose_rcnn_R_101_FPN_DL_soft_s1x/250713061/model_final_1d3314.pkl'
        else:
            canonical_mesh_name = 'sheep_5004'
            config_path = '%s/projects/DensePose/configs/cse/densepose_rcnn_R_50_FPN_soft_animals_CA_finetune_4k.yaml'%(detbase)
            weight_path = 'https://dl.fbaipublicfiles.com/densepose/cse/densepose_rcnn_R_50_FPN_soft_animals_CA_finetune_4k/253498611/model_final_6d69b7.pkl'
        canonical_mesh_path = 'mesh_material/%s_sph.pkl'%canonical_mesh_name
        
        with open(canonical_mesh_path, 'rb') as f:
            dp = pickle.load(f)
            self.dp_verts = dp['vertices']
            self.dp_faces = dp['faces'].astype(int)
            self.dp_verts = torch.Tensor(self.dp_verts).cuda(self.device)
            self.dp_faces = torch.Tensor(self.dp_faces).cuda(self.device).long()
            
            self.dp_verts -= self.dp_verts.mean(0)[None]
            #self.dp_verts /= self.dp_verts.abs().max()
            self.dp_verts = F.normalize(self.dp_verts, 2,-1) #TODO unit sphere
            self.dp_verts_unit = self.dp_verts.clone()
            self.dp_verts *= (self.near_far[:,1] - self.near_far[:,0]).mean()/2
            
            # visualize
            self.dp_vis = self.dp_verts.detach()
            self.dp_vmin = self.dp_vis.min(0)[0][None]
            self.dp_vis = self.dp_vis - self.dp_vmin
            self.dp_vmax = self.dp_vis.max(0)[0][None]
            self.dp_vis = self.dp_vis / self.dp_vmax

            # save colorvis
            trimesh.Trimesh(self.dp_verts_unit.cpu().numpy(), 
                            dp['faces'], 
                            vertex_colors = self.dp_vis.cpu().numpy())\
                            .export('tmp/%s.obj'%canonical_mesh_name)
            
            if opts.unc_filter:
                pass
                #from utils.cselib import create_cse
                ## load surface embedding
                #_, _, mesh_vertex_embeddings = create_cse(config_path,
                #                                                weight_path)
                #self.dp_embed = mesh_vertex_embeddings[canonical_mesh_name]

        # add densepose mlp
        if opts.use_embed:
            self.num_feat = 16
            # TODO change this to D-8
            self.nerf_feat = NeRFFeat(init_beta=1., in_channels=in_channels_xyz,
                         D=5, W=128, out_channels=self.num_feat)
            self.nerf_models['nerf_feat'] = self.nerf_feat

            if opts.ft_cse:
                from nnutils.cse import CSENet
                self.csenet = CSENet(ishuman=opts.use_human)

        # add dp projection layer
        if opts.dp_proj: self.dp_proj = ConvProj(self.num_feat)

        # train feedforward nets
        if opts.train_cnn or opts.use_cnn:
            self.cnn_ff = BANMoCNN(cnn_in_channels = 3)

        # add uncertainty MLP
        if opts.use_unc:
            # input, (x,y,t)+code, output, (1)
            vid_code_dim=32  # add video-specific code
            self.vid_code = embed_net(self.num_vid, vid_code_dim)
            self.nerf_unc = BaseMLP( in_channels=in_channels_xyz+vid_code_dim, 
                                     D=8, W=256, out_channels=1)
            self.nerf_models['nerf_unc'] = self.nerf_unc

        if opts.warmup_pose_ep>0:
            # soft renderer
            import soft_renderer as sr
            self.mesh_renderer = sr.SoftRenderer(image_size=256, sigma_val=1e-12, 
                           camera_mode='look_at',perspective=False, aggr_func_rgb='hard',
                           light_mode='vertex', light_intensity_ambient=1.,light_intensity_directionals=0.)


        if opts.phys_opt:
            sys.path.insert(0,"vidyn")
            from env_utils.warp_env import Scene
            from env_utils.vis import Logger
            if opts.phys_vid=='':
                opts.phys_vid = list(range(self.num_vid))
            else:
                opts.phys_vid = [int(i) for i in opts.phys_vid.split(',')]
            model_dict = {}
            model_dict['bg_rts'] = self.trsi_2d
            model_dict['nerf_root_rts'] = self.nerf_root_rts
            model_dict['nerf_body_rts'] = self.nerf_body_rts
            model_dict['ks_params'] = self.ks_param
            #self.phys_env = Scene(opts, model_dict, dt=0.0005, use_dr=True) # 400 steps
            self.phys_env = Scene(opts, model_dict, dt=0.001, use_dr=True) # 200 steps
            self.env_vis = Logger(opts)

    def forward_default(self, batch):
        opts = self.opts
        # get root poses
        if (not opts.train_cnn) and (not opts.use_cnn):
            rtk_all = self.compute_rts()
            rtk_np = rtk_all.clone().detach().cpu().numpy()
            valid_rts = self.latest_vars['idk'].astype(bool)
            self.latest_vars['rtk'][valid_rts,:3] = rtk_np[valid_rts]

        # change near-far plane for all views
        if self.progress>=opts.nf_reset:
            self.near_far.data = get_near_far(
                                          self.near_far.data,
                                          self.latest_vars)

        if opts.debug:
            torch.cuda.synchronize()
            start_time = time.time()
        if opts.lineload:
            bs = self.set_input(batch, load_line=True)
        else:
            bs = self.set_input(batch)
        
        if opts.debug:
            torch.cuda.synchronize()
            print('set input time:%.2f'%(time.time()-start_time))
        rtk = self.rtk
        kaug= self.kaug
        embedid=self.embedid
        aux_out={}
        
        # Render
        rendered, rand_inds = self.nerf_render(rtk, kaug, embedid, 
                nsample=opts.nsample, ndepth=opts.ndepth)
        
        if opts.debug:
            torch.cuda.synchronize()
            print('set input + render time:%.2f'%(time.time()-start_time))

        # image and silhouette loss
        sil_at_samp = rendered['sil_at_samp']
        bgsil_at_samp = rendered['bgsil_at_samp']
        sil_at_samp_flo = rendered['sil_at_samp_flo']
        vis_at_samp = rendered['vis_at_samp']

        if opts.loss_flt:
            # frame-level rejection of bad segmentations
            if opts.lineload:
                invalid_idx = loss_filter_line(self.latest_vars['sil_err'],
                                               self.errid.long(),self.frameid.long(),
                                               rendered['sil_loss_samp']*opts.sil_wt,
                                               opts.img_size)
            else:
                sil_err, invalid_idx = loss_filter(self.latest_vars['sil_err'], 
                                             rendered['sil_loss_samp']*opts.sil_wt,
                                             sil_at_samp>-1, scale_factor=10)
                self.latest_vars['sil_err'][self.errid.long()] = sil_err
            
            if self.progress > (opts.warmup_steps):
                rendered['sil_loss_samp'][invalid_idx] *= 0.
                if invalid_idx.sum()>0:
                    print('%d removed from sil'%(invalid_idx.sum()))
        
        img_loss_samp = opts.img_wt*rendered['img_loss_samp']
        if opts.loss_flt:
            img_loss_samp[invalid_idx] *= 0
        img_loss = img_loss_samp
        if opts.rm_novp and (not opts.use_compose):
            img_loss = img_loss * rendered['sil_coarse'].detach()
        if opts.use_compose:
            img_loss = img_loss.mean() # eval on all pts
        else:
            img_loss = img_loss[sil_at_samp[...,0]>0].mean() # eval on valid pts
        sil_loss_samp = opts.sil_wt*rendered['sil_loss_samp']
        sil_loss = sil_loss_samp[vis_at_samp>0].mean()
        aux_out['sil_loss'] = sil_loss
        aux_out['img_loss'] = img_loss
        total_loss = img_loss
        total_loss = total_loss + sil_loss 
        
        if opts.bgmlp != "" and (not opts.use_compose):
            bg_loss = (rendered['img_at_samp'] - rendered['bg_coarse']).pow(2).mean(-1)
            bg_loss = bg_loss[bgsil_at_samp[...,0]>0].mean() # eval on valid pts
            aux_out['bg_loss'] = bg_loss
            total_loss = total_loss + bg_loss
          
        if opts.use_embed:
            # feat rnd loss
            frnd_loss_samp = opts.frnd_wt*rendered['frnd_loss_samp']
            if opts.loss_flt:
                frnd_loss_samp[invalid_idx] *= 0
            if opts.rm_novp:
                frnd_loss_samp = frnd_loss_samp * rendered['sil_coarse'].detach()
            feat_rnd_loss = frnd_loss_samp[sil_at_samp[...,0]>0].mean() # eval on valid pts
            aux_out['feat_rnd_loss'] = feat_rnd_loss
            total_loss = total_loss + feat_rnd_loss
  
        # viser loss
        if opts.use_embed:
            feat_err_samp = rendered['feat_err']* opts.feat_wt
            if opts.loss_flt:
                feat_err_samp[invalid_idx] *= 0
            
            feat_loss = feat_err_samp
            if opts.rm_novp:
                feat_loss = feat_loss * rendered['sil_coarse'].detach()
            feat_loss = feat_loss[sil_at_samp>0].mean()
            total_loss = total_loss + feat_loss
            aux_out['feat_loss'] = feat_loss
            aux_out['beta_feat'] = self.nerf_feat.beta.clone().detach()[0]

        
        if opts.use_proj:
            proj_err_samp = rendered['proj_err']* opts.proj_wt
            #if opts.freeze_proj:
            #    proj_err_samp[rendered['match_unc'] > 0.8] = 0
            if opts.loss_flt:
                proj_err_samp[invalid_idx] *= 0

            proj_loss = proj_err_samp[sil_at_samp>0].mean()
            aux_out['proj_loss'] = proj_loss
            if opts.freeze_proj:
                total_loss = total_loss + proj_loss
                ## warm up by only using projection loss to optimize bones
                warmup_weight = (self.progress - opts.proj_start)/(opts.proj_end-opts.proj_start)
                warmup_weight = (warmup_weight - 0.8) * 5 #  [-4,1]
                warmup_weight = np.clip(warmup_weight, 0,1)
                if (self.progress > opts.proj_start and \
                    self.progress < opts.proj_end):
                    total_loss = total_loss*warmup_weight + \
                               10*proj_loss*(1-warmup_weight)
            else:
                # only add it after feature volume is trained well
                total_loss = total_loss + proj_loss
        
        # flow loss
        if opts.use_corresp:
            flo_loss_samp = rendered['flo_loss_samp']
            if opts.loss_flt:
                flo_loss_samp[invalid_idx] *= 0
            if opts.rm_novp:
                flo_loss_samp = flo_loss_samp * rendered['sil_coarse'].detach()

            # eval on valid pts
            flo_loss = flo_loss_samp[sil_at_samp_flo[...,0]].mean() * 2
            #flo_loss = flo_loss_samp[sil_at_samp_flo[...,0]].mean()
            flo_loss = flo_loss * opts.flow_wt
    
            # warm up by only using flow loss to optimize root pose
            if self.loss_select == 0:
                total_loss = total_loss*0. + flo_loss
            else:
                total_loss = total_loss + flo_loss
            aux_out['flo_loss'] = flo_loss
        
        # regularization 
        if 'frame_cyc_dis' in rendered.keys():
            # cycle loss
            cyc_loss = rendered['frame_cyc_dis'].mean()
            total_loss = total_loss + cyc_loss * opts.cyc_wt
            #total_loss = total_loss + cyc_loss*0
            aux_out['cyc_loss'] = cyc_loss

            # globally rigid prior
            rig_loss = 0.0001*rendered['frame_rigloss'].mean()
            if opts.rig_loss:
                total_loss = total_loss + rig_loss
            else:
                total_loss = total_loss + rig_loss*0
            aux_out['rig_loss'] = rig_loss

            # elastic energy for se3 field / translation field
            if 'elastic_loss' in rendered.keys():
                elastic_loss = rendered['elastic_loss'].mean() * 1e-3
                total_loss = total_loss + elastic_loss
                aux_out['elastic_loss'] = elastic_loss

            # arap loss for deformation
            if 'arap_loss' in rendered.keys():
                arap_loss = rendered['arap_loss']
                arap_loss = arap_loss[arap_loss>0].mean() * opts.arap_wt
                total_loss = total_loss + arap_loss
                aux_out['arap_loss'] = arap_loss

        # regularization of root poses
        if opts.root_sm:
            root_sm_loss = compute_root_sm_2nd_loss(rtk_all, self.data_offset)
            aux_out['root_sm_loss'] = root_sm_loss
            total_loss = total_loss + root_sm_loss


        if opts.eikonal_wt > 0:
            ekl_loss = rendered['eikonal_loss']
            ekl_loss = ekl_loss[ekl_loss>0].mean()
            ekl_loss = opts.eikonal_wt*ekl_loss
            aux_out['ekl_loss'] = ekl_loss
            total_loss = total_loss + ekl_loss

        #if opts.pre_skel=='a1' or opts.pre_skel=='laikago':
        #    # use annotated 3d keypoints
        #    rest_joints, rest_angles = self.nerf_body_rts.forward_abs()
        #    rest_joints = rest_joints.view(-1,12)[:,9:]
        #    bones_rst = self.bones
        #    bones_rst,_ = zero_to_rest_bone(self, bones_rst)
        #    k3d_loss = 0
        #    for k,v in self.name2bpos.items():
        #        v = torch.Tensor(v).to(self.device)
        #        idx = self.robot.urdf.name2joints_idx[k]+1
        #        k3d_loss += (bones_rst[idx,:3] - v).pow(2).sum()
        #    total_loss = total_loss + k3d_loss
        #    aux_out['k3d_loss'] = k3d_loss

        #    # force joints to be inside surface
        #    from nnutils.geom_utils import evaluate_mlp
        #    joints_embed =self.embedding_xyz(rest_joints)[:,None]
        #    jsdf = evaluate_mlp(self.nerf_coarse, joints_embed , sigma_only=True)
        #    jsdf_loss = F.relu(-jsdf).mean()
        #    total_loss = total_loss + jsdf_loss
        #    aux_out['jsdf_loss'] = jsdf_loss

        # bone location regularization: pull bones away from empth space (low sdf)
        if opts.lbs and opts.bone_loc_reg>0:
            mesh_rest = self.latest_vars['mesh_rest']
            if len(mesh_rest.vertices)>100: # not a degenerate mesh
                mesh_verts = [torch.Tensor(mesh_rest.vertices).to(self.device)]
                mesh_faces = [torch.Tensor(mesh_rest.faces).to(self.device).long()]
                #TODO deform the mesh
                if opts.mlp_deform:
                    num_pts = mesh_verts[0].shape[0]
                    query_time = torch.ones(num_pts,1,device=self.device).long()
                    query_time = query_time*np.random.randint(0,self.num_fr)
                    dfm_code = self.dfm_code(query_time)[:,None]
                    mesh_verts[0] = self.mlp_deform.forward(dfm_code,
                                                mesh_verts[0][:,None,None])[:,0,0]
                    #trimesh.Trimesh(mesh_verts[0].detach().cpu()).export('tmp/0.obj')
                if opts.pre_skel!="":
                    if opts.bone_len_reg>0:
                        bone_len_loss = self.nerf_body_rts.jlen_scale_z.pow(2).mean()
                        bone_len_loss = bone_len_loss * opts.bone_len_reg
                        total_loss = total_loss + bone_len_loss
                        aux_out['bone_len_loss'] = bone_len_loss

                    rest_rbrt, rest_angles = self.nerf_body_rts.forward_abs()
                    #TODO currently match the same shape to original bone length
                    # may want to extract different meshes across videos

                    # rest regularization
                    target_angles = self.nerf_body_rts.rest_angles.to(self.device)
                    rest_angle_loss = (target_angles - rest_angles).pow(2).mean()
                    if opts.rest_angle_wt>0:
                        total_loss = total_loss + rest_angle_loss * opts.rest_angle_wt
                    aux_out['rest_angle_loss'] = rest_angle_loss

                    rest_rbrt = rest_rbrt.view(-1,12)
                    rest_se3 = torch.eye(4).to(self.device)
                    rest_se3 = rest_se3[None].repeat(len(rest_rbrt),1,1)
                    rest_se3[:,:3,:3] = rest_rbrt[:,:9].view(-1,3,3)
                    rest_se3[:,:3,3] = rest_rbrt[:,9:]

                    # can to robot
                    center, orient, scale = vec_to_sim3(self.robot.sim3)
                    #TODO do not update center/orient of the canonical bone
                    center = center.detach()
                    orient = orient.detach()
                    se3 = torch.eye(4).to(self.device)
                    se3[:3,:3] = orient.T
                    se3[:3,3:] = -orient.T@center[:,None]
                    rest_rbrt = se3[None] @ rest_se3
                    rest_rot = rest_rbrt[:,:3,:3]
                    rest_trn = rest_rbrt[:,:3,3:] / scale[None,:,None]

                    # sink-horn loss
                    verts_part = []
                    faces_part = []
                    num_prev_verts = 0
                    for link in self.robot.urdf._reverse_topo:
                        if len(link.visuals)==0: continue
                        path = self.robot.urdf._paths_to_base[link]
                        if len(path)>1:
                            joint = self.robot.urdf._G.get_edge_data(path[0], path[1])['joint']
                            idx = self.robot.urdf.name2query_idx[joint.name] + 1 # for link
                        else:
                            idx = 0
                        # get robot parts
                        #mesh_part = link.visuals[0].geometry.meshes[0]
                        mesh_part = link.collision_mesh # this transforms to link origin
                        verts = torch.cuda.FloatTensor(mesh_part.vertices)
                        faces = torch.cuda.LongTensor(mesh_part.faces   )
                        # apply transforms
                        verts = verts.matmul(rest_rot[idx].T) + rest_trn[idx].T

                        verts_part.append(verts)
                        faces_part.append(faces + num_prev_verts)
                        num_prev_verts += len(verts)

                    verts_part = torch.cat(verts_part,0)
                    faces_part = torch.cat(faces_part,0)
                    # robot to can
                    verts_part = verts_part*scale[None]
                    verts_part = verts_part@orient.T
                    verts_part = verts_part+center[None]
                    # sample points
                    mesh_robot= pytorch3d.structures.meshes.Meshes(
                                    verts=verts_part[None], faces=faces_part[None])
                    robot_samp = pytorch3d.ops.sample_points_from_meshes(mesh_robot,
                                                1000, return_normals=False)
                    #trimesh.Trimesh(robot_samp[0].detach().cpu()).export('tmp/0.obj')
                    bones_target = robot_samp[0]

                    # force robot points to be inside surface
                    robot_pts_embed =self.embedding_xyz(bones_target)[:,None]
                    if opts.use_category:
                        rand_id = torch.randint(self.num_vid, robot_pts_embed.shape[:-1])
                        rand_id = rand_id.to(self.device)
                        rsdf = evaluate_mlp(self.nerf_coarse, robot_pts_embed, vidid=rand_id)
                    else:
                        rsdf = evaluate_mlp(self.nerf_coarse, robot_pts_embed)
                    rsdf_loss = opts.rsdf_wt*F.relu(-rsdf).mean()  # out: negative
                    total_loss = total_loss + rsdf_loss
                    aux_out['rsdf_loss'] = rsdf_loss
                else:
                    bones_rst = self.bones
                    bones_rst,_ = zero_to_rest_bone(self, bones_rst)
                    bones_target = bones_rst[:,:3]

                try:
                    mesh_rest = pytorch3d.structures.meshes.Meshes(verts=mesh_verts, faces=mesh_faces)
                except:
                    mesh_rest = pytorch3d.structures.meshes.Meshes(verts=mesh_verts, faces=mesh_faces)
                shape_samp = pytorch3d.ops.sample_points_from_meshes(mesh_rest,
                                        1000, return_normals=False)
                shape_samp = shape_samp[0]#.to(self.device)
                from geomloss import SamplesLoss
                samploss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
                bone_loc_loss = samploss(bones_target*10, shape_samp*10)
                bone_loc_loss = opts.bone_loc_reg*bone_loc_loss
                total_loss = total_loss + bone_loc_loss
                aux_out['bone_loc_loss'] = bone_loc_loss

        # skinning weights reg
        if 'dskin' in rendered.keys():
            dskin_reg_loss = 0.001*rendered['dskin'].mean()
            total_loss = total_loss + dskin_reg_loss
            aux_out['dskin_reg_loss'] = dskin_reg_loss
        
        if 'eskin' in rendered.keys():
            eskin_reg_loss = 1e-4*rendered['eskin'].mean()
            total_loss = total_loss + eskin_reg_loss
            aux_out['eskin_reg_loss'] = eskin_reg_loss

        # deformation reg
        if 'deform_delta' in rendered.keys():
            dfm_reg_loss = rendered['deform_delta'].mean()
            total_loss = total_loss + opts.dfm_wt*dfm_reg_loss
            aux_out['dfm_reg_loss'] = dfm_reg_loss
 
        # visibility loss
        if 'vis_loss' in rendered.keys():
            vis_loss = 0.01*rendered['vis_loss'].mean()
            total_loss = total_loss + vis_loss
            aux_out['visibility_loss'] = vis_loss
        
        ## density decay loss
        #if 'density_decay_loss' in rendered.keys():
        #    density_decay_loss = 1e-3*rendered['density_decay_loss'].mean()
        #    total_loss = total_loss + density_decay_loss
        #    aux_out['density_decay_loss'] = density_decay_loss

        ## entropy
        #silp = rendered['sil_coarse']
        #silpn = 1-silp
        #silp = silp.clamp(1e-6, 1-1e-6)
        #silpn = silpn.clamp(1e-6, 1-1e-6)
        #entropy_sil = -silp*torch.log2(silp) - silpn*torch.log2(silpn)
        #entropy_loss = entropy_sil.mean()*2e-3
        #total_loss = total_loss + entropy_loss
        #aux_out['entropy_loss'] = entropy_loss

        #if opts.use_category:
        #    category_wt = self.nerf_coarse.vid_code.weight
        #    category_loss = (category_wt[None] - category_wt[:,None])
        #    category_loss = category_loss.pow(2).mean()
        #    total_loss = total_loss +  category_loss
        #    aux_out['category_loss'] = category_loss

        # uncertainty MLP inference
        if opts.use_unc:
            # add uncertainty MLP loss, loss = | |img-img_r|*sil - unc_pred |
            unc_pred = rendered['unc_pred']
            unc_rgb = sil_at_samp[...,0]*img_loss_samp.mean(-1)
            unc_feat= (sil_at_samp*feat_err_samp)[...,0]
            unc_proj= (sil_at_samp*proj_err_samp)[...,0]
            unc_sil = sil_loss_samp[...,0]
            #unc_accumulated = unc_feat + unc_proj
            #unc_accumulated = unc_feat + unc_proj + unc_rgb*0.1
#            unc_accumulated = unc_feat + unc_proj + unc_rgb
            unc_accumulated = unc_rgb
#            unc_accumulated = unc_rgb + unc_sil

            unc_loss = (unc_accumulated.detach() - unc_pred[...,0]).pow(2)
            unc_loss = unc_loss.mean()
            aux_out['unc_loss'] = unc_loss
            total_loss = total_loss + unc_loss


        # cse feature tuning
        if opts.ft_cse and opts.mt_cse:
            csenet_loss = (self.csenet_feats - self.csepre_feats).pow(2).sum(1)
            csenet_loss = csenet_loss[self.dp_feats_mask].mean()* 1e-5
            if self.progress < opts.mtcse_steps:
                total_loss = total_loss*0 + csenet_loss
            else:
                total_loss = total_loss + csenet_loss
            aux_out['csenet_loss'] = csenet_loss

        if opts.train_cnn: 
            # to get intrinsics within the bbox
            cnn_loss = compute_cnn_loss(self.cnn_ff, self.imgs, rtk.detach(), 
    rand_inds, rendered['depth_rnd'].detach(), rendered['sil_coarse'].detach(),aux_out)
            total_loss = total_loss + cnn_loss

        #if opts.phys_opt:
        #    #TODO add physics prior
        #    # pred delta
        #    from env_utils.warp_env import Scene, rotate_frame, se3_loss
        #    steps_fr = torch.linspace(0,self.num_fr-1, self.num_fr-1, device=self.device).long()
        #    #delta_ja_est = self.phys_env.delta_joint_est_mlp(steps_fr.reshape(-1,1))
        #    #delta_root = self.phys_env.delta_root_mlp(steps_fr.reshape(1,-1,1))

        #    # pred origin
        #    ja_est = self.nerf_body_rts.forward_abs(steps_fr.reshape(-1,1))[1]
        #    ja_phys = self.phys_env.cache_ja
        #    #ja_phys = delta_ja_est + ja_est
        #    root_est = self.phys_env.pred_est_q(steps_fr.reshape(1,-1)) 
        #    #root_phys = rotate_frame(self.phys_env.global_q, root_est) # in world space
        #    #root_phys = self.phys_env.compose_delta(root_phys, delta_root) # delta x target
        #    root_phys = self.phys_env.cache_q

        #    phys_pose_loss = 0.1*(ja_phys.detach() - ja_est).pow(2).mean()
        #    phys_root_loss = se3_loss(root_phys.detach(), root_est).mean()
        #    phys_loss = phys_pose_loss + phys_root_loss
        #    aux_out['phys_pose_loss'] = phys_pose_loss
        #    aux_out['phys_root_loss'] = phys_root_loss
        #    total_loss = total_loss + phys_loss*opts.phys_wt

        # save some variables
        if opts.lbs:
            aux_out['skin_scale'] = self.skin_aux[0].clone().detach()
            aux_out['skin_const'] = self.skin_aux[1].clone().detach()
        
        total_loss = total_loss * opts.total_wt
        aux_out['total_loss'] = total_loss
        aux_out['beta'] = self.nerf_coarse.beta.clone().detach()[0]
        if opts.debug:
            torch.cuda.synchronize()
            print('set input + render + loss time:%.2f'%(time.time()-start_time))
        return total_loss, aux_out
    
    def forward_phys(self, _):
        opts=self.opts
        #frameid = self.frameid.view(2,-1)[0].long().cuda()
        self.phys_env.total_steps = self.total_steps
        if self.total_steps%100==0:
            torch.cuda.empty_cache()
            self.phys_env.reinit_envs(1, wdw_length=30,is_eval=True)
            for vidid in self.opts.phys_vid:
                #vidid=1
                if self.opts.local_rank==0:
                    loss_dict = self.phys_env(frame_start=torch.zeros(1).to(self.device)+\
                                                 self.phys_env.data_offset[vidid])
                    data = self.phys_env.query()
                    self.env_vis.show('%02d-%05d'%(vidid,self.total_steps), data)
                #break
            #loss_dict = self.phys_env(frame_start=torch.zeros(1).to(self.device)+\
            #                                     self.phys_env.data_offset[4]+100)
            #loss_dict = self.phys_env(frame_start=torch.zeros(1).to(self.device)+\
            #                                     self.phys_env.data_offset[10]+60)
            self.phys_env.reinit_envs(800//opts.phys_wdw_len, 
                                    wdw_length=opts.phys_wdw_len,is_eval=False)
            torch.cuda.empty_cache()
           
            #with torch.no_grad():
            #    mesh = extract_mesh(self, self.opts.chunk, \
            #                                 256, self.opts.mc_threshold/2,
            #                                 vidid=4, is_eval=True)['mesh']
            #    embedid = torch.Tensor([self.phys_env.data_offset[4]+120]).cuda()
            #    delta_ja_ref = self.phys_env.delta_joint_ref_mlp(embedid.reshape(-1,1))
            #    self.nerf_body_rts.rest_angles += delta_ja_ref.cpu()
            #    verts,_ = warp_fw(self.opts, self, {}, mesh.vertices, embedid.long())
            #    trimesh.Trimesh(verts, mesh.faces).export('tmp/0.obj')
        loss_dict = self.phys_env()

        return loss_dict['total_loss'],loss_dict

    def forward_warmup_rootmlp(self, batch):
        """
        batch variable is not never being used here
        """
        # render ground-truth data
        opts = self.opts
        device = self.device
        
        # loss
        aux_out={}
        self.rtk = torch.zeros(self.num_fr,4,4).to(device)
        self.frameid = torch.Tensor(range(self.num_fr)).to(device)
        self.dataid,_ = fid_reindex(self.frameid, self.num_vid, self.data_offset)
        self.convert_root_pose()

        rtk_gt = torch.Tensor(self.latest_vars['rtk']).to(device)
        _ = rtk_loss(self.rtk, rtk_gt, aux_out)
        root_sm_loss =  compute_root_sm_2nd_loss(self.rtk, self.data_offset)
        
        total_loss = 0.1*aux_out['rot_loss'] + 0.01*root_sm_loss
        aux_out['warmup_root_sm_loss'] = root_sm_loss
        del aux_out['trn_loss']

        return total_loss, aux_out
    
    def forward_warmup_shape(self, batch):
        """
        batch variable is not never being used here
        """
        # render ground-truth data
        opts = self.opts
        
        # loss
        shape_factor = 0.1 / 3
        aux_out={}
        total_loss = shape_init_loss(self.dp_verts_unit*shape_factor,self.dp_faces, \
                              self.nerf_coarse, self.embedding_xyz, 
            bound_factor=opts.bound_factor * 1.2, use_ellips=opts.init_ellips)
        aux_out['shape_init_loss'] = total_loss

        return total_loss, aux_out
    
    def forward_warmup_skin(self, batch):
        """
        batch variable is not never being used here
        """
        # render ground-truth data
        opts = self.opts
        
        # Sample points
        nsample =10000
        shape_factor = 0.1 / 3
        pts = self.dp_verts_unit*shape_factor
        obj_bound = pts.abs().max(0)[0][None,None]
        bound = obj_bound * opts.bound_factor
        pts = torch.rand(1,nsample,3).to(self.device)*2*bound-bound

        # loss
        bones = self.bones.detach()
        pose_code = self.rest_pose_code(torch.Tensor([0]).long().to(self.device))[None]
        skin_pred = center_mlp_skinning(pts, self.embedding_xyz, bones, 
                    pose_code,  self.nerf_skin)
        skin_gt = gauss_skinning(bones, pts) # bs, N, B
        total_loss = (skin_gt - skin_pred).pow(2).mean()

        aux_out={}
        aux_out['skin_init_loss'] = total_loss

        return total_loss, aux_out

    def forward_warmup(self, batch):
        """
        batch variable is not never being used here
        """
        # render ground-truth data
        opts = self.opts
        bs_rd = 16
        with torch.no_grad():
            vertex_color = self.dp_embed
            dp_feats_rd, rtk_raw = self.render_dp(self.dp_verts_unit, 
                    self.dp_faces, vertex_color, self.near_far, self.device, 
                    self.mesh_renderer, bs_rd)

        aux_out={}
        # predict delta se3
        root_rts = self.nerf_root_rts(dp_feats_rd)
        root_rmat = root_rts[:,0,:9].view(-1,3,3)
        root_tmat = root_rts[:,0,9:12]    

        # construct base se3
        rtk = torch.zeros(bs_rd, 4,4).to(self.device)
        rtk[:,:3] = create_base_se3(bs_rd, self.device)

        # compose se3
        rmat = rtk[:,:3,:3]
        tmat = rtk[:,:3,3]
        tmat = tmat + rmat.matmul(root_tmat[...,None])[...,0]
        rmat = rmat.matmul(root_rmat)
        rtk[:,:3,:3] = rmat
        rtk[:,:3,3] = tmat.detach() # do not train translation
        
        # loss
        total_loss = rtk_loss(rtk, rtk_raw, aux_out)

        aux_out['total_loss'] = total_loss

        return total_loss, aux_out
    
    def nerf_render(self, rtk, kaug, embedid, nsample=256, ndepth=128):
        opts=self.opts
        # render rays
        if opts.debug:
            torch.cuda.synchronize()
            start_time = time.time()

        # 2bs,...
        Rmat, Tmat, Kinv = self.prepare_ray_cams(rtk, kaug)
        bs = Kinv.shape[0]
        
        # for batch:2bs,            nsample+x
        # for line: 2bs*(nsample+x),1
        rand_inds, rays, frameid, errid = self.sample_pxs(bs, nsample, Rmat, Tmat, Kinv,
        self.dataid, self.frameid, self.frameid_sub, self.embedid,self.lineid,self.errid,
        self.imgs, self.masks, self.bgmasks, self.vis2d, self.flow, self.occ, self.dp_feats)
        self.frameid = frameid # only used in loss filter
        self.errid = errid


        if opts.debug:
            torch.cuda.synchronize()
            print('prepare rays time: %.2f'%(time.time()-start_time))

        bs_rays = rays['bs'] * rays['nsample'] # over pixels
        results=defaultdict(list)
        for i in range(0, bs_rays, opts.chunk):
            rays_chunk = chunk_rays(rays,i,opts.chunk)
            # decide whether to use fine samples 
            if self.progress > opts.fine_steps:
                self.use_fine = True
            else:
                self.use_fine = False
            rendered_chunks = render_rays(self.nerf_models,
                        self.embeddings,
                        rays_chunk,
                        N_samples = ndepth,
                        use_disp=False,
                        perturb=opts.perturb,
                        noise_std=opts.noise_std,
                        chunk=opts.chunk, # chunk size is effective in val mode
                        obj_bound=self.latest_vars['obj_bound'],
                        use_fine=self.use_fine,
                        img_size=self.img_size,
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
                if self.training:
                    v = v.view(rays['bs'],rays['nsample'],-1)
                else:
                    v = v.view(bs,self.img_size, self.img_size, -1)
            results[k] = v
        if opts.debug:
            torch.cuda.synchronize()
            print('rendering time: %.2f'%(time.time()-start_time))
        
        # viser feature matching
        if opts.use_embed:
            results['pts_pred'] = (results['pts_pred'] - torch.Tensor(self.vis_min[None]).\
                    to(self.device)) / torch.Tensor(self.vis_len[None]).to(self.device)
            results['pts_exp']  = (results['pts_exp'] - torch.Tensor(self.vis_min[None]).\
                    to(self.device)) / torch.Tensor(self.vis_len[None]).to(self.device)
            results['pts_pred'] = results['pts_pred'].clamp(0,1)
            results['pts_exp']  = results['pts_exp'].clamp(0,1)

        if opts.debug:
            torch.cuda.synchronize()
            print('compute flow time: %.2f'%(time.time()-start_time))
        return results, rand_inds

    
    @staticmethod
    def render_dp(dp_verts_unit, dp_faces, dp_embed, near_far, device, 
                  mesh_renderer, bs):
        """
        render a pair of (densepose feature bsx16x112x112, se3)
        input is densepose surface model and near-far plane
        """
        verts = dp_verts_unit
        faces = dp_faces
        dp_embed = dp_embed
        num_verts, embed_dim = dp_embed.shape
        img_size = 256
        crop_size = 112
        focal = 2
        std_rot = 6.28 # rotation std
        std_dep = 0.5 # depth std


        # scale geometry and translation based on near-far plane
        d_mean = near_far.mean()
        verts = verts / 3 * d_mean # scale based on mean depth
        dep_rand = 1 + np.random.normal(0,std_dep,bs)
        dep_rand = torch.Tensor(dep_rand).to(device)
        d_obj = d_mean * dep_rand
        d_obj = torch.max(d_obj, 1.2*1/3 * d_mean)
        
        # set cameras
        rot_rand = np.random.normal(0,std_rot,(bs,3))
        rot_rand = torch.Tensor(rot_rand).to(device)
        Rmat = transforms.axis_angle_to_matrix(rot_rand)
        Tmat = torch.cat([torch.zeros(bs, 2).to(device), d_obj[:,None]],-1)
        K =    torch.Tensor([[focal,focal,0,0]]).to(device).repeat(bs,1)
        
        # add RTK: [R_3x3|T_3x1]
        #          [fx,fy,px,py], to the ndc space
        Kimg = torch.Tensor([[focal*img_size/2.,focal*img_size/2.,img_size/2.,
                            img_size/2.]]).to(device).repeat(bs,1)
        rtk = torch.zeros(bs,4,4).to(device)
        rtk[:,:3,:3] = Rmat
        rtk[:,:3, 3] = Tmat
        rtk[:,3, :]  = Kimg

        # repeat mesh
        verts = verts[None].repeat(bs,1,1)
        faces = faces[None].repeat(bs,1,1)
        dp_embed = dp_embed[None].repeat(bs,1,1)

        # obj-cam transform 
        verts = obj_to_cam(verts, Rmat, Tmat)
        
        # pespective projection
        verts = pinhole_cam(verts, K)
        
        # render sil+rgb
        rendered = []
        for i in range(0,embed_dim,3):
            dp_chunk = dp_embed[...,i:i+3]
            dp_chunk_size = dp_chunk.shape[-1]
            if dp_chunk_size<3:
                dp_chunk = torch.cat([dp_chunk,
                    dp_embed[...,:(3-dp_chunk_size)]],-1)
            rendered_chunk = render_color(mesh_renderer, verts, faces, 
                    dp_chunk,  texture_type='vertex')
            rendered_chunk = rendered_chunk[:,:3]
            rendered.append(rendered_chunk)
        rendered = torch.cat(rendered, 1)
        rendered = rendered[:,:embed_dim]

        # resize to bounding box
        rendered_crops = []
        for i in range(bs):
            mask = rendered[i].max(0)[0]>0
            mask = mask.cpu().numpy()
            indices = np.where(mask>0); xid = indices[1]; yid = indices[0]
            center = ( (xid.max()+xid.min())//2, (yid.max()+yid.min())//2)
            length = ( int((xid.max()-xid.min())*1.//2 ), 
                      int((yid.max()-yid.min())*1.//2  ))
            left,top,w,h = [center[0]-length[0], center[1]-length[1],
                    length[0]*2, length[1]*2]
            rendered_crop = torchvision.transforms.functional.resized_crop(\
                    rendered[i], top,left,h,w,(50,50))
            # mask augmentation
            rendered_crop = mask_aug(rendered_crop)

            rendered_crops.append( rendered_crop)
            #cv2.imwrite('%d.png'%i, rendered_crop.std(0).cpu().numpy()*1000)

        rendered_crops = torch.stack(rendered_crops,0)
        rendered_crops = F.interpolate(rendered_crops, (crop_size, crop_size), 
                mode='bilinear')
        rendered_crops = F.normalize(rendered_crops, 2,1)
        return rendered_crops, rtk

    @staticmethod
    def prepare_ray_cams(rtk, kaug):
        """ 
        in: rtk, kaug
        out: Rmat, Tmat, Kinv
        """
        Rmat = rtk[:,:3,:3]
        Tmat = rtk[:,:3,3]
        Kmat = K2mat(rtk[:,3,:])
        Kaug = K2inv(kaug) # p = Kaug Kmat P
        Kinv = Kmatinv(Kaug.matmul(Kmat))
        return Rmat, Tmat, Kinv

    def sample_pxs(self, bs, nsample, Rmat, Tmat, Kinv,
                   dataid, frameid, frameid_sub, embedid, lineid,errid,
                   imgs, masks, bgmasks, vis2d, flow, occ, dp_feats):
        """
        make sure self. is not modified
        xys:    bs, nsample, 2
        rand_inds: bs, nsample
        """
        opts = self.opts
        Kinv_in=Kinv.clone()
        dataid_in=dataid.clone()
        frameid_sub_in = frameid_sub.clone()
        
        # sample 1x points, sample 4x points for further selection
        nsample_a = 4*nsample
        rand_inds, xys = sample_xy(self.img_size, bs, nsample+nsample_a, self.device,
                               return_all= not(self.training), lineid=lineid)

        if self.training and opts.use_unc and \
                self.progress >= (opts.warmup_steps):
            is_active=True
            nsample_s = int(opts.nactive * nsample)  # active 
            nsample   = int(nsample*(1-opts.nactive)) # uniform
        else:
            is_active=False

        if self.training:
            rand_inds_a, xys_a = rand_inds[:,-nsample_a:].clone(), xys[:,-nsample_a:].clone()
            rand_inds, xys     = rand_inds[:,:nsample].clone(), xys[:,:nsample].clone()

            if opts.lineload:
                # expand frameid, Rmat,Tmat, Kinv
                frameid_a=        frameid[:,None].repeat(1,nsample_a)
                frameid_sub_a=frameid_sub[:,None].repeat(1,nsample_a)
                dataid_a=          dataid[:,None].repeat(1,nsample_a)
                errid_a=            errid[:,None].repeat(1,nsample_a)
                Rmat_a = Rmat[:,None].repeat(1,nsample_a,1,1)
                Tmat_a = Tmat[:,None].repeat(1,nsample_a,1)
                Kinv_a = Kinv[:,None].repeat(1,nsample_a,1,1)
                # expand         
                frameid =         frameid[:,None].repeat(1,nsample)
                frameid_sub = frameid_sub[:,None].repeat(1,nsample)
                dataid =           dataid[:,None].repeat(1,nsample)
                errid =             errid[:,None].repeat(1,nsample)
                Rmat = Rmat[:,None].repeat(1,nsample,1,1)
                Tmat = Tmat[:,None].repeat(1,nsample,1)
                Kinv = Kinv[:,None].repeat(1,nsample,1,1)

                batch_map   = torch.Tensor(range(bs)).to(self.device)[:,None].long()
                batch_map_a = batch_map.repeat(1,nsample_a)
                batch_map   = batch_map.repeat(1,nsample)

        # importance sampling
        if is_active:
            with torch.no_grad():
                # run uncertainty estimation
                ts = frameid_sub_in.to(self.device) / self.max_ts * 2 -1
                ts = ts[:,None,None].repeat(1,nsample_a,1)
                dataid_in = dataid_in.long().to(self.device)
                vid_code = self.vid_code(dataid_in)[:,None].repeat(1,nsample_a,1)

                # convert to normalized coords
                xysn = torch.cat([xys_a, torch.ones_like(xys_a[...,:1])],2)
                xysn = xysn.matmul(Kinv_in.permute(0,2,1))[...,:2]

                xyt = torch.cat([xysn, ts],-1)
                xyt_embedded = self.embedding_xyz(xyt)
                xyt_code = torch.cat([xyt_embedded, vid_code],-1)
                unc_pred = self.nerf_unc(xyt_code)[...,0]

            # preprocess to format 2,bs,w
            if opts.lineload:
                unc_pred = unc_pred.view(2,-1)
                xys =     xys.view(2,-1,2)
                xys_a = xys_a.view(2,-1,2)
                rand_inds =     rand_inds.view(2,-1)
                rand_inds_a = rand_inds_a.view(2,-1)
                frameid   =   frameid.view(2,-1)
                frameid_a = frameid_a.view(2,-1)
                frameid_sub   =   frameid_sub.view(2,-1)
                frameid_sub_a = frameid_sub_a.view(2,-1)
                dataid   =   dataid.view(2,-1)
                dataid_a = dataid_a.view(2,-1)
                errid   =     errid.view(2,-1)
                errid_a   = errid_a.view(2,-1)
                batch_map   =   batch_map.view(2,-1)
                batch_map_a = batch_map_a.view(2,-1)
                Rmat   =   Rmat.view(2,-1,3,3)
                Rmat_a = Rmat_a.view(2,-1,3,3)
                Tmat   =   Tmat.view(2,-1,3)
                Tmat_a = Tmat_a.view(2,-1,3)
                Kinv   =   Kinv.view(2,-1,3,3)
                Kinv_a = Kinv_a.view(2,-1,3,3)

                nsample_s = nsample_s * bs//2
                bs=2

                # merge top nsamples
                topk_samp = unc_pred.topk(nsample_s,dim=-1)[1] # bs,nsamp
                
                # use the first imgs (in a pair) sampled index
                xys_a =       torch.stack(          [xys_a[i][topk_samp[0]] for i in range(bs)],0)
                rand_inds_a = torch.stack(    [rand_inds_a[i][topk_samp[0]] for i in range(bs)],0)
                frameid_a =       torch.stack(  [frameid_a[i][topk_samp[0]] for i in range(bs)],0)
                frameid_sub_a=torch.stack(  [frameid_sub_a[i][topk_samp[0]] for i in range(bs)],0)
                dataid_a =         torch.stack(  [dataid_a[i][topk_samp[0]] for i in range(bs)],0)
                errid_a =           torch.stack(  [errid_a[i][topk_samp[0]] for i in range(bs)],0)
                batch_map_a =   torch.stack(  [batch_map_a[i][topk_samp[0]] for i in range(bs)],0)
                Rmat_a =             torch.stack(  [Rmat_a[i][topk_samp[0]] for i in range(bs)],0)
                Tmat_a =             torch.stack(  [Tmat_a[i][topk_samp[0]] for i in range(bs)],0)
                Kinv_a =             torch.stack(  [Kinv_a[i][topk_samp[0]] for i in range(bs)],0)

                xys =       torch.cat([xys,xys_a],1)
                rand_inds = torch.cat([rand_inds,rand_inds_a],1)
                frameid =   torch.cat([frameid,frameid_a],1)
                frameid_sub=torch.cat([frameid_sub,frameid_sub_a],1)
                dataid =    torch.cat([dataid,dataid_a],1)
                errid =     torch.cat([errid,errid_a],1)
                batch_map = torch.cat([batch_map,batch_map_a],1)
                Rmat =      torch.cat([Rmat,Rmat_a],1)
                Tmat =      torch.cat([Tmat,Tmat_a],1)
                Kinv =      torch.cat([Kinv,Kinv_a],1)
            else:
                topk_samp = unc_pred.topk(nsample_s,dim=-1)[1] # bs,nsamp
                
                xys_a =       torch.stack(      [xys_a[i][topk_samp[i]] for i in range(bs)],0)
                rand_inds_a = torch.stack([rand_inds_a[i][topk_samp[i]] for i in range(bs)],0)
                
                xys =       torch.cat([xys,xys_a],1)
                rand_inds = torch.cat([rand_inds,rand_inds_a],1)
        

        # for line: reshape to 2*bs, 1,...
        if self.training and opts.lineload:
            frameid =         frameid.view(-1)
            frameid_sub = frameid_sub.view(-1)
            dataid =           dataid.view(-1)
            errid =             errid.view(-1)
            batch_map =     batch_map.view(-1)
            xys =           xys.view(-1,1,2)
            rand_inds = rand_inds.view(-1,1)
            Rmat = Rmat.view(-1,3,3)
            Tmat = Tmat.view(-1,3)
            Kinv = Kinv.view(-1,3,3)

        near_far = self.near_far[frameid.long()]
        rays = raycast(xys, Rmat, Tmat, Kinv, near_far)
       
        # need to reshape dataid, frameid_sub, embedid #TODO embedid equiv to frameid
        self.update_rays(rays, bs>1, dataid, frameid_sub, frameid, xys, Kinv)
        
        if 'bones' in self.nerf_models.keys():
            # update delta rts fw
            self.update_delta_rts(rays)
        
        # compute background 
        if opts.bgmlp != "":
            rays['trsi_bg'],_,_ = self.trsi_2d(rays['xy_uncrop'], 
                                frameid.long().to(self.device))

        # for line: 2bs*nsamp,1
        # for batch:2bs,nsamp
        #TODO reshape imgs, masks, etc.
        if self.training and opts.lineload:
            self.obs_to_rays_line(rays, rand_inds, imgs, masks, bgmasks, vis2d, flow, occ, 
                    dp_feats, batch_map)
        else:
            self.obs_to_rays(rays, rand_inds, imgs, masks, bgmasks, vis2d, flow, occ, dp_feats)

        # TODO visualize samples
        #pdb.set_trace()
        #self.imgs_samp = []
        #for i in range(bs):
        #    self.imgs_samp.append(draw_pts(self.imgs[i], xys_a[i]))
        #self.imgs_samp = torch.stack(self.imgs_samp,0)

        return rand_inds, rays, frameid, errid
    
    def obs_to_rays_line(self, rays, rand_inds, imgs, masks, bgmasks, vis2d,
            flow, occ, dp_feats,batch_map):
        """
        convert imgs, masks, flow, occ, dp_feats to rays
        rand_map: map pixel index to original batch index
        rand_inds: bs, 
        """
        opts = self.opts
        rays['img_at_samp']=torch.gather(imgs[batch_map][...,0], 2, 
                rand_inds[:,None].repeat(1,3,1))[:,None][...,0]
        rays['sil_at_samp']=torch.gather(masks[batch_map][...,0], 2, 
                rand_inds[:,None].repeat(1,1,1))[:,None][...,0]
        rays['bgsil_at_samp']=torch.gather(bgmasks[batch_map][...,0], 2, 
                rand_inds[:,None].repeat(1,1,1))[:,None][...,0]
        rays['vis_at_samp']=torch.gather(vis2d[batch_map][...,0], 2, 
                rand_inds[:,None].repeat(1,1,1))[:,None][...,0]
        rays['flo_at_samp']=torch.gather(flow[batch_map][...,0], 2, 
                rand_inds[:,None].repeat(1,2,1))[:,None][...,0]
        rays['cfd_at_samp']=torch.gather(occ[batch_map][...,0], 2, 
                rand_inds[:,None].repeat(1,1,1))[:,None][...,0]
        if opts.use_embed:
            rays['feats_at_samp']=torch.gather(dp_feats[batch_map][...,0], 2, 
                rand_inds[:,None].repeat(1,16,1))[:,None][...,0]
     
    def obs_to_rays(self, rays, rand_inds, imgs, masks, bgmasks, vis2d,
            flow, occ, dp_feats):
        """
        convert imgs, masks, flow, occ, dp_feats to rays
        """
        opts = self.opts
        bs = imgs.shape[0]
        rays['img_at_samp'] = torch.stack([imgs[i].view(3,-1).T[rand_inds[i]]\
                                for i in range(bs)],0) # bs,ns,3
        rays['sil_at_samp'] = torch.stack([masks[i].view(-1,1)[rand_inds[i]]\
                                for i in range(bs)],0) # bs,ns,1
        rays['bgsil_at_samp'] = torch.stack([bgmasks[i].view(-1,1)[rand_inds[i]]\
                                for i in range(bs)],0) # bs,ns,1
        rays['vis_at_samp'] = torch.stack([vis2d[i].view(-1,1)[rand_inds[i]]\
                                for i in range(bs)],0) # bs,ns,1
        rays['flo_at_samp'] = torch.stack([flow[i].view(2,-1).T[rand_inds[i]]\
                                for i in range(bs)],0) # bs,ns,2
        rays['cfd_at_samp'] = torch.stack([occ[i].view(-1,1)[rand_inds[i]]\
                                for i in range(bs)],0) # bs,ns,1
        if opts.use_embed:
            feats_at_samp = [dp_feats[i].view(16,-1).T\
                             [rand_inds[i].long()] for i in range(bs)]
            feats_at_samp = torch.stack(feats_at_samp,0) # bs,ns,num_feat
            rays['feats_at_samp'] = feats_at_samp
        
    def update_delta_rts(self, rays):
        """
        change bone_rts_fw to delta fw
        """
        opts = self.opts
        bones_rst, bone_rts_rst = zero_to_rest_bone(self, self.nerf_models['bones'])
                                                      #vid=rays['vidid'][:,0,0]) # instance-specific rest shape
        self.nerf_models['bones_rst']=bones_rst

        # delta rts
        rays['bone_rts'] = zero_to_rest_dpose(opts, rays['bone_rts'], bone_rts_rst)

        if 'bone_rts_target' in rays.keys():       
            rays['bone_rts_target'] = rays['bone_rts'].view(2,-1).flip(0).\
                                                reshape(rays['bone_rts'].shape)

    def update_rays(self, rays, is_pair, dataid, frameid_sub, embedid, xys, Kinv):
        """
        """
        opts = self.opts
        embedid = embedid.long().to(self.device)

        # pass time-dependent inputs
        time_embedded = self.pose_code(embedid)[:,None]
        vidid,_ = fid_reindex(embedid, self.num_vid, self.data_offset)
        rays['vidid'] = vidid[:,None,None].repeat(1,rays['nsample'],1)
        rays['time_embedded'] = time_embedded.repeat(1,rays['nsample'],1)
        if opts.mlp_deform:
            dfm_code = self.dfm_code(embedid)[:,None]
            rays['dfm_code'] = dfm_code.repeat(1,rays['nsample'],1)
        if opts.lbs:
            bone_rts = self.nerf_body_rts(embedid)
            rays['bone_rts'] = bone_rts.repeat(1,rays['nsample'],1)

        # append target frame rtk
        if is_pair:
            rtk_vec = rays['rtk_vec'] # bs, N, 21
            rtk_vec_target = rtk_vec.view(2,-1).flip(0)
            rays['rtk_vec_target'] = rtk_vec_target.reshape(rays['rtk_vec'].shape)
            
            if opts.lbs:
                rays['bone_rts_target'] = rays['bone_rts'].view(2,-1).flip(0).\
                                                    reshape(rays['bone_rts'].shape)

        if opts.env_code:
            if hasattr(self.env_code, "app_code_pred"):
                rays['env_code'] = self.env_code.app_code_pred[:,None]
            else:
                rays['env_code'] = self.env_code(embedid)[:,None]
            rays['env_code'] = rays['env_code'].repeat(1,rays['nsample'],1)

        if opts.use_unc:
            ts = frameid_sub.to(self.device) / self.max_ts * 2 -1
            ts = ts[:,None,None].repeat(1,rays['nsample'],1)
            rays['ts'] = ts
        
            dataid = dataid.long().to(self.device)
            vid_code = self.vid_code(dataid)[:,None].repeat(1,rays['nsample'],1)
            rays['vid_code'] = vid_code
            
            xysn = torch.cat([xys, torch.ones_like(xys[...,:1])],2)
            xysn = xysn.matmul(Kinv.permute(0,2,1))[...,:2]
            rays['xysn'] = xysn

    def convert_line_input(self, batch):
        device = self.device
        opts = self.opts
        # convert to float
        for k,v in batch.items():
            batch[k] = batch[k].float()

        bs=batch['dataid'].shape[0]

        self.imgs         = batch['img']         .view(bs,2,3, -1).permute(1,0,2,3).reshape(bs*2,3, -1,1).to(device)
        self.masks        = batch['mask']        .view(bs,2,1, -1).permute(1,0,2,3).reshape(bs*2,1, -1,1).to(device)
        self.vis2d        = batch['vis2d']       .view(bs,2,1, -1).permute(1,0,2,3).reshape(bs*2,1, -1,1).to(device)
        self.flow         = batch['flow']        .view(bs,2,2, -1).permute(1,0,2,3).reshape(bs*2,2, -1,1).to(device)
        self.occ          = batch['occ']         .view(bs,2,1, -1).permute(1,0,2,3).reshape(bs*2,1, -1,1).to(device)
        self.dps          = batch['dp']          .view(bs,2,1, -1).permute(1,0,2,3).reshape(bs*2,1, -1,1).to(device)
        self.dp_feats     = batch['dp_feat_rsmp'].view(bs,2,16,-1).permute(1,0,2,3).reshape(bs*2,16,-1,1).to(device)
        #TODO process with linear projector
        if opts.dp_proj: self.dp_feats = self.dp_proj(self.dp_feats)
        self.dp_feats     = F.normalize(self.dp_feats, 2,1)
        self.rtk          = batch['rtk']         .view(bs,-1,4,4).permute(1,0,2,3).reshape(-1,4,4)    .to(device)
        self.kaug         = batch['kaug']        .view(bs,-1,4).permute(1,0,2).reshape(-1,4)          .to(device)
        self.frameid      = batch['frameid']     .view(bs,-1).permute(1,0).reshape(-1).cpu()
        self.dataid       = batch['dataid']      .view(bs,-1).permute(1,0).reshape(-1).cpu()
        self.lineid       = batch['lineid']      .view(bs,-1).permute(1,0).reshape(-1).to(device)
      
        self.frameid_sub = self.frameid.clone() # id within a video
        self.embedid = self.frameid + self.data_offset[self.dataid.long()]
        self.frameid = self.frameid + self.data_offset[self.dataid.long()]
        self.errid = self.frameid*opts.img_size + self.lineid.cpu() # for err filter
        self.rt_raw  = self.rtk.clone()[:,:3]

        # process silhouette
        self.bgmasks = ((1-self.masks)*self.vis2d)>0
        self.bgmasks =self.bgmasks.float()

        self.masks = (self.masks*self.vis2d)>0
        self.masks = self.masks.float()

    def convert_batch_input(self, batch):
        device = self.device
        opts = self.opts
        if batch['img'].dim()==4:
            bs,_,h,w = batch['img'].shape
        else:
            bs,_,_,h,w = batch['img'].shape
        # convert to float
        for k,v in batch.items():
            batch[k] = batch[k].float()

        img_tensor = batch['img'].view(bs,-1,3,h,w).permute(1,0,2,3,4).reshape(-1,3,h,w)
        self.imgs         = img_tensor.to(device)
        self.masks        = batch['mask']        .view(bs,-1,h,w).permute(1,0,2,3).reshape(-1,h,w)      .to(device)
        self.vis2d        = batch['vis2d']        .view(bs,-1,h,w).permute(1,0,2,3).reshape(-1,h,w)     .to(device)
        self.dps          = batch['dp']          .view(bs,-1,h,w).permute(1,0,2,3).reshape(-1,h,w)      .to(device)
        dpfd = 16
        dpfs = 112
        self.dp_feats     = batch['dp_feat']     .view(bs,-1,dpfd,dpfs,dpfs).permute(1,0,2,3,4).reshape(-1,dpfd,dpfs,dpfs).to(device)
        self.dp_bbox      = batch['dp_bbox']     .view(bs,-1,4).permute(1,0,2).reshape(-1,4)          .to(device)
        # augment images
        if opts.train_cnn:
            color_aug = torchvision.transforms.ColorJitter(brightness=0.2,
                                        contrast=0.2, saturation=0.2, hue=0.5)
            self.imgs = color_aug(self.imgs)
            for k in range(bs):
                # rand mask
                if np.random.rand()>0.5:
                    rct = ( np.random.rand(2) *h).astype(int)
                    rsz = ((np.random.rand(2)*0.2+0.1)*h).astype(int)
                    lb = np.minimum(np.maximum(rct-rsz, [0,0]), [h, h])
                    ub = np.minimum(np.maximum(rct+rsz, [0,0]), [h, h])
                    self.imgs[k,:,lb[0]:ub[0], lb[1]:ub[1]] = \
                        torch.Tensor(np.random.rand(3)).to(device)[:,None,None]
            self.imgs = self.imgs * self.masks[:,None]

        if opts.use_embed and opts.ft_cse and (not self.is_warmup_pose):
            self.dp_feats_mask = self.dp_feats.abs().sum(1)>0
            self.csepre_feats = self.dp_feats.clone()
            # unnormalized features
            self.csenet_feats, self.dps = self.csenet(self.imgs, self.masks)
            # for visualization
            self.dps = self.dps * self.dp_feats_mask.float()
            if self.progress > opts.ftcse_steps:
                self.dp_feats = self.csenet_feats
            else:
                self.dp_feats = self.csenet_feats.detach()
        #TODO process with linear projector
        if opts.dp_proj: self.dp_feats = self.dp_proj(self.dp_feats)
        self.dp_feats     = F.normalize(self.dp_feats, 2,1)
        self.rtk          = batch['rtk']         .view(bs,-1,4,4).permute(1,0,2,3).reshape(-1,4,4)    .to(device)
        self.kaug         = batch['kaug']        .view(bs,-1,4).permute(1,0,2).reshape(-1,4)          .to(device)
        self.frameid      = batch['frameid']     .view(bs,-1).permute(1,0).reshape(-1).cpu()
        self.dataid       = batch['dataid']      .view(bs,-1).permute(1,0).reshape(-1).cpu()
      
        self.frameid_sub = self.frameid.clone() # id within a video
        self.embedid = self.frameid + self.data_offset[self.dataid.long()]
        self.frameid = self.frameid + self.data_offset[self.dataid.long()]
        self.errid = self.frameid # for err filter
        self.rt_raw  = self.rtk.clone()[:,:3]

        # process silhouette
        self.bgmasks = ((1-self.masks)*self.vis2d)>0
        self.bgmasks =self.bgmasks.float()

        self.masks = (self.masks*self.vis2d)>0
        self.masks = self.masks.float()
        
        self.flow = batch['flow'].view(bs,-1,2,h,w).permute(1,0,2,3,4).reshape(-1,2,h,w).to(device)
        self.occ  = batch['occ'].view(bs,-1,h,w).permute(1,0,2,3).reshape(-1,h,w)     .to(device)
        self.lineid = None

    def convert_root_pose(self):
        """
        query embedding and predict root pose
        assumes has self.
        {rtk, frameid, dp_feats, dps, masks, kaug }
        produces self.
        {rtk}
        """
        opts = self.opts
        bs = self.rtk.shape[0]
        device = self.device

        # scale initial poses
        if self.use_cam:
            self.rtk[:,:3,3] = self.rtk[:,:3,3] / self.obj_scale
        else:
            self.rtk[:,:3] = create_base_se3(bs, device)

        # compute delta pose (excluding initialization time)
        if self.opts.root_opt:
            if self.root_basis == 'cnn':
                frame_code = self.dp_feats
            elif self.root_basis == 'mlp' or self.root_basis == 'exp'\
              or self.root_basis == 'expmlp':
                frame_code = self.frameid.long().to(device)
            else: print('error'); exit()
            root_rts = self.nerf_root_rts(frame_code)
            if opts.use_cnn: 
                # replace rotations before rendering
                if opts.cnn_code:
                    rts_pred, app_code_pred, shape_code_pred = \
                            self.cnn_ff.forward_with_code(self.imgs)
                    root_rts[...,:9] = rts_pred[...,:9]
                    # replace at test but this does not save gradients
                    vidid,_ = fid_reindex(self.frameid, self.num_vid, self.data_offset)
                    vidid = vidid.long()
                    frameid = self.frameid.long()
                    self.env_code.weight.data[frameid] = app_code_pred
                    #self.nerf_coarse.vid_code.weight.data[vidid] = shape_code_pred
                    # save grads
                    self.env_code.app_code_pred = app_code_pred
                    #self.nerf_coarse.vid_code.shape_code_pred = \
                    #        self.nerf_coarse.vid_code.weight.data.clone()
                    #self.nerf_coarse.vid_code.shape_code_pred[vidid] = shape_code_pred
                else:
                    root_rts[...,:9] = self.cnn_ff(self.imgs)[...,:9]
            self.rtk = refine_rt(self.rtk, root_rts)

        self.rtk[:,3,:] = self.ks_param[self.dataid.long()] #TODO kmat

    def compute_rts(self):
        """
        Assumpions
        - use_cam
        - use mlp or exp root pose 
        input:  rt_raw representing the initial root poses
        output: current estimate of rtks for all frames
        """
        device = self.device
        opts = self.opts
        frameid = torch.Tensor(range(self.num_fr)).to(device).long()

        if self.use_cam:
            # scale initial poses
            rt_raw = torch.Tensor(self.latest_vars['rt_raw']).to(device)
            rt_raw[:,:3,3] = rt_raw[:,:3,3] / self.obj_scale
        else:
            rt_raw = create_base_se3(self.num_fr, device)
        
        # compute mlp rts
        if opts.root_opt:
            if self.root_basis == 'mlp' or self.root_basis == 'exp'\
            or self.root_basis == 'expmlp':
                root_rts = self.nerf_root_rts(frameid)
            else: print('error'); exit()
            rt_raw = refine_rt(rt_raw, root_rts)
        return rt_raw

    @staticmethod
    def save_latest_vars(latest_vars, rtk, kaug, frameid, rt_raw):
        """
        in: 
        {rtk, kaug, rt_raw, frameid}
        out:
        {latest_vars}
        these are only used in get_near_far_plane and compute_visibility
        """
        if not torch.is_tensor(rtk):
            rtk = torch.Tensor(rtk)
            kaug = torch.Tensor(kaug)
            frameid = torch.Tensor(frameid)
            rt_raw = torch.Tensor(rt_raw)
        rtk = rtk.clone().detach()
        Kmat = K2mat(rtk[:,3])
        Kaug = K2inv(kaug) # p = Kaug Kmat P
        rtk[:,3] = mat2K(Kaug.matmul(Kmat))

        # TODO don't want to save k at eval time (due to different intrinsics)
        latest_vars['rtk'][frameid.long()] = rtk.cpu().numpy()
        latest_vars['rt_raw'][frameid.long()] = rt_raw.cpu().numpy()
        latest_vars['idk'][frameid.long()] = 1

    def set_input(self, batch, load_line=False):
        device = self.device
        opts = self.opts

        if load_line:
            self.convert_line_input(batch)
        else:
            self.convert_batch_input(batch)
        bs = self.imgs.shape[0]
        
        if opts.lineload and self.training:
            self.dp_feats = self.dp_feats
        else:
            self.dp_feats,_ = resample_dp(self.dp_feats, 
                    self.dp_bbox, self.kaug, self.img_size)
        
        self.convert_root_pose()
      
        self.save_latest_vars(self.latest_vars, self.rtk, 
                                          self.kaug, self.frameid, self.rt_raw)
        
        if self.training and self.opts.anneal_freq:
            alpha = self.num_freqs * \
                self.progress / (opts.warmup_steps)
            #if alpha>self.alpha.data[0]:
            self.alpha.data[0] = min(max(6, alpha),self.num_freqs) # alpha from 6 to 10
            self.embedding_xyz.alpha = self.alpha.data[0]
            self.embedding_dir.alpha = self.alpha.data[0]

        return bs

