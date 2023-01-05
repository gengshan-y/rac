# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import os
import os.path as osp
import sys
sys.path.insert(0,'third_party')
import time
import pdb
import numpy as np
from absl import flags
import cv2
import time

from nnutils import banmo
import subprocess
from torch.utils.tensorboard import SummaryWriter
from kmeans_pytorch import kmeans
import torch.distributed as dist
import torch.nn.functional as F
import trimesh
import torchvision
from torch.autograd import Variable
from collections import defaultdict
from pytorch3d import transforms
from torch.nn.utils import clip_grad_norm_
from matplotlib.pyplot import cm

from nnutils.geom_utils import lbs, reinit_bones, warp_bw, warp_fw, vec_to_sim3,\
                               obj_to_cam, get_near_far, near_far_to_bound, \
                               compute_point_visibility, process_so3_seq, \
                               ood_check_cse, align_sfm_sim3, skinning, \
                               zero_to_rest_bone, fid_reindex, K2mat, mat2K, K2inv, \
                               create_base_se3, refine_rt, skinning, extract_bone_sdf,\
                                optimize_scale, extract_mesh, se3_mat2vec
from nnutils.urdf_utils import visualize_joints
from nnutils.nerf import grab_xyz_weights
from ext_utils.flowlib import flow_to_image
from utils.io import mkdir_p
from nnutils.vis_utils import image_grid
from dataloader import frameloader
from utils.io import save_vid, draw_cams, extract_data_info, merge_dict,\
        render_root_txt, save_bones, draw_cams_pair
from utils.colors import label_colormap

class DataParallelPassthrough(torch.nn.parallel.DistributedDataParallel):
    """
    for multi-gpu access
    """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
    
    def __delattr__(self, name):
        try:
            return super().__delattr__(name)
        except AttributeError:
            return delattr(self.module, name)
    
class v2s_trainer():
    def __init__(self, opts, is_eval=False):
        self.opts = opts
        self.is_eval=is_eval
        self.local_rank = opts.local_rank
        self.save_dir = os.path.join(opts.checkpoint_dir, opts.logname)
        
        self.accu_steps = opts.accu_steps
        
        # write logs
        if opts.local_rank==0:
            if not os.path.exists(self.save_dir): os.makedirs(self.save_dir)
            log_file = os.path.join(self.save_dir, 'opts.log')
            if not self.is_eval:
                if os.path.exists(log_file):
                    os.remove(log_file)
                opts.append_flags_into_file(log_file)

    def define_model(self, data_info):
        opts = self.opts
        self.device = torch.device('cuda:{}'.format(opts.local_rank))
        self.model = banmo.banmo(opts, data_info)
        self.model.forward = self.model.forward_default
        self.num_epochs = opts.num_epochs

        # load model
        if opts.model_path!='':
            self.load_network(opts.model_path, is_eval=self.is_eval)
        if self.is_eval==False and opts.bg_path!= "":
            self.load_bg_network()
# and 'trsi_2d.near_far' not in states.keys(): # new bg model

        if self.is_eval:
            self.model = self.model.to(self.device)
        else:
            # ddp
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = self.model.to(self.device)

            self.model = DataParallelPassthrough(
                    self.model,
                    device_ids=[opts.local_rank],
                    output_device=opts.local_rank,
                    find_unused_parameters=True,
            )
        return
    
    def init_dataset(self):
        opts = self.opts
        opts_dict = {}
        opts_dict['n_data_workers'] = opts.n_data_workers
        opts_dict['batch_size'] = opts.batch_size
        opts_dict['seqname'] = opts.seqname
        opts_dict['img_size'] = opts.img_size
        opts_dict['ngpu'] = opts.ngpu
        opts_dict['local_rank'] = opts.local_rank
        opts_dict['rtk_path'] = opts.rtk_path
        opts_dict['preload']= False
        opts_dict['accu_steps'] = opts.accu_steps
        opts_dict['crop_factor'] = opts.crop_factor

        if self.is_eval and opts.rtk_path=='' and opts.model_path!='':
            # automatically load cameras in the logdir
            model_dir = opts.model_path.rsplit('/',1)[0]
            cam_dir = '%s/init-cam/'%model_dir
            if os.path.isdir(cam_dir):
                opts_dict['rtk_path'] = cam_dir

        self.dataloader = frameloader.data_loader(opts_dict)
        if opts.lineload:
            opts_dict['load_prefix'] = opts.load_prefix
            opts_dict['lineload'] = True
            opts_dict['multiply'] = True # multiple samples in dataset
            self.trainloader = frameloader.data_loader(opts_dict)
            opts_dict['lineload'] = False
            del opts_dict['multiply']
        else:
            opts_dict['multiply'] = True
            self.trainloader = frameloader.data_loader(opts_dict)
            del opts_dict['multiply']
        opts_dict['img_size'] = opts.render_size
        self.evalloader = frameloader.eval_loader(opts_dict)

        # compute data offset
        data_info = extract_data_info(self.evalloader)
        return data_info
    
    def init_training(self):
        opts = self.opts
        # set as module attributes since they do not change across gpus
        self.model.module.final_steps = self.num_epochs * \
                                min(200,len(self.trainloader)) * opts.accu_steps
        # ideally should be greater than 200 batches

        params_nerf_coarse=[]
        params_nerf_beta=[]
        params_nerf_feat=[]
        params_nerf_beta_feat=[]
        params_nerf_fine=[]
        params_nerf_unc=[]
        params_phys_env=[]
        params_trsi_2d=[]
        params_trsi_2d_cam=[]
        params_nerf_skin=[]
        params_mlp_deform=[]
        params_dfm_code=[]
        params_nerf_vis=[]
        params_nerf_root_rts=[]
        params_nerf_body_rts=[]
        params_root_code=[]
        params_pose_code=[]
        params_env_code=[]
        params_vid_code=[]
        params_bones=[]
        params_sim3=[]
        params_skin_aux=[]
        params_ks=[]
        params_nerf_dp=[]
        params_csenet=[]
        params_cnn_ff=[]
        params_conv3d=[]
        params_dpproj=[]
        for name,p in self.model.named_parameters():
            if 'nerf_coarse' in name and 'beta' not in name:
                params_nerf_coarse.append(p)
            elif 'nerf_coarse' in name and 'beta' in name:
                params_nerf_beta.append(p)
            elif 'nerf_feat' in name and 'beta' not in name and 'conv3d_net' not in name:
                params_nerf_feat.append(p)
            elif 'nerf_feat' in name and 'beta' in name:
                params_nerf_beta_feat.append(p)
            elif 'nerf_fine' in name:
                params_nerf_fine.append(p)
            elif 'nerf_unc' in name:
                params_nerf_unc.append(p)
            elif 'phys_env' in name:
                params_phys_env.append(p)
            elif 'trsi_2d' in name and 'cam_mlp' not in name:
                params_trsi_2d.append(p)
            elif 'trsi_2d' in name and 'cam_mlp' in name:
                params_trsi_2d_cam.append(p)
            elif 'nerf_skin' in name:
                params_nerf_skin.append(p)
            elif 'mlp_deform' in name:
                params_mlp_deform.append(p)
            elif 'dfm_code' in name:
                params_dfm_code.append(p)
            elif 'nerf_vis' in name:
                params_nerf_vis.append(p)
            elif 'nerf_root_rts' in name:
                params_nerf_root_rts.append(p)
            elif 'nerf_body_rts' in name:
                params_nerf_body_rts.append(p)
            elif 'root_code' in name:
                params_root_code.append(p)
            elif 'pose_code' in name or 'rest_pose_code' in name:
                params_pose_code.append(p)
            elif 'env_code' in name:
                params_env_code.append(p)
            elif 'vid_code' in name:
                params_vid_code.append(p)
            elif 'module.bones' == name:
                params_bones.append(p)
            elif 'module.sim3' == name:
                params_sim3.append(p)
            elif 'module.skin_aux' == name:
                params_skin_aux.append(p)
            elif 'module.ks_param' == name:
                params_ks.append(p)
            elif 'nerf_dp' in name:
                params_nerf_dp.append(p)
            elif 'csenet' in name:
                params_csenet.append(p)
            elif 'cnn_ff' in name:
                params_cnn_ff.append(p)
            elif 'conv3d_net' in name:
                params_conv3d.append(p)
            elif 'dp_proj' in name:
                params_dpproj.append(p)
            else: continue
            if opts.local_rank==0:
                print('optimized params: %s'%name)

        self.optimizer = torch.optim.AdamW(
            [{'params': params_nerf_coarse},
             {'params': params_nerf_beta},
             {'params': params_nerf_feat},
             {'params': params_nerf_beta_feat},
             {'params': params_nerf_fine},
             {'params': params_nerf_unc},
             {'params': params_phys_env},
             {'params': params_trsi_2d},
             {'params': params_trsi_2d_cam},
             {'params': params_nerf_skin},
             {'params': params_mlp_deform},
             {'params': params_dfm_code},
             {'params': params_nerf_vis},
             {'params': params_nerf_root_rts},
             {'params': params_nerf_body_rts},
             {'params': params_root_code},
             {'params': params_pose_code},
             {'params': params_env_code},
             {'params': params_vid_code},
             {'params': params_bones},
             {'params': params_sim3},
             {'params': params_skin_aux},
             {'params': params_ks},
             {'params': params_nerf_dp},
             {'params': params_csenet},
             {'params': params_cnn_ff},
             {'params': params_conv3d},
             {'params': params_dpproj},
            ],
            lr=opts.learning_rate,betas=(0.9, 0.999),weight_decay=1e-4)

        if self.model.root_basis=='exp':
            lr_nerf_root_rts = 10
        elif self.model.root_basis=='cnn':
            lr_nerf_root_rts = 0.2
        elif self.model.root_basis=='mlp':
            lr_nerf_root_rts = 1 
        elif self.model.root_basis=='expmlp':
            lr_nerf_root_rts = 1 
        else: print('error'); exit()
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,\
                        [opts.learning_rate, # params_nerf_coarse
                         opts.learning_rate, # params_nerf_beta
                         opts.learning_rate, # params_nerf_feat
                      10*opts.learning_rate, # params_nerf_beta_feat
                         opts.learning_rate, # params_nerf_fine
                         opts.learning_rate, # params_nerf_unc
                         opts.learning_rate, # params_phys_env
                         opts.learning_rate, # params_trsi_2d
                         opts.learning_rate, # params_trsi_2d_cam
                         opts.learning_rate, # params_nerf_skin
                         opts.learning_rate, # params_mlp_deform
                         opts.learning_rate, # params_dfm_code
                         opts.learning_rate, # params_nerf_vis
        lr_nerf_root_rts*opts.learning_rate, # params_nerf_root_rts
                         opts.learning_rate, # params_nerf_body_rts
        lr_nerf_root_rts*opts.learning_rate, # params_root_code
                         opts.learning_rate, # params_pose_code
                         opts.learning_rate, # params_env_code
                         opts.learning_rate, # params_vid_code
                         opts.learning_rate, # params_bones
                         opts.learning_rate, # params_sim3
                      10*opts.learning_rate, # params_skin_aux
                      10*opts.learning_rate, # params_ks
                         opts.learning_rate, # params_nerf_dp
                     0.2*opts.learning_rate, # params_csenet
                     0.2*opts.learning_rate, # params_cnn_ff
                     0.2*opts.learning_rate, # params_conv3d
                         opts.learning_rate, # params_dpproj
            ],
            int(self.model.module.final_steps/self.accu_steps),
            pct_start=2./self.num_epochs, # use 2 epochs to warm up
            cycle_momentum=False, 
            anneal_strategy='linear',
            final_div_factor=1./5, div_factor = 25,
            )
    
    def save_network(self, epoch_label, prefix=''):
        if self.opts.local_rank==0:
            param_path = '%s/%sparams_%s.pth'%(self.save_dir,prefix,epoch_label)
            save_dict = self.model.state_dict()
            torch.save(save_dict, param_path)

            var_path = '%s/%svars_%s.npy'%(self.save_dir,prefix,epoch_label)
            latest_vars = self.model.latest_vars.copy()
            del latest_vars['fp_err']  
            del latest_vars['flo_err']   
            del latest_vars['sil_err'] 
            del latest_vars['flo_err_hist']
            np.save(var_path, latest_vars)
            return
    
    @staticmethod
    def rm_module_prefix(states, prefix='module'):
        new_dict = {}
        for i in states.keys():
            v = states[i]
            if i[:len(prefix)] == prefix:
                i = i[len(prefix)+1:]
            new_dict[i] = v
        return new_dict

    def load_network(self,model_path=None, is_eval=True, rm_prefix=True):
        opts = self.opts
        states = torch.load(model_path,map_location='cpu')
        if rm_prefix: states = self.rm_module_prefix(states)
        var_path = model_path.replace('params', 'vars').replace('.pth', '.npy')
        latest_vars = np.load(var_path,allow_pickle=True)[()]
        
        if is_eval:
            # load variables
            self.model.latest_vars = latest_vars
        else:
            self.model.latest_vars['mesh_rest'] = latest_vars['mesh_rest']
        
        # if size mismatch, delete all related variables
        if rm_prefix and states['near_far'].shape[0] != self.model.near_far.shape[0]:
            print('!!!deleting video specific dicts due to size mismatch!!!')
            self.del_key( states, 'near_far') 
            self.del_key( states, 'root_code.weight') # only applies to root_basis=mlp
            self.del_key( states, 'pose_code.weight')
            self.del_key( states, 'pose_code.basis_mlp.weight')
            self.del_key( states, 'robot.pose_code.basis_mlp.weight')
            self.del_key( states, 'nerf_body_rts.pose_code.weight')
            self.del_key( states, 'nerf_body_rts.pose_code.basis_mlp.weight')
            self.del_key( states, 'nerf_root_rts.0.weight')
            self.del_key( states, 'nerf_root_rts.root_code.weight')
            self.del_key( states, 'nerf_root_rts.root_code.basis_mlp.weight')
            self.del_key( states, 'nerf_root_rts.delta_rt.0.basis_mlp.weight')
            self.del_key( states, 'nerf_root_rts.base_rt.se3')
            self.del_key( states, 'nerf_root_rts.delta_rt.0.weight')
            self.del_key( states, 'env_code.weight')
            self.del_key( states, 'env_code.basis_mlp.weight')
            if 'vid_code.weight' in states.keys():
                self.del_key( states, 'vid_code.weight')
            if 'ks_param' in states.keys():
                self.del_key( states, 'ks_param')
            if 'nerf_coarse.vid_code.weight' in states.keys():
                self.del_key( states, 'nerf_coarse.vid_code.weight')
            if 'trsi_2d.tcode.basis_mlp.weight' in states.keys():
                self.del_key( states, 'trsi_2d.tcode.basis_mlp.weight')
            if 'robot.jlen_scale' in states.keys():
                self.del_key( states, 'robot.jlen_scale')
                self.del_key( states, 'nerf_body_rts.jlen_scale')
            if 'nerf_body_rts.sim3_vid' in states.keys():
                self.del_key( states, 'nerf_body_rts.sim3_vid')
                self.del_key( states, 'robot.sim3_vid')

            # delete pose basis(backbones)
            if not opts.keep_pose_basis:
                del_key_list = []
                for k in states.keys():
                    if 'nerf_body_rts' in k or 'nerf_root_rts' in k:
                        del_key_list.append(k)
                for k in del_key_list:
                    print(k)
                    self.del_key( states, k)
    
        if rm_prefix and opts.lbs and states['bones'].shape[0] != self.model.bones.shape[0]:
            self.del_key(states, 'bones')
            states = self.rm_module_prefix(states, prefix='nerf_skin')
            states = self.rm_module_prefix(states, prefix='nerf_body_rts')


        # load some variables
        # this is important for volume matching
        if latest_vars['obj_bound'].size==1:
            latest_vars['obj_bound'] = latest_vars['obj_bound'] * np.ones(3)
        self.model.latest_vars['obj_bound'] = latest_vars['obj_bound'] 

        ##TODO
        #del states['robot.jlen_scale']
        #del states['sim3']
        #del states['robot.sim3']
        #del states['nerf_body_rts.sim3']
        if 'trsi_2d.bg2world' in states.keys() and len(states['trsi_2d.bg2world'].shape)==3:
            del states['trsi_2d.bg2world']

        #TODO 
        #pdb.set_trace()
        #states['trsi_2d.bg2fg_scale'][0] = 2
        #states['trsi_2d.bg2fg_scale'][0] = 2
        
         
        # TODO tmp delete bgmlp env code
        if opts.bgmlp != '' and 'trsi_2d.env_code.basis_mlp.weight' in states.keys()\
            and self.model.trsi_2d.env_code.basis_mlp.weight.shape[1] !=\
            states['trsi_2d.env_code.basis_mlp.weight'].shape[1]:
            del_key_list = []
            for k in states.keys():
                if 'trsi_2d.env_code' in k or 'bg_mlp.env_code' in k or 'phys_env.bg_rts.env_code' in k:
                    del_key_list.append(k)
            for k in del_key_list:
                self.del_key( states, k)

        # load nerf_coarse, nerf_bone/root (not code), nerf_vis, nerf_feat, nerf_unc
        #TODO somehow, this will reset the batch stats for 
        # a pretrained cse model, to keep those, we want to manually copy to states
        if opts.ft_cse and \
          'csenet.net.backbone.fpn_lateral2.weight' not in states.keys():
            self.add_cse_to_states(self.model, states)
        self.model.load_state_dict(states, strict=False)

        return
    
    def load_bg_network(self):
        opts = self.opts
        # load bg weights
        bg_states = torch.load(opts.bg_path, map_location='cpu')
        if opts.copy_bgfl:
            # change focal length to bg focal length
            self.model.ks_param.data = bg_states['module.ks_param']

        if opts.bgmlp != '':
            bg_nerf_states = self.rm_module_prefix(bg_states, 
                        prefix='module.nerf_coarse')
            bg_cam_states = self.rm_module_prefix(bg_states, 
                        prefix='module.nerf_root_rts')
            bg_vis_states = self.rm_module_prefix(bg_states, 
                        prefix='module.nerf_vis')
            bg_env_states = self.rm_module_prefix(bg_states, 
                        prefix='module.env_code')
            self.model.trsi_2d.nerf_mlp.load_state_dict(bg_nerf_states, strict=False)
            self.model.trsi_2d.cam_mlp.load_state_dict( bg_cam_states, strict=False)
            self.model.trsi_2d.nerf_vis.load_state_dict(bg_vis_states, strict=False)
            self.model.trsi_2d.env_code.load_state_dict(bg_env_states, strict=False)
            self.model.trsi_2d.embedding_xyz.alpha = int(bg_states['module.alpha'][0].numpy())
            self.model.trsi_2d.embedding_dir.alpha = int(bg_states['module.alpha'][0].numpy())
            # load near far
            self.model.trsi_2d.near_far.data = bg_states['module.near_far']
            # vars
            bg_var_path = opts.bg_path.replace('params', 'vars').replace('.pth', '.npy')
            self.model.trsi_2d.latest_vars = np.load(bg_var_path,allow_pickle=True)[()]

    @staticmethod 
    def add_cse_to_states(model, states):
        states_init = model.state_dict()
        for k in states_init.keys():
            v = states_init[k]
            if 'csenet' in k:
                states[k] = v

    @staticmethod
    def eval_cam(opts, model, evalloader, idx_render=None): 
        """
        idx_render: list of frame index to render
        model.latest is changed
        """
        opts = opts
        with torch.no_grad():
            model.eval()
            # load data
            for dataset in evalloader.dataset.datasets:
                dataset.load_pair = False
            batch = []
            for i in idx_render:
                batch.append( evalloader.dataset[i] )
            batch = evalloader.collate_fn(batch)
            for dataset in evalloader.dataset.datasets:
                dataset.load_pair = True

            #TODO can be further accelerated
            model.convert_batch_input(batch)

            if opts.unc_filter:
                # process densepoe feature
                valid_list, error_list = ood_check_cse(model.dp_feats, 
                                        model.dp_embed, 
                                        model.dps.long())
                valid_list = valid_list.cpu().numpy()
                error_list = error_list.cpu().numpy()
            else:
                valid_list = np.ones( len(idx_render))
                error_list = np.zeros(len(idx_render))

            model.convert_root_pose()
            rtk = model.rtk
            kaug = model.kaug
            rt_raw = model.rt_raw

            # extract mesh sequences
            aux_seq = {
                       'is_valid':[],
                       'err_valid':[],
                       'rtk':[],
                       'rt_raw': [],
                       'kaug':[],
                       'frameid': [],
                       'impath':[],
                       }
            for idx,_ in enumerate(idx_render):
                frameid=int(model.frameid[idx].cpu().numpy())
                #if opts.local_rank==0: 
                print('extracting frame %d'%(frameid))
                aux_seq['rtk'].append(rtk[idx].cpu().numpy())
                aux_seq['rt_raw'].append(rt_raw[idx].cpu().numpy())
                aux_seq['kaug'].append(kaug[idx].cpu().numpy())
                aux_seq['frameid'].append(frameid)
                aux_seq['is_valid'].append(valid_list[idx])
                aux_seq['err_valid'].append(error_list[idx])
                
                impath = model.impath[frameid]
                aux_seq['impath'].append(impath)
        return aux_seq
  
    def eval(self, idx_render=None, dynamic_mesh=False): 
        """
        idx_render: list of frame index to render
        dynamic_mesh: whether to extract canonical shape, or dynamic shape
        """
        opts = self.opts
        with torch.no_grad():
            self.model.eval()

            # run marching cubes on canonical shape
            if idx_render is not None: # at inference time
                embedid=torch.Tensor(idx_render) # needs to +1 
                vidid,_ = fid_reindex(embedid, self.model.num_vid,
                        self.model.data_offset - range(self.model.num_vid+1))
                vidid = vidid[0]
            else:
                vidid = None
            ## to retrive shape/app code
            #if opts.cnn_code:
            #    batch = [self.evalloader.dataset[i] for i in idx_render]
            #    batch = self.evalloader.collate_fn(batch)
            #    self.model.set_input(batch)
            mesh_dict_rest = extract_mesh(self.model, opts.chunk, \
                                         opts.sample_grid3d, opts.mc_threshold,
                                            vidid=vidid, is_eval=self.is_eval)
            # save canonical mesh and extract skinning weights TODO merge
            mesh_rest = mesh_dict_rest['mesh']
            if len(mesh_rest.vertices)>100:
                self.model.latest_vars['mesh_rest'] = mesh_rest

            # choose a grid image or the whold video
            if idx_render is None: # render 9 frames
                idx_render = np.linspace(0,len(self.evalloader)-1, 9, dtype=int)

            # render
            chunk=opts.rnd_frame_chunk
            rendered_seq = defaultdict(list)
            aux_seq = {'mesh_rest': mesh_dict_rest['mesh'],
                       'mesh':[],
                       'rtk':[],
                       'se3':[],
                       'impath':[],
                       'bone':[],}
            #TODO get bg mesh, transform to fg coords
            if (opts.bgmlp=='nerf' or opts.bgmlp=='hmnerf') and vidid is not None:
                frameid=torch.cuda.LongTensor(idx_render)+vidid.long() # from evalid to frameid
                bgrt = self.model.trsi_2d.get_rts(frameid)
                bgk = self.model.ks_param[vidid.long()]
                bgk = bgk[None].repeat(len(idx_render),1)[:,None]
                bgrtk = torch.cat([bgrt, bgk],1)
                aux_seq['bgrtk'] = bgrtk.cpu().numpy()
                mesh_dict_bg = extract_mesh(self.model.trsi_2d, opts.chunk, \
                                             opts.sample_grid3d, opts.mc_threshold,
                                                vidid=vidid, is_eval=self.is_eval)
                aux_seq['mesh_bg'] = mesh_dict_bg['mesh']
            if opts.pre_skel!="":
                aux_seq['parent_idx'] = self.model.robot.urdf.parent_idx
                aux_seq['joint'] = []
                aux_seq['angle'] = []
                aux_seq['kps'] = []
                aux_seq['skel_render'] = []
            for j in range(0, len(idx_render), chunk):
                batch = []
                idx_chunk = idx_render[j:j+chunk]
                for i in idx_chunk:
                    batch.append( self.evalloader.dataset[i] )
                batch = self.evalloader.collate_fn(batch)
                rendered = self.render_vid(self.model, batch) 
            
                for k, v in rendered.items():
                    rendered_seq[k] += [v]
                    
                hbs=len(idx_chunk)
                sil_rszd = F.interpolate(self.model.masks[:hbs,None], 
                            (opts.render_size, opts.render_size))[:,0,...,None]
                rendered_seq['img'] += [self.model.imgs.permute(0,2,3,1)[:hbs]]
                rendered_seq['sil'] += [self.model.masks[...,None]      [:hbs]]
                rendered_seq['flo'] += [self.model.flow.permute(0,2,3,1)[:hbs]]
                rendered_seq['dpc'] += [self.model.dp_vis[self.model.dps.long()][:hbs]]
                rendered_seq['occ'] += [self.model.occ[...,None]      [:hbs]]
                rendered_seq['feat']+= [self.model.dp_feats.std(1)[...,None][:hbs]]
                rendered_seq['flo_coarse'][-1]       *= sil_rszd 
                #rendered_seq['img_loss_samp'][-1]    *= sil_rszd 
                if 'frame_cyc_dis' in rendered_seq.keys() and \
                    len(rendered_seq['frame_cyc_dis'])>0:
                    rendered_seq['frame_cyc_dis'][-1] *= 255/rendered_seq['frame_cyc_dis'][-1].max()
                    rendered_seq['frame_rigloss'][-1] *= 255/rendered_seq['frame_rigloss'][-1].max()
                if opts.use_embed:
                    rendered_seq['pts_pred'][-1] *= sil_rszd 
                    rendered_seq['pts_exp'] [-1] *= rendered_seq['sil_coarse'][-1]
                    rendered_seq['feat_err'][-1] *= sil_rszd
                    rendered_seq['feat_err'][-1] *= 255/rendered_seq['feat_err'][-1].max()
                    match_unc = rendered_seq['match_unc'][-1]
                    match_unc_binary = (match_unc > 0.8).float()
                    match_unc_norm = (match_unc - match_unc.min()) /\
                                     (match_unc.max() - match_unc.min())
                    rendered_seq['match_unc'][-1] = torch.cat([match_unc_norm,
                                                               match_unc_binary],2)
                if opts.use_proj:
                    rendered_seq['proj_err'][-1] *= sil_rszd
                    rendered_seq['proj_err'][-1] *= 255/rendered_seq['proj_err'][-1].max()
                if opts.use_unc:
                    rendered_seq['unc_pred'][-1] -= rendered_seq['unc_pred'][-1].min()
                    rendered_seq['unc_pred'][-1] *= 255/rendered_seq['unc_pred'][-1].max()
                if 'dph_pred' in rendered_seq.keys():
                    import matplotlib.cm
                    cmap = matplotlib.cm.get_cmap('plasma')
                    dph_pred = cmap(rendered_seq['dph_pred'][-1].cpu().numpy())
                    rendered_seq['dph_pred'][-1] = torch.Tensor(dph_pred[...,:3])

                # extract mesh sequences
                for idx in range(len(idx_chunk)):
                    frameid=self.model.frameid[idx].long()
                    embedid=self.model.embedid[idx].long()
                    print('extracting frame %d'%(frameid.cpu().numpy()))
                    # run marching cubes
                    if dynamic_mesh:
                        if not opts.queryfw:
                           mesh_dict_rest=None 
                        mesh_dict = extract_mesh(self.model,opts.chunk,
                                            opts.sample_grid3d, opts.mc_threshold,
                                        embedid=embedid, mesh_dict_in=mesh_dict_rest,
                                        is_eval=self.is_eval)
                        mesh=mesh_dict['mesh']
                        mesh.visual.vertex_colors = mesh_dict_rest['mesh'].\
                               visual.vertex_colors # assign rest surface color

                        # save bones
                        if 'bones' in mesh_dict.keys():
                            bone = mesh_dict['bones'][0].cpu().numpy()
                            aux_seq['bone'].append(bone)
                            se3 = mesh_dict['se3'][0,0].cpu().numpy()
                            aux_seq['se3'].append(se3)
                        # save joints
                        if opts.pre_skel!="":
                            joint = mesh_dict['joints'][0].cpu().numpy()
                            angle = mesh_dict['angles'].cpu().numpy()
                            kps   = mesh_dict['kps']
                            aux_seq['joint'].append(joint)
                            aux_seq['angle'].append(angle)
                            aux_seq['kps'].append(kps)
                        if 'skel_render' in mesh_dict.keys():
                            aux_seq['skel_render'].append(mesh_dict['skel_render'])
                    else:
                        mesh=mesh_dict_rest['mesh']
                    aux_seq['mesh'].append(mesh)

                    # save cams
                    aux_seq['rtk'].append(self.model.rtk[idx].cpu().numpy())
                    
                    # save image list
                    impath = self.model.impath[frameid]
                    aux_seq['impath'].append(impath)

            # save canonical mesh and extract skinning weights
            mesh_rest = aux_seq['mesh_rest']
            if len(mesh_rest.vertices)>100:
                self.model.latest_vars['mesh_rest'] = mesh_rest
            if opts.lbs:
                bones_rst = self.model.bones
                bones_rst,_ = zero_to_rest_bone(self.model, bones_rst)
                # compute skinning color
                if mesh_rest.vertices.shape[0]>100:
                    rest_verts = torch.Tensor(mesh_rest.vertices).to(self.device)
                    nerf_skin = self.model.nerf_skin if opts.nerf_skin else None
                    rest_pose_code = self.model.rest_pose_code(torch.Tensor([0])\
                                            .long().to(self.device))
                    skins,_ = skinning(rest_verts[None], 
                            self.model.embedding_xyz,
                            bones_rst, rest_pose_code, 
                            nerf_skin)
                    skins = skins[0]
                    skins = skins.cpu().numpy()
   
                    num_bones = skins.shape[-1]
                    colormap = label_colormap()
                    # TODO use a larger color map
                    colormap = np.repeat(colormap[None],4,axis=0).reshape(-1,3)
                    colormap = colormap[:num_bones]
                    colormap = (colormap[None] * skins[...,None]).sum(1)

                    mesh_rest_skin = mesh_rest.copy()
                    mesh_rest_skin.visual.vertex_colors = colormap
                    aux_seq['mesh_rest_skin'] = mesh_rest_skin
                    aux_seq['val_rest_skin'] = skins
                    self.model.latest_vars['mesh_rest_skin'] = mesh_rest_skin

                aux_seq['bone_rest'] = bones_rst.cpu().numpy()
        
            # draw camera trajectory
            suffix_id=999 
            if hasattr(self.model, 'epoch'):
                suffix_id = self.model.epoch
            if opts.local_rank==0:
                mesh_cam = draw_cams(aux_seq['rtk'])
                mesh_cam.export('%s/mesh_cam-%02d.obj'%(self.save_dir,suffix_id))
            
                mesh_path = '%s/mesh_rest-%02d.obj'%(self.save_dir,suffix_id)
                mesh_rest.export(mesh_path)
                
                if opts.lbs:
                    mesh_path = '%s/mesh_skin-%02d.obj'%(self.save_dir,suffix_id)
                    self.model.latest_vars['mesh_rest_skin'].export(mesh_path)
                    bone_rest = aux_seq['bone_rest']
                    bone_path = '%s/bone_rest-%02d.obj'%(self.save_dir,suffix_id)
                    save_bones(bone_rest, 0.1, bone_path)

                    if opts.pre_skel!="":
                        joint_path = '%s/joint_rest-%02d.obj'%(self.save_dir,suffix_id)
                        rob_path = '%s/rob-rest-%02d.jpg'%(self.save_dir,suffix_id)
                        joint_rest, angle_rest, robot_rendered, rest_robot_mesh,_ = \
                                visualize_joints(self.model, robot_save_path = rob_path)
                        save_bones(joint_rest[0].cpu().numpy(), 0.1, joint_path,
                                parent=self.model.robot.urdf.parent_idx)
                        self.model.latest_vars['rest_robot_mesh'] = rest_robot_mesh
                        robot_mesh_path = '%s/robot_rest-%02d.obj'%(self.save_dir,suffix_id)
                        rest_robot_mesh.export(robot_mesh_path)
                        aux_seq['sim3'] = self.model.robot.sim3.detach().cpu().numpy()

            # save images
            for k,v in rendered_seq.items():
                rendered_seq[k] = torch.cat(rendered_seq[k],0)
                ##TODO
                #if opts.local_rank==0:
                #    print('saving %s to gif'%k)
                #    is_flow = self.isflow(k)
                #    upsample_frame = min(30,len(rendered_seq[k]))
                #    save_vid('%s/%s'%(self.save_dir,k), 
                #            rendered_seq[k].cpu().numpy(), 
                #            suffix='.gif', upsample_frame=upsample_frame, 
                #            is_flow=is_flow)

        return rendered_seq, aux_seq

    def train(self):
        opts = self.opts
        if opts.local_rank==0:
            log = SummaryWriter('%s/%s'%(opts.checkpoint_dir,opts.logname), comment=opts.logname)
        else: log=None
        self.model.module.total_steps = 0
        self.model.module.progress = 0
        torch.manual_seed(8)  # do it again
        torch.cuda.manual_seed(1)

        # warmup shape
        if opts.warmup_shape_ep>0:
            self.warmup_shape(log)
        
        # warmup skin
        if opts.warmup_skin_ep>0 and opts.nerf_skin=='cmlp':
            self.warmup_skin(log)

        # CNN pose warmup or  load CNN
        if opts.warmup_pose_ep>0 or opts.pose_cnn_path!='':
            self.warmup_pose(log, pose_cnn_path=opts.pose_cnn_path)
        else:
            # save cameras to latest vars and file
            if opts.use_rtk_file:
                self.model.module.use_cam=True
                root_opt = opts.root_opt
                self.model.module.opts.root_opt=False
                self.extract_cams(self.dataloader)
                self.model.module.use_cam=opts.use_cam
                self.model.module.opts.root_opt = root_opt
            elif opts.use_cnn:
                # at scaling up stage, load intrinsics for each image
                if opts.warmup_rootmlp: 
                    self.load_cams(self.dataloader)
                else: pass
            else:
                if opts.warmup_rootmlp: 
                    self.extract_cams(self.dataloader)
                else: pass

        #TODO train mlp
        if opts.warmup_rootmlp:
            if not opts.use_cnn:
                # set se3 directly
                rmat = torch.Tensor(self.model.latest_vars['rtk'][:,:3,:3])
                quat = transforms.matrix_to_quaternion(rmat).to(self.device)
                self.model.module.nerf_root_rts.base_rt.se3.data[:,3:] = quat

            # set trans directly based on bbox + orthgraphic proj model
            # depth = fl * h3d/h2d = fl/512 * 0.2 (1/10 unit sphere)
            # xytrn = tz * (pxy-size/2)/fl
            if opts.depth_init=='hw':
                depth = np.sqrt(self.model.latest_vars['rtk'][:,3,0]*\
                                        self.model.latest_vars['rtk'][:,3,1])
            elif opts.depth_init=='h':
                depth = self.model.latest_vars['rtk'][:,3,1]
            depth = 0.1* depth / opts.render_size
            xytrn = depth[...,None] * \
                    (0.5*opts.render_size - self.model.latest_vars['rtk'][:,3,2:4]) \
                    / self.model.latest_vars['rtk'][:,3,:2]
            tmat = torch.Tensor(np.concatenate([xytrn, depth[...,None]],1)).to(self.device)
            tmat[depth==0] = tmat[np.where(depth==0)[0]-1]
            self.model.module.latest_vars['rtk'][:,:3,3] = tmat.cpu().detach()

            tmat[...,2] -= 0.3
            tmat = tmat*10 # to accound for *0.1 in expmlp
            self.model.module.nerf_root_rts.base_rt.se3.data[:,:3] = tmat

            if opts.use_cnn:
                # update near-far planes (for vis)
                self.model.module.near_far.data = get_near_far(
                                              self.model.module.near_far.data,
                                              self.model.module.latest_vars,
                                              tol_fac=1.2)
                self.model.module.near_far.data[depth==0] = \
                self.model.module.near_far.data[np.where(depth==0)[0]-1]
            else:
                # update cached vars
                rtk_all = self.model.module.compute_rts()
                self.model.module.latest_vars['rtk'][:,:3] = rtk_all.clone().detach().cpu().numpy()
                self.model.module.latest_vars['rtk'][depth==0] = \
                         self.model.module.latest_vars['rtk'][np.where(depth==0)[0]-1]
                self.model.module.latest_vars['idk'][depth==0] = \
                         self.model.module.latest_vars['idk'][np.where(depth==0)[0]-1]

        if opts.fit_scale and opts.local_rank==0:
            #TODO fit scale
            self.model.eval()
            with torch.no_grad():
                fgrts = self.model.compute_rts()
            for vidid in opts.phys_vid:
                #vidid=1
                with torch.no_grad():
                    vidid=torch.tensor(vidid, device=self.device)
                    mesh_bg = extract_mesh(self.model.trsi_2d, opts.chunk, \
                                             256, 0,
                                             #opts.sample_grid3d, opts.mc_threshold,
                                                vidid=vidid, is_eval=True)['mesh']
                    # need to use a strict threshold and replace this TODO
                    #mesh_bg = self.model.trsi_2d.latest_vars['mesh_%02d'%(vidid)]
                    mesh_fg = self.model.latest_vars['mesh_rest']
                    # get bgcams and fgcams
                    frameids = range(self.model.data_offset[vidid],self.model.data_offset[vidid+1])
                    frameids = torch.tensor(frameids, device=self.device)
                    bgrt = self.model.trsi_2d.get_rts(frameids)
                    fgrt = fgrts[frameids]
                    dummy = torch.tensor([[0,0,0,1]], device=self.device)
                    dummy = dummy.repeat(len(frameids),1)[:,None]
                    bgrt = torch.cat([bgrt, dummy],1).cpu()
                    fgrt = torch.cat([fgrt, dummy],1).cpu()
                    # find the lowest point
                    low_pt_idx = np.argmin(mesh_fg.vertices[:,1])
                    low_pt = mesh_fg.vertices[low_pt_idx][None]
                    #high_pt_idx = np.argmax(mesh_fg.vertices[:,1])
                    #high_pt = mesh_fg.vertices[high_pt_idx][None]
                    high_pt = mesh_fg.vertices.mean(0)[None]
                    # find the trajectory of the lowest point
                    #low_pt_traj,_ = warp_fw(self.opts, self.model, {}, 
                    #        low_pt, frameids[:,None], robot_render=False)
                    #low_pt_traj = np.concatenate([low_pt_traj,np.ones_like(low_pt_traj)[:,:1]],-1)
                    #low_pt_traj = low_pt_traj[...,None]
                    from nnutils.urdf_utils import query_kps
                    low_pt_traj = query_kps(self.model,query_time=frameids) # T,4,k
                    # find the trajectory of the center point
                    high_pt_traj,_ = warp_fw(self.opts, self.model, {}, 
                            high_pt, frameids[:,None], robot_render=False)
                    high_pt_traj = np.concatenate([high_pt_traj,np.ones_like(high_pt_traj)[:,:1]],-1)
                    high_pt_traj = high_pt_traj[...,None]
                # fit the scale
                #use_mesh_plane = len(self.model.robot.urdf.kp_links)<3 # bipedal
                use_mesh_plane = True
                bg2fg_scale,_,bg2world,_ = optimize_scale(bgrt, fgrt, low_pt_traj, 
                    mesh_bg, debug=True,high_pt=high_pt_traj,use_mesh_plane=use_mesh_plane)
                #TODO
                mesh_bg.vertices = mesh_bg.vertices * bg2fg_scale
                mesh_bg.vertices = mesh_bg.vertices @ bg2world[:3,:3].T + bg2world[:3,3][None]
                mesh_bg.export('tmp/meshbg-%d.obj'%vidid)
                bg2world = torch.Tensor(bg2world).to(self.device)
                self.model.module.trsi_2d.bg2world.data[vidid] = se3_mat2vec(bg2world)
                self.model.module.trsi_2d.bg2fg_scale.data[vidid] *= bg2fg_scale[0]
                self.model.module.trsi_2d.bg2fg_scale.data[vidid] *= 1.2 # TODO floating
                #break
        if opts.fit_scale:
            # sync up scale across gpus
            dist.broadcast(self.model.module.trsi_2d.bg2world,0)
            dist.broadcast(self.model.module.trsi_2d.bg2fg_scale,0)

        # clear buffers for pytorch1.10+
        try: self.model._assign_modules_buffers()
        except: pass

        # set near-far plane
        if opts.model_path=='':
            self.reset_nf()

        # reset idk in latest_vars
        self.model.module.latest_vars['idk'][:] = 0.
   
        #TODO reset beta
        if opts.reset_beta:
            self.model.module.nerf_coarse.beta.data[:] = 0.1

        # start training
        for epoch in range(0, self.num_epochs):
            self.model.epoch = epoch

            # evaluation
            torch.cuda.empty_cache()
            self.model.module.img_size = opts.render_size
            rendered_seq, aux_seq = self.eval()                
            self.model.module.img_size = opts.img_size
            if epoch==0: self.save_network('0') # to save some cameras
            if opts.local_rank==0: self.add_image_grid(rendered_seq, log, epoch)
            if opts.local_rank==0: 
                self.add_mesh(log, 'mesh_rest', aux_seq['mesh_rest'], epoch)

            self.reset_hparams(epoch)

            torch.cuda.empty_cache()
            
            ## TODO harded coded
            #if opts.freeze_proj:
            #    if self.model.module.progress<0.8:
            #        #opts.nsample=64
            #        opts.ndepth=2
            #    else:
            #        #opts.nsample = nsample
            #        opts.ndepth = self.model.module.ndepth_bk

            self.train_one_epoch(epoch, log)
            
            print('saving the model at the end of epoch {:d}, iters {:d}'.\
                              format(epoch, self.model.module.total_steps))
            self.save_network('latest')
            self.save_network(str(epoch+1))

    @staticmethod
    def save_cams(opts,aux_seq, save_prefix, latest_vars,datasets, evalsets, obj_scale,
            trainloader=None, unc_filter=True):
        """
        save cameras to dir and modify dataset 
        """
        mkdir_p(save_prefix)
        dataset_dict={dataset.imglist[0].split('/')[-2]:dataset for dataset in datasets}
        evalset_dict={dataset.imglist[0].split('/')[-2]:dataset for dataset in evalsets}
        if trainloader is not None:
            line_dict={dataset.imglist[0].split('/')[-2]:dataset for dataset in trainloader}

        length = len(aux_seq['impath'])
        valid_ids = aux_seq['is_valid']
        idx_combine = 0
        for i in range(length):
            impath = aux_seq['impath'][i]
            seqname = impath.split('/')[-2]
            rtk = aux_seq['rtk'][i]
           
            if unc_filter:
                # in the same sequance find the closest valid frame and replace it
                seq_idx = np.asarray([seqname == i.split('/')[-2] \
                        for i in aux_seq['impath']])
                valid_ids_seq = np.where(valid_ids * seq_idx)[0]
                if opts.local_rank==0 and i==0: 
                    print('%s: %d frames are valid'%(seqname, len(valid_ids_seq)))
                if len(valid_ids_seq)>0 and not aux_seq['is_valid'][i]:
                    closest_valid_idx = valid_ids_seq[np.abs(i-valid_ids_seq).argmin()]
                    rtk[:3,:3] = aux_seq['rtk'][closest_valid_idx][:3,:3]

            # rescale translation according to input near-far plane
            rtk[:3,3] = rtk[:3,3]*obj_scale
            rtklist = dataset_dict[seqname].rtklist
            idx = int(impath.split('/')[-1].split('.')[-2])
            save_path = '%s/%s-%05d.txt'%(save_prefix, seqname, idx)
            np.savetxt(save_path, rtk)
            rtklist[idx] = save_path
            evalset_dict[seqname].rtklist[idx] = save_path
            if trainloader is not None:
                line_dict[seqname].rtklist[idx] = save_path
            
            #save to rtraw 
            latest_vars['rt_raw'][idx_combine] = rtk[:3,:4]
            latest_vars['rtk'][idx_combine,:3,:3] = rtk[:3,:3]

            if idx==len(rtklist)-2:
                # to cover the last
                save_path = '%s/%s-%05d.txt'%(save_prefix, seqname, idx+1)
                if opts.local_rank==0: print('writing cam %s'%save_path)
                np.savetxt(save_path, rtk)
                rtklist[idx+1] = save_path
                evalset_dict[seqname].rtklist[idx+1] = save_path
                if trainloader is not None:
                    line_dict[seqname].rtklist[idx+1] = save_path

                idx_combine += 1
                latest_vars['rt_raw'][idx_combine] = rtk[:3,:4]
                latest_vars['rtk'][idx_combine,:3,:3] = rtk[:3,:3]
            idx_combine += 1
        
        
    def extract_cams(self, full_loader):
        # store cameras
        opts = self.opts
        idx_render = range(len(self.evalloader))

        # parallelized version this
        chunk = int(np.ceil(len(idx_render)/opts.ngpu))
        aux_seq = [[None]]*opts.ngpu
        for i in range(0, opts.ngpu):
            if opts.local_rank==i:
                aux_seq_sub = self.eval_cam(self.opts, self.model, self.evalloader,\
                                    idx_render=idx_render[i*chunk:(i+1)*chunk])
                aux_seq[i] = [aux_seq_sub]
            else:
                aux_seq[i] = [{}]
        dist.barrier()
        for i in range(0, opts.ngpu):
            dist.broadcast_object_list(aux_seq[i],i) # [None] => [dict]
        dist.barrier()
        for i in range(len(aux_seq)):
            aux_seq[i] = aux_seq[i][0]

        aux_seq = merge_dict(aux_seq)

        #TODO may need to recompute after removing the invalid predictions
        # need to keep this to compute near-far planes
        self.model.save_latest_vars(self.model.latest_vars,
            aux_seq['rtk'], aux_seq['kaug'], aux_seq['frameid'], aux_seq['rt_raw'])

        aux_seq['rtk'] = np.asarray(aux_seq['rtk'])
        aux_seq['kaug'] = np.asarray(aux_seq['kaug'])
        aux_seq['is_valid'] = np.asarray(aux_seq['is_valid'])
        aux_seq['err_valid'] = np.asarray(aux_seq['err_valid'])

        save_prefix = '%s/init-cam'%(self.save_dir)
        trainloader=self.trainloader.dataset.datasets
        self.save_cams(opts,aux_seq, save_prefix,
                    self.model.module.latest_vars,
                    full_loader.dataset.datasets,
                self.evalloader.dataset.datasets,
                self.model.obj_scale, trainloader=trainloader,
                unc_filter=opts.unc_filter)
        
        dist.barrier() # wait untail all have finished
        if opts.local_rank==0:
            # draw camera trajectory
            for dataset in full_loader.dataset.datasets:
                seqname = dataset.imglist[0].split('/')[-2]
                render_root_txt('%s/%s-'%(save_prefix,seqname), 0)
        
    def load_cams(self, full_loader):
        # store cameras => only intrinsics
        opts = self.opts
        idx_render = range(len(self.evalloader))
        chunk = 50
        aux_seq = []
        for i in range(0, len(idx_render), chunk):
            # load data
            for dataset in self.evalloader.dataset.datasets:
                dataset.load_pair = False
            batch = []
            for i in idx_render[i:i+chunk]:
                batch.append( self.evalloader.dataset[i] )
            batch = self.evalloader.collate_fn(batch)
            for dataset in self.evalloader.dataset.datasets:
                dataset.load_pair = True

            if batch['img'].dim()==4:
                bs,_,h,w = batch['img'].shape
            else:
                bs,_,_,h,w = batch['img'].shape
            device = self.device
            imgs         = batch['img'].view(bs,-1,3,h,w).permute(1,0,2,3,4).reshape(-1,3,h,w).float().to(device)
            rtk          = batch['rtk']         .view(bs,-1,4,4).permute(1,0,2,3).reshape(-1,4,4).float().to(device)
            kaug         = batch['kaug']        .view(bs,-1,4).permute(1,0,2).reshape(-1,4)          .to(device)
            frameid      = batch['frameid']     .view(bs,-1).permute(1,0).reshape(-1).cpu()
            dataid       = batch['dataid']      .view(bs,-1).permute(1,0).reshape(-1).cpu()
            frameid = frameid + self.model.data_offset[dataid.long()]
           
            rtk[:,:3] = create_base_se3(bs, device)
            root_rts = self.model.cnn_ff(imgs)
            rtk = refine_rt(rtk, root_rts)
            rtk[:,3,:] = self.model.ks_param[dataid.long()] #TODO kmat
            
            rtk = rtk.clone().detach()
            Kmat = K2mat(rtk[:,3])
            Kaug = K2inv(kaug) # p = Kaug Kmat P
            rtk[:,3] = mat2K(Kaug.matmul(Kmat))
            self.model.module.latest_vars['rtk'][frameid.long()] = rtk.cpu().numpy()
            self.model.module.latest_vars['idk'][frameid.long()] = 1
                
    def reset_nf(self):
        opts = self.opts
        # save near-far plane
        shape_verts = self.model.dp_verts_unit / 3 * self.model.near_far.mean() / 3
        shape_verts = shape_verts * 1.2
        # save object bound if first stage
        if opts.model_path=='' and opts.bound_factor>0:
            shape_verts = shape_verts*opts.bound_factor
            self.model.module.latest_vars['obj_bound'] = \
            shape_verts.abs().max(0)[0].detach().cpu().numpy()

        if self.model.near_far[:,0].sum()==0: # if no valid nf plane loaded
            self.model.near_far.data = get_near_far(self.model.near_far.data,
                                                self.model.latest_vars,
                                         pts=shape_verts.detach().cpu().numpy())
        save_path = '%s/init-nf.txt'%(self.save_dir)
        save_nf = self.model.near_far.data.cpu().numpy() * self.model.obj_scale
        np.savetxt(save_path, save_nf)
    
    def warmup_shape(self, log):
        opts = self.opts

        # force using warmup forward, dataloader, cnn root
        self.model.module.forward = self.model.module.forward_warmup_shape
        full_loader = self.trainloader  # store original loader
        self.trainloader = range(200)
        self.num_epochs = opts.warmup_shape_ep

        # training
        self.init_training()
        for epoch in range(0, opts.warmup_shape_ep):
            self.model.epoch = epoch
            self.train_one_epoch(epoch, log, warmup=True)

        # restore dataloader, rts, forward function
        self.model.module.forward = self.model.module.forward_default
        self.trainloader = full_loader
        self.num_epochs = opts.num_epochs

        # start from low learning rate again
        self.init_training()
        self.model.module.total_steps = 0
        self.model.module.progress = 0.
    
    def warmup_skin(self, log):
        opts = self.opts

        # force using warmup forward, dataloader, cnn root
        self.model.module.forward = self.model.module.forward_warmup_skin
        full_loader = self.trainloader  # store original loader
        self.trainloader = range(200)
        self.num_epochs = opts.warmup_skin_ep

        # training
        self.init_training()
        for epoch in range(0, opts.warmup_skin_ep):
            self.model.epoch = epoch
            self.train_one_epoch(epoch, log, warmup=True)

        # restore dataloader, rts, forward function
        self.model.module.forward = self.model.module.forward_default
        self.trainloader = full_loader
        self.num_epochs = opts.num_epochs

        # start from low learning rate again
        self.init_training()
        self.model.module.total_steps = 0
        self.model.module.progress = 0.

    def warmup_pose(self, log, pose_cnn_path):
        opts = self.opts

        # force using warmup forward, dataloader, cnn root
        self.model.module.root_basis = 'cnn'
        self.model.module.use_cam = False
        self.model.module.forward = self.model.module.forward_warmup
        full_loader = self.dataloader  # store original loader
        self.dataloader = range(200)
        original_rp = self.model.module.nerf_root_rts
        self.model.module.nerf_root_rts = self.model.module.dp_root_rts
        del self.model.module.dp_root_rts
        self.num_epochs = opts.warmup_pose_ep
        self.model.module.is_warmup_pose=True

        if pose_cnn_path=='':
            # training
            self.init_training()
            for epoch in range(0, opts.warmup_pose_ep):
                self.model.epoch = epoch
                self.train_one_epoch(epoch, log, warmup=True)
                self.save_network(str(epoch+1), 'cnn-') 

                # eval
                #_,_ = self.model.forward_warmup(None)
                # rendered_seq = self.model.warmup_rendered 
                # if opts.local_rank==0: self.add_image_grid(rendered_seq, log, epoch)
        else: 
            pose_states = torch.load(opts.pose_cnn_path, map_location='cpu')
            pose_states = self.rm_module_prefix(pose_states, 
                    prefix='module.nerf_root_rts')
            self.model.module.nerf_root_rts.load_state_dict(pose_states, 
                                                        strict=False)

        # extract camera and near far planes
        self.extract_cams(full_loader)

        # restore dataloader, rts, forward function
        self.model.module.root_basis=opts.root_basis
        self.model.module.use_cam = opts.use_cam
        self.model.module.forward = self.model.module.forward_default
        self.dataloader = full_loader
        del self.model.module.nerf_root_rts
        self.model.module.nerf_root_rts = original_rp
        self.num_epochs = opts.num_epochs
        self.model.module.is_warmup_pose=False

        # start from low learning rate again
        self.init_training()
        self.model.module.total_steps = 0
        self.model.module.progress = 0.
            
    def train_one_epoch(self, epoch, log, warmup=False):
        """
        training loop in a epoch
        """
        opts = self.opts
        self.model.train()
        dataloader = self.trainloader
    
        if not warmup: dataloader.sampler.set_epoch(epoch) # necessary for shuffling
        for i, batch in enumerate(dataloader):
            if i==200*opts.accu_steps:
                break
            
            if opts.debug:
                if 'start_time' in locals().keys():
                    torch.cuda.synchronize()
                    print('load time:%.2f'%(time.time()-start_time))

            if not warmup:
                self.model.module.progress = float(self.model.total_steps) /\
                                               self.model.final_steps
                self.select_loss_indicator(i)
                self.update_root_indicator(i)
                self.update_body_indicator(i)
                self.update_shape_indicator(i)
                self.update_skin_indicator(i)
                self.update_cvf_indicator(i)

            # 0-400 / 400-4000 =:> 
            #if self.opts.phys_opt:
            cyc_len = opts.cyc_len
            phys_cyc_len = opts.phys_cyc_len
            if self.opts.phys_opt and self.model.total_steps%cyc_len<phys_cyc_len:
            #if self.opts.phys_opt and self.model.total_steps%4000<401:
            #if self.opts.phys_opt and self.model.total_steps%5<1:
            #if self.opts.phys_opt and epoch%20<2:
                # DR to DP
                if self.model.total_steps%cyc_len==0: 
                    self.model.phys_env.override_states()
                    self.model.phys_env.total_loss_hist = []
                #if self.model.total_steps%4000==0: self.model.phys_env.override_states()
                #elif self.model.total_steps%4000==400: self.model.phys_env.override_states_inv()
                torch.cuda.empty_cache()
                self.model.module.forward = self.model.module.forward_phys
                phys_loss, phys_aux = self.model(None)
                self.model.module.forward = self.model.module.forward_default
                total_loss = phys_loss
                aux_out = phys_aux
                self.model.phys_env.backward(total_loss) 
                grad_list = self.model.phys_env.update()
                aux_out.update(grad_list)

                # DP to DR
                if self.model.total_steps%cyc_len==phys_cyc_len-1: 
                    if self.model.phys_env.total_loss_hist[-1] > \
                            self.model.phys_env.total_loss_hist[0]*10:
                        latest_path = '%s/params_latest.pth'%(self.save_dir)
                        self.load_network(latest_path, is_eval=False, rm_prefix=False)
                    else:
                        self.model.phys_env.override_states_inv()
            else:
                total_loss,aux_out = self.model(batch)
                total_loss = total_loss/self.accu_steps

                if opts.debug:
                    if 'start_time' in locals().keys():
                        torch.cuda.synchronize()
                        print('forward time:%.2f'%(time.time()-start_time))

                total_loss.mean().backward()
                
                if opts.debug:
                    if 'start_time' in locals().keys():
                        torch.cuda.synchronize()
                        print('forward back time:%.2f'%(time.time()-start_time))

                if (i+1)%self.accu_steps == 0:
                    self.clip_grad(aux_out)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    #if aux_out['nerf_root_rts_g']>1*opts.clip_scale and \
                    #                self.model.total_steps>200*self.accu_steps:
                    #    latest_path = '%s/params_latest.pth'%(self.save_dir)
                    #    self.load_network(latest_path, is_eval=False, rm_prefix=False)
                    
                for i,param_group in enumerate(self.optimizer.param_groups):
                    aux_out['lr_%02d'%i] = param_group['lr']

            self.model.module.total_steps += 1
            self.model.module.counter_frz_rebone -= 1./self.model.final_steps
            aux_out['counter_frz_rebone'] = self.model.module.counter_frz_rebone
            self.model.module.counter_frz_randrt -= 1./self.model.final_steps
            aux_out['counter_frz_randrt'] = self.model.module.counter_frz_randrt

            # update random code prob for category model
            if opts.use_category:
                rand_ratio = opts.catep_bgn + self.model.module.progress *\
                                            (opts.catep_end-opts.catep_bgn)
                self.model.module.nerf_coarse.rand_ratio = rand_ratio
                aux_out['rand_ratio'] = rand_ratio
            if opts.symm_shape:
                symm_ratio = opts.symm_bgn + self.model.module.progress *\
                                            (opts.symm_end-opts.symm_bgn)
                self.model.module.nerf_coarse.symm_ratio = np.clip(symm_ratio, 0, 1)
                aux_out['symm_ratio'] = symm_ratio

            if opts.local_rank==0: 
                self.save_logs(log, aux_out, self.model.module.total_steps, 
                        epoch)
            
            if opts.debug:
                if 'start_time' in locals().keys():
                    torch.cuda.synchronize()
                    print('total step time:%.2f'%(time.time()-start_time))
                torch.cuda.synchronize()
                start_time = time.time()
    
    def update_cvf_indicator(self, i):
        """
        whether to update canoical volume features
        1: update all
        0: freeze 
        """
        opts = self.opts

        # incremental optimization
        # or during kp reprojection optimization
        if (opts.model_path!='' and \
        self.model.module.progress < opts.warmup_steps)\
         or (opts.freeze_proj and self.model.module.progress >= opts.proj_start and \
               self.model.module.progress <(opts.proj_start + opts.proj_end)):
            self.model.module.cvf_update = 0
        else:
            self.model.module.cvf_update = 1
        
        # freeze shape after rebone        
        if self.model.module.counter_frz_rebone > 0:
            self.model.module.cvf_update = 0

        # freeze shape after random noise
        if self.model.module.counter_frz_randrt > 0:
            self.model.module.cvf_update = 0

        if opts.freeze_cvf:
            self.model.module.cvf_update = 0
    
    def update_shape_indicator(self, i):
        """
        whether to update shape
        1: update all
        0: freeze shape
        """
        opts = self.opts
        # incremental optimization
        # or during kp reprojection optimization
        if (opts.model_path!='' and \
        self.model.module.progress < opts.warmup_steps)\
         or (opts.freeze_proj and self.model.module.progress >= opts.proj_start and \
               self.model.module.progress <(opts.proj_start + opts.proj_end)):
            self.model.module.shape_update = 0
        else:
            self.model.module.shape_update = 1

        # freeze shape after rebone        
        if self.model.module.counter_frz_rebone > 0:
            self.model.module.shape_update = 0

        # freeze shape after rand root pose
        if self.model.module.counter_frz_randrt > 0:
            self.model.module.shape_update = 0

        if opts.freeze_shape:
            self.model.module.shape_update = 0
    
    def update_skin_indicator(self, i):
        """
        whether to update shape
        1: update all
        0: freeze shape
        """
        opts = self.opts
        # incremental optimization
        if (opts.model_path!='' and \
        self.model.module.progress < opts.warmup_steps):
            self.model.module.skin_update = 0
        else:
            self.model.module.skin_update = 1

        # freeze after rand root pose
        if self.model.module.counter_frz_randrt > 0:
            self.model.module.skin_update = 0

        if opts.freeze_skin:
            self.model.module.skin_update = 0
    
    def update_root_indicator(self, i):
        """
        whether to update root pose
        1: update
        0: freeze
        """
        opts = self.opts
        if (opts.freeze_proj and \
            opts.root_stab and \
           self.model.module.progress >=(opts.frzroot_start) and \
           self.model.module.progress <=(opts.proj_start + opts.proj_end+0.01))\
           : # to stablize
            self.model.module.root_update = 0
        else:
            self.model.module.root_update = 1
        
        # freeze shape after rebone        
        if self.model.module.counter_frz_rebone > 0:
            self.model.module.root_update = 0
        
        if opts.freeze_root: # to stablize
            self.model.module.root_update = 0
    
    def update_body_indicator(self, i):
        """
        whether to update root pose
        1: update
        0: freeze
        """
        opts = self.opts
        if opts.freeze_proj and \
           self.model.module.progress <=opts.frzbody_end: 
            self.model.module.body_update = 0
        else:
            self.model.module.body_update = 1

        if self.opts.freeze_body_mlp:
            self.model.module.body_update = 0

        
    def select_loss_indicator(self, i):
        """
        0: flo
        1: flo/sil/rgb
        """
        opts = self.opts
        if not opts.root_opt or \
            self.model.module.progress > (opts.warmup_steps):
            self.model.module.loss_select = 1
        elif i%2 == 0:
            self.model.module.loss_select = 0
        else:
            self.model.module.loss_select = 1

        #self.model.module.loss_select=1
        

    def reset_hparams(self, epoch):
        """
        reset hyper-parameters based on current geometry / cameras
        """
        opts = self.opts
        mesh_rest = self.model.latest_vars['mesh_rest']

        # reset object bound, for feature matching
        if epoch>int(self.num_epochs*(opts.bound_reset)):
            if mesh_rest.vertices.shape[0]>100:
                self.model.latest_vars['obj_bound'] = 1.5*np.abs(mesh_rest.vertices).max(0)
        
        # reinit bones based on extracted surface
        # only reinit for the initialization phase
        #if opts.lbs and opts.model_path=='' and \
        if opts.lbs and opts.model_path=='' and \
                        (epoch==int(self.num_epochs*opts.reinit_bone_steps) or\
                         epoch==int(self.num_epochs*opts.warmup_steps)//2):
            reinit_bones(self.model.module, mesh_rest, opts.num_bones)
            if opts.nerf_skin: self.model.module.nerf_skin.reinit()
            self.init_training() # add new params to optimizer
            if epoch>0:
                # freeze weights of root pose in the following 1% iters
                self.model.module.counter_frz_rebone = 0.02
                #reset error stats
                self.model.module.latest_vars['fp_err']      [:]=0
                self.model.module.latest_vars['flo_err']     [:]=0
                self.model.module.latest_vars['sil_err']     [:]=0
                self.model.module.latest_vars['flo_err_hist'][:]=0

        # need to add bones back at 2nd opt
        if opts.model_path!='' and opts.lbs:
            self.model.module.nerf_models['bones'] = self.model.module.bones

        # add nerf-skin when the shape is good
        if opts.lbs and opts.nerf_skin and \
                epoch==int(self.num_epochs*opts.dskin_steps):
            self.model.module.nerf_models['nerf_skin'] = self.model.module.nerf_skin

        # inject noise to root pose
        #if epoch>0 and opts.rand_rep>0 and \
        if opts.rand_rep>0 and \
           epoch%int(self.num_epochs*opts.rand_rep)==0 and \
           (1 - epoch/self.num_epochs)>opts.rand_rep:
            # load current poses and plot camera before randomaization
            cam_pre =self.model.compute_rts()
            rcoarse = cam_pre[:,:3,:3]
            gain = np.exp(-5*self.model.module.progress)

            # add Gaussian noise (0,std) in deg for each rotation axis => sqrt(3)std degree in total
            rlen = rcoarse.shape[0]
            std=np.pi/180*opts.rand_std * gain # decay
            rot_rand = torch.normal(torch.zeros(rlen,3),torch.ones(rlen,3)*std)
            rot_rand = rot_rand.clip(-2*std, 2*std)
            rot_rand = rot_rand.to(self.device)
            rot_rand = transforms.so3_exponential_map(rot_rand)
            rcoarse = rot_rand.matmul(rcoarse)

            # absorb current estimate to base rotation
            quat = transforms.matrix_to_quaternion(rcoarse).clone()
            self.model.module.nerf_root_rts.base_rt.se3.data[:,3:] = quat
            
            # absorb current estimate to base translation #TODO
            tmat = cam_pre[:,:3,3].clone()
            tmat[...,2] -= 0.3
            tmat = tmat*10 # to accound for *0.1 in expmlp
            self.model.module.nerf_root_rts.base_rt.se3.data[:,:3] = tmat 

            # re-init MLP weights
            self.model.module.nerf_root_rts.delta_rt[0].reinit(gain=1)
            self.model.module.nerf_root_rts.delta_rt[1].reinit(gain=1)

            # freeze weights of shape/ce in the following 1% iters
            self.model.module.counter_frz_randrt = 0.01

            # update near-far planes
            rtk_all = self.model.module.compute_rts()
            rtk_np = rtk_all.clone().detach().cpu().numpy()
            self.model.module.latest_vars['rtk'][:,:3] = rtk_np
            self.model.module.latest_vars['idk'][:] = 1
            self.model.module.near_far.data = get_near_far(
                                          self.model.module.near_far.data,
                                          self.model.module.latest_vars,
                                          tol_fac=1.2)

            self.init_training()

            # plot camera after randomaization
            cam_post =self.model.compute_rts()
            cam_mesh_pre, cam_mesh_post,line_mesh = draw_cams_pair(
                                           cam_pre.detach(). cpu().numpy(),
                                           cam_post.detach().cpu().numpy())
            cam_mesh_pre. export('%s/rand-%03d-pre.obj' %(self.save_dir,epoch))
            cam_mesh_post.export('%s/rand-%03d-post.obj'%(self.save_dir,epoch))
            line_mesh    .export('%s/rand-%03d-line.obj'%(self.save_dir,epoch))
        self.broadcast()

    def broadcast(self):
        """
        broadcast variables to other models
        """
        dist.barrier()
        if self.opts.lbs:
            #dist.broadcast_object_list(
            #        [self.model.module.num_bones,],
            #        0)
            dist.broadcast(self.model.module.bones,0)
            dist.broadcast(self.model.module.nerf_body_rts.linear_final.weight, 0)
            dist.broadcast(self.model.module.nerf_body_rts.linear_final.bias, 0)

        dist.broadcast(self.model.module.near_far,0)
   
    def clip_grad(self, aux_out):
        """
        gradient clipping
        """
        is_invalid_grad=False
        grad_nerf_coarse=[]
        grad_nerf_beta=[]
        grad_nerf_feat=[]
        grad_nerf_beta_feat=[]
        grad_nerf_fine=[]
        grad_nerf_unc=[]
        grad_phys_env=[]
        grad_trsi_2d=[]
        grad_trsi_2d_cam=[]
        grad_nerf_skin=[]
        grad_mlp_deform=[]
        grad_dfm_code=[]
        grad_nerf_vis=[]
        grad_nerf_root_rts=[]
        grad_nerf_body_rts=[]
        grad_root_code=[]
        grad_pose_code=[]
        grad_env_code=[]
        grad_vid_code=[]
        grad_bones=[]
        grad_sim3=[]
        grad_skin_aux=[]
        grad_ks=[]
        grad_nerf_dp=[]
        grad_csenet=[]
        grad_cnn_ff=[]
        grad_conv3d=[]
        grad_dpproj=[]
        for name,p in self.model.named_parameters():
            try: 
                pgrad_nan = p.grad.isnan()
                if pgrad_nan.sum()>0: 
                    print(name)
                    is_invalid_grad=True
            except: pass
            if 'nerf_coarse' in name and 'beta' not in name:
                grad_nerf_coarse.append(p)
            elif 'nerf_coarse' in name and 'beta' in name:
                grad_nerf_beta.append(p)
            elif 'nerf_feat' in name and 'beta' not in name and 'conv3d_net' not in name:
                grad_nerf_feat.append(p)
            elif 'nerf_feat' in name and 'beta' in name:
                grad_nerf_beta_feat.append(p)
            elif 'nerf_fine' in name:
                grad_nerf_fine.append(p)
            elif 'nerf_unc' in name:
                grad_nerf_unc.append(p)
            elif 'phys_env' in name:
                grad_phys_env.append(p)
            elif 'trsi_2d' in name and 'cam_mlp' not in name:
                grad_trsi_2d.append(p)
            elif 'trsi_2d' in name and 'cam_mlp' in name:
                grad_trsi_2d_cam.append(p)
            elif 'nerf_skin' in name:
                grad_nerf_skin.append(p)
            elif 'mlp_deform' in name:
                grad_mlp_deform.append(p)
            elif 'dfm_code' in name:
                grad_dfm_code.append(p)
            elif 'nerf_vis' in name:
                grad_nerf_vis.append(p)
            elif 'nerf_root_rts' in name:
                grad_nerf_root_rts.append(p)
            elif 'nerf_body_rts' in name:
                grad_nerf_body_rts.append(p)
            elif 'root_code' in name:
                grad_root_code.append(p)
            elif 'pose_code' in name or 'rest_pose_code' in name:
                grad_pose_code.append(p)
            elif 'env_code' in name:
                grad_env_code.append(p)
            elif 'vid_code' in name:
                grad_vid_code.append(p)
            elif 'module.bones' == name:
                grad_bones.append(p)
            elif 'module.sim3' == name:
                grad_sim3.append(p)
            elif 'module.skin_aux' == name:
                grad_skin_aux.append(p)
            elif 'module.ks_param' == name:
                grad_ks.append(p)
            elif 'nerf_dp' in name:
                grad_nerf_dp.append(p)
            elif 'csenet' in name:
                grad_csenet.append(p)
            elif 'cnn_ff' in name:
                grad_cnn_ff.append(p)
            elif 'conv3d_net' in name:
                grad_conv3d.append(p)
            elif 'dp_proj' in name:
                grad_dpproj.append(p)
            else: continue
        
        # freeze root pose when using re-projection loss only
        if self.model.module.root_update == 0:
            self.zero_grad_list(grad_ks)
            self.zero_grad_list(grad_root_code)
            self.zero_grad_list(grad_nerf_root_rts)
        if self.model.module.body_update == 0:
            self.zero_grad_list(grad_pose_code)
            self.zero_grad_list(grad_nerf_body_rts)
        if self.model.module.shape_update == 0:
            self.zero_grad_list(grad_nerf_coarse)
            self.zero_grad_list(grad_nerf_beta)
            self.zero_grad_list(grad_nerf_vis)
            self.zero_grad_list(grad_mlp_deform)
            self.zero_grad_list(grad_dfm_code)
        if self.model.module.skin_update == 0:
            self.zero_grad_list(grad_bones)
            self.zero_grad_list(grad_nerf_skin)
            self.zero_grad_list(grad_skin_aux)
            self.zero_grad_list(grad_sim3)
        if self.model.module.cvf_update == 0:
            self.zero_grad_list(grad_nerf_feat)
            self.zero_grad_list(grad_nerf_beta_feat)
            self.zero_grad_list(grad_csenet)
            self.zero_grad_list(grad_conv3d)
            self.zero_grad_list(grad_dpproj)
        if self.opts.freeze_bg:
            self.zero_grad_list(grad_trsi_2d)
        if self.opts.freeze_bgcam:
            self.zero_grad_list(grad_trsi_2d_cam)
        clip_scale=self.opts.clip_scale
 
        #TODO don't clip root pose
        aux_out['nerf_coarse_g']   = clip_grad_norm_(grad_nerf_coarse,    1*clip_scale)
        aux_out['nerf_beta_g']     = clip_grad_norm_(grad_nerf_beta,      1*clip_scale)
        aux_out['nerf_feat_g']     = clip_grad_norm_(grad_nerf_feat,     .1*clip_scale)
        aux_out['nerf_beta_feat_g']= clip_grad_norm_(grad_nerf_beta_feat,.1*clip_scale)
        aux_out['nerf_fine_g']     = clip_grad_norm_(grad_nerf_fine,     .1*clip_scale)
        aux_out['nerf_unc_g']     = clip_grad_norm_(grad_nerf_unc,       .1*clip_scale)
        aux_out['phys_env_g']     = clip_grad_norm_(grad_phys_env,       .1*clip_scale)
        aux_out['trsi_2d_g']      = clip_grad_norm_(grad_trsi_2d,        .1*clip_scale)
        aux_out['trsi_2d_cam_g']  = clip_grad_norm_(grad_trsi_2d_cam,        .1*clip_scale)
        aux_out['nerf_skin_g']     = clip_grad_norm_(grad_nerf_skin,     .1*clip_scale)
        aux_out['mlp_deform_g']   = clip_grad_norm_(grad_mlp_deform,     .1*clip_scale)
        aux_out['dfm_code_g']   = clip_grad_norm_(grad_dfm_code,         .1*clip_scale)
        aux_out['nerf_vis_g']      = clip_grad_norm_(grad_nerf_vis,      .1*clip_scale)
        aux_out['nerf_root_rts_g'] = clip_grad_norm_(grad_nerf_root_rts,100*clip_scale)
        aux_out['nerf_body_rts_g'] = clip_grad_norm_(grad_nerf_body_rts,100*clip_scale)
        aux_out['root_code_g']= clip_grad_norm_(grad_root_code,          .1*clip_scale)
        aux_out['pose_code_g']= clip_grad_norm_(grad_pose_code,         100*clip_scale)
        aux_out['env_code_g']      = clip_grad_norm_(grad_env_code,      .1*clip_scale)
        aux_out['vid_code_g']      = clip_grad_norm_(grad_vid_code,      .1*clip_scale)
        aux_out['bones_g']         = clip_grad_norm_(grad_bones,          1*clip_scale)
        aux_out['skin_aux_g']   = clip_grad_norm_(grad_skin_aux,         .1*clip_scale)
        aux_out['sim3']   = clip_grad_norm_(grad_sim3,         .1*clip_scale)
        aux_out['ks_g']            = clip_grad_norm_(grad_ks,            .1*clip_scale)
        aux_out['nerf_dp_g']       = clip_grad_norm_(grad_nerf_dp,       .1*clip_scale)
        aux_out['csenet_g']        = clip_grad_norm_(grad_csenet,        .1*clip_scale)
        aux_out['dpproj_g']        = clip_grad_norm_(grad_dpproj,        .1*clip_scale)
        aux_out['cnn_ff_g']        = clip_grad_norm_(grad_cnn_ff,       100*clip_scale)
        aux_out['conv3d_g']        = clip_grad_norm_(grad_conv3d,       100*clip_scale)

        #if aux_out['nerf_root_rts_g']>10:
        #    is_invalid_grad = True
        if is_invalid_grad:
            self.zero_grad_list(self.model.parameters())
            
    @staticmethod
    def find_nerf_coarse(nerf_model):
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
                # get the weights according to coarse posec
                # 63 = 3 + 60
                # 60 = (num_freqs, 2, 3)
                out_dim = p.shape[0]
                pos_dim = nerf_model.in_channels_xyz-nerf_model.in_channels_code
                # TODO
                num_coarse = 8 # out of 10
                #num_coarse = 10 # out of 10
                #num_coarse = 1 # out of 10
           #     p.grad[:,:3] = 0 # xyz
           #     p.grad[:,3:pos_dim].view(out_dim,-1,6)[:,:num_coarse] = 0 # xyz-coarse
                p.grad[:,pos_dim:] = 0 # others
            else:
                param_list.append(p)
        return param_list

    @staticmethod 
    def render_vid(model, batch):
        opts=model.opts
        model.set_input(batch)
        rtk = model.rtk
        kaug=model.kaug.clone()
        embedid=model.embedid
            
        if opts.train_cnn:
            # replace rot for eval visualization
            rtk_base = create_base_se3(rtk.shape[0], rtk.device)
            root_rts = model.cnn_ff(model.imgs)
            rtk[:,:3,:3] = refine_rt(rtk_base, root_rts)[:,:3,:3]
        
        rendered, _ = model.nerf_render(rtk, kaug, embedid, ndepth=opts.ndepth)
        if 'xyz_camera_vis' in rendered.keys():    del rendered['xyz_camera_vis']   
        if 'xyz_canonical_vis' in rendered.keys(): del rendered['xyz_canonical_vis']
        if 'pts_exp_vis' in rendered.keys():       del rendered['pts_exp_vis']      
        if 'pts_pred_vis' in rendered.keys():      del rendered['pts_pred_vis']     
        rendered_first = {}
        for k,v in rendered.items():
            if v.dim()>0: 
                bs=v.shape[0]
                rendered_first[k] = v[:bs//2] # remove loss term

        #if opts.train_cnn:
        #    dph_pred = F.interpolate(dph_pred[:bs//2], (model.img_size, model.img_size))
        #    depth_near = dph_pred.median() /3
        #    depth_far = dph_pred.median() *3
        #    dph_pred = (dph_pred - depth_near) / (depth_far - depth_near)
        #    rendered_first['dph_pred'] = dph_pred[:,0]
        return rendered_first 

    def save_logs(self, log, aux_output, total_steps, epoch):
        for k,v in aux_output.items():
            self.add_scalar(log, k, aux_output,total_steps)
        
    def add_image_grid(self, rendered_seq, log, epoch):
        for k,v in rendered_seq.items():
            grid_img = image_grid(rendered_seq[k],3,3)
            if k=='depth_rnd':scale=True
            elif k=='occ':scale=True
            elif k=='unc_pred':scale=True
            elif k=='proj_err':scale=True
            elif k=='feat_err':scale=True
            else: scale=False
            self.add_image(log, k, grid_img, epoch, scale=scale)

    def add_image(self, log,tag,timg,step,scale=True):
        """
        timg, h,w,x
        """

        if self.isflow(tag):
            timg = timg.detach().cpu().numpy()
            timg = flow_to_image(timg)
        elif scale:
            timg = (timg-timg.min())/(timg.max()-timg.min())
        else:
            timg = torch.clamp(timg, 0,1)
    
        if len(timg.shape)==2:
            formats='HW'
        elif timg.shape[0]==3:
            formats='CHW'
            print('error'); pdb.set_trace()
        else:
            formats='HWC'

        log.add_image(tag,timg,step,dataformats=formats)

    @staticmethod
    def add_scalar(log,tag,data,step):
        if tag in data.keys():
            log.add_scalar(tag,  data[tag], step)

    @staticmethod
    def add_mesh(log, tag, mesh, step):
        log.add_mesh(tag, mesh.vertices[None], 
                            colors=mesh.visual.vertex_colors[None][...,:3],
                            faces=mesh.faces[None], global_step=step)

    @staticmethod
    def del_key(states, key):
        if key in states.keys():
            del states[key]
    
    @staticmethod
    def isflow(tag):
        flolist = ['flo_coarse', 'fdp_coarse', 'flo', 'fdp', 'flo_at_samp']
        if tag in flolist:
           return True
        else:
            return False

    @staticmethod
    def zero_grad_list(paramlist):
        """
        Clears the gradients of all optimized :class:`torch.Tensor` 
        """
        for p in paramlist:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

