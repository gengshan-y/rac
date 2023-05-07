"""
CUDA_VISIBLE_DEVICES=1 python explore.py --flagfile logdir/hmnerf-cate-ama-e120-b512-ft3/opts.log --nolineload --seqname ama-old
"""

import os
os.environ["PYOPENGL_PLATFORM"] = "egl" #opengl seems to only work with TPU

import time
from absl import flags, app
import sys
sys.path.insert(0,'third_party')
import numpy as np
import torch
import torchvision
import glob
import pdb
import cv2
import trimesh
from scipy.spatial.transform import Rotation as R
import imageio
import pyrender
from nnutils.train_utils import v2s_trainer
from nnutils.geom_utils import extract_mesh_simp, zero_to_rest_bone, \
                                zero_to_rest_dpose, skinning, lbs, se3exp_to_vec

from utils.io import save_vid, str_to_frame, save_bones, bones_to_mesh
opts = flags.FLAGS

flags.DEFINE_bool('interp_beta',False,'whether to interp code')
flags.DEFINE_bool('apply_lbs',True,'whether to apply lbs')
flags.DEFINE_integer('svid',69,'beta of the source mesh')
flags.DEFINE_integer('tvid',45,'beta of the target mesh')
flags.DEFINE_integer('tcap',-1,'cap number of target frames')
            
def get_center_crop(img_path, img_size=None):
    print(img_path)
    sil_path = img_path.replace('JPEGImages', 'Annotations').replace('.jpg', '.png')
    img = cv2.imread(img_path)[:,:,::-1].copy()
    sil = cv2.imread(sil_path,0)
    indices = np.where(sil>0); xid = indices[1]; yid = indices[0]
    center = ( int((xid.max()+xid.min())/2), int((yid.max()+yid.min())/2) )
    length = int(1.2*max(xid.max()-xid.min(), yid.max()-yid.min()))
    img = torchvision.transforms.functional.crop(torch.Tensor(img).permute(2,0,1), 
                    center[1]-length//2, center[0]-length//2, length, length)
    if img_size is not None:
        img = cv2.resize(img.permute(1,2,0).numpy(), (img_size, img_size))
    return img

def render_mesh(renderer, mesh_rest, canonical_rot, cam_offset, focal_fac, img_size):
    mesh_rest = mesh_rest.copy()
    mesh_rest.vertices = mesh_rest.vertices @ canonical_rot.T
    mesh_rest.vertices[:,1:] *= -1 
    mesh_rest.vertices += cam_offset[None] 
    scene = pyrender.Scene(ambient_light=0.4*np.asarray([1.,1.,1.,1.]))
    meshr = pyrender.Mesh.from_trimesh(mesh_rest,smooth=True)
    meshr._primitives[0].material.RoughnessFactor=.5
    scene.add_node( pyrender.Node(mesh=meshr ))

    focal = [focal_fac*img_size,focal_fac*img_size]
    ppoint = [img_size/2,img_size/2]
    cam = pyrender.IntrinsicsCamera(
            focal[0],
            focal[0],
            ppoint[0],
            ppoint[1],
            znear=1e-3,zfar=1000)
    cam_pose = -np.eye(4); cam_pose[0,0]=1; cam_pose[-1,-1]=1
    scene.add(cam, pose=cam_pose)
    
    direc_l = pyrender.DirectionalLight(color=np.ones(3), intensity=6.0)
    theta = 7*np.pi/9
    light_pose = np.asarray([[1,0,0,0],[0,np.cos(theta),-np.sin(theta),0],[0,np.sin(theta),np.cos(theta),0],[0,0,0,1]])
    direc_l_node = scene.add(direc_l, pose=light_pose)
    
    color, _ = renderer.render(scene,flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL | pyrender.RenderFlags.SKIP_CULL_FACES)
    color = color.copy()
    return color

def main(_):
    opts.model_path = 'logdir/%s/params_latest.pth'%opts.logname
    if opts.interp_beta:
        out_path = 'logdir/%s/explore-interp-%d'%(opts.logname,opts.svid)
    else:
        out_path = 'logdir/%s/explore-motion-%d'%(opts.logname,opts.svid)
    trainer = v2s_trainer(opts, is_eval=True)
    data_info = trainer.init_dataset()    
    trainer.define_model(data_info)
    seqname=opts.seqname

    model = trainer.model
    model.eval()

    # params
    img_size = 512
    grid_size= 256
    #grid_size= 64

    embedid_all = list(range(data_info['offset'][opts.tvid],data_info['offset'][opts.tvid+1]))
    if opts.tcap>0: embedid_all = embedid_all[:opts.tcap]
    #embedid = 100 # bouncing
    #embedid = 7401 # crouch
    #embedid = 10765+20 # cheetah 00
    #embedid = 10305+20 # sphinx 00

    #vidid_all = range(len(data_info['offset'])-1)
    #vidid_all = list(range(50, 60))
    vidid_all = [opts.svid]
    
    if opts.interp_beta:
        embedid_all = embedid_all[:1]
        vidid_all = vidid_all*30

    show_rest_pose = False
    #show_rest_pose = True # no articulation, no se3, but with bone stretching

    #show_deform = True # deformation model
    show_deform = False


    # dog
    canonical_rot = cv2.Rodrigues(np.asarray([0,np.pi/3,0]))[0]
    cam_offset = np.asarray([0,0,0.6])
    ## human
    #canonical_rot = cv2.Rodrigues(np.asarray([0,0,0]))[0]
    #cam_offset = np.asarray([0,0,0.6])
    ##cat
    #canonical_rot = cv2.Rodrigues(np.asarray([0,np.pi/6,0]))[0] # demo
    #cam_offset = np.asarray([0.02,-0.01,0.9])
    #canonical_rot = cv2.Rodrigues(np.asarray([0,np.pi/3,0]))[0]
    #cam_offset = np.asarray([0.04,-0.02,0.9])
    # quadruped
    #canonical_rot = cv2.Rodrigues(np.asarray([0,np.pi/3,0]))[0]
    #cam_offset = np.asarray([0,0,0.9])
    ##car
    #canonical_rot = cv2.Rodrigues(np.asarray([0,np.pi/2,0]))[0] @  cv2.Rodrigues(np.asarray([np.pi,0,0]))[0]
    #cam_offset = np.asarray([0.,0.,0.6])

    focal_fac = 4
    
    # render shape
    frames = []
    renderer = pyrender.OffscreenRenderer(img_size, img_size)
    with torch.no_grad():

        if opts.interp_beta:
            # randomly sample code beta
            beta_code = model.nerf_coarse.vid_code.weight
            #dis_mat = (beta_code[:,None] - beta_code[None]).norm(2,-1)
            #id1,id2 = torch.where(dis_mat.max() == dis_mat)[0]
            id1,id2 = opts.tvid, vidid_all[0]
            beta1 = beta_code[id1]
            beta2 = beta_code[id2]
            #beta_mean = beta_code.mean(0)
            #beta_std = beta_code.std(0)
            #beta1 = torch.normal(mean = beta_mean, std = beta_std)
            #beta2 = torch.normal(mean = beta_mean, std = beta_std)
            ts = torch.linspace(0,1,len(vidid_all), device=model.device)
            betas = beta1.lerp(beta2, ts[:,None])

            jlen_scale1 = model.nerf_body_rts.jlen_scale_z+model.nerf_body_rts.jlen_scale[id1]
            jlen_scale2 = model.nerf_body_rts.jlen_scale_z+model.nerf_body_rts.jlen_scale[id2]
            jlen_scale = jlen_scale1.lerp(jlen_scale2, ts[:,None])[:,None]

            sim3a = model.nerf_body_rts.sim3+model.nerf_body_rts.sim3_vid[id1]
            sim3b = model.nerf_body_rts.sim3+model.nerf_body_rts.sim3_vid[id2]
            sim3b[:7] = sim3a[:7]
            sim3 = sim3a.lerp(sim3b, ts[:,None])[:,None]
        else:
            betas = [None]*len(vidid_all)
            jlen_scale = [None]*len(vidid_all)
            sim3 = model.nerf_body_rts.sim3+model.nerf_body_rts.sim3_vid[vidid_all[0]]
            sim3[:7] = (model.nerf_body_rts.sim3+model.nerf_body_rts.sim3_vid[opts.tvid])[:7]
            sim3 = sim3[None,None].repeat(len(vidid_all), 1,1)

        for it,vidid in enumerate(vidid_all):
            # get image
            img_path = data_info['impath'][model.data_offset[vidid]]
            img_cropped = get_center_crop(img_path, img_size=img_size//4)

            # get mesh
            mesh_rest = extract_mesh_simp(model, opts.chunk, \
                               grid_size, 0, vidid=vidid, is_eval=True, beta=betas[it])
            num_pts = mesh_rest.vertices.shape[0]

            # stretch + lbs: mesh to mesh
            query_vidid = vidid*torch.ones(1).long().to(model.device)

            for jt,embedid in enumerate(embedid_all):
                print(embedid)
                # get target image
                img_path = data_info['impath'][embedid]
                target_img = get_center_crop(img_path, img_size=img_size//4)
                
                pts_can=torch.Tensor(mesh_rest.vertices).to(model.device)
                pts_can = pts_can[:,None]
                query_time = torch.ones(1,1).long().to(model.device)*embedid
                if opts.apply_lbs:
                    # soft deform
                    if show_deform and opts.mlp_deform:
                        # N,1,C vs N,d,1,3
                        dfm_code = model.dfm_code(query_time).repeat(num_pts,1)[:,None]
                        pts_can = model.mlp_deform.forward(dfm_code, 
                                                    pts_can[:,:,None])[:,:,0]

                    # zero-to-rest
                    bones_rst, bone_rts_rst = zero_to_rest_bone(model, model.bones)

                    # stretch + fk
                    if show_rest_pose:
                        query_code = torch.zeros(1,opts.t_embed_dim).to(model.device)
                    else:
                        query_code = model.nerf_body_rts.pose_code(query_time)
                    bone_rts_fw = model.nerf_body_rts.forward_decode(query_code, 
                                            query_vidid, show_rest_pose=show_rest_pose,
                     jlen_scale_in=jlen_scale[it], sim3_in=sim3[it])
                    bone_rts_fw = zero_to_rest_dpose(opts, bone_rts_fw, bone_rts_rst)
                    bone_rts_fw = bone_rts_fw.repeat(num_pts, 1,1)

                    # skin
                    rest_pose_code = model.rest_pose_code(torch.Tensor([0]).long().to(model.device))
                    skin_forward,_ = skinning(pts_can, model.embedding_xyz, bones_rst, 
                                        rest_pose_code, model.nerf_skin)

                    # dqs
                    pts_dfm,bones_dfm = lbs(bones_rst, bone_rts_fw, skin_forward, 
                            pts_can,backward=False)
                else:
                    pts_dfm = pts_can

                # re-create mesh
                mesh_dfm = trimesh.Trimesh( pts_dfm[:,0].cpu().numpy(), mesh_rest.faces, 
                                          vertex_colors=mesh_rest.visual.vertex_colors)

                # get joint vis
                joints,_ = model.nerf_body_rts.forward_abs(x=query_time,vid=query_vidid, 
                      show_rest_pose=show_rest_pose, 
                     jlen_scale_in=jlen_scale[it], sim3_in=sim3[it])
                joints = joints.view(1,-1,12)
                joints = torch.cat([joints[:,:,:9].view(1,-1,3,3), 
                                    joints[:,:,9:].view(1,-1,3,1)],-1)
                joints = se3exp_to_vec(joints[0]).cpu().numpy()
                joint_mesh = bones_to_mesh(joints, 0.1, parent=model.robot.urdf.parent_idx)

                # render
                j_col = render_mesh(renderer, joint_mesh, canonical_rot, cam_offset, focal_fac, img_size)
                color = render_mesh(renderer, mesh_dfm, canonical_rot, cam_offset, focal_fac, img_size)
                mesh_dfm.visual.vertex_colors[:,:3] = 128
                col_g = render_mesh(renderer, mesh_dfm, canonical_rot, cam_offset, focal_fac, img_size)
                #TODO linblend
                color = lin_brush(col_g, color, jt/len(embedid_all))
                color[:img_cropped.shape[0], :img_cropped.shape[1]] = img_cropped
                j_col[:target_img.shape[0],  :target_img.shape[1] ] = target_img
                color = np.concatenate([j_col,color],1)
                frames.append(color)

            cv2.imwrite('tmp/color-%02d.jpg'%vidid, color[:,:,::-1])
            mesh_dfm.export('tmp/mesh-%02d.obj'%vidid)
        save_vid(out_path, frames, suffix='.mp4',upsample_frame=0)


def lin_brush(col_g, color, t):
    t = np.sin(t*np.pi)
    imgw = col_g.shape[1]
    wpx = int(t*imgw)
    img_new = np.zeros_like(color)
    img_new[:,:wpx] = color[:,:wpx]
    img_new[:,wpx:] = col_g[:,wpx:]

    return img_new
if __name__ == '__main__':
    app.run(main)
