import pdb
import time
import cv2
import numpy as np
import trimesh
import torch
from pytorch3d import transforms
from nnutils.geom_utils import vec_to_sim3, se3exp_to_vec, se3_vec2mat, se3_mat2rt,\
                                fid_reindex
from scipy.spatial.transform import Rotation as R

def robot2parent_idx(urdf):
    """
    get parent idx from urdf
    """
    ball_joint=urdf.ball_joint
    name2joints_idx = urdf.name2joints_idx
    parent_idx = [-1] + [0] * len(name2joints_idx.keys())
    for idx,link in enumerate(urdf._reverse_topo):
        path = urdf._paths_to_base[link]
        # potentially connected to root
        if len(path) == 2:
            joint = urdf._G.get_edge_data(path[0], path[1])['joint']
            if joint.name in name2joints_idx.keys():
                joint_idx = name2joints_idx[joint.name]
                parent_idx[joint_idx+1] = 0
            
        if len(path)>2:
            for jdx in range(len(path)-1):
                # find the current joint
                joint = urdf._G.get_edge_data(path[jdx], path[jdx+1])['joint']
                if joint.name in name2joints_idx.keys():
                    joint_idx = name2joints_idx[joint.name]
                    for kdx in range(jdx+1, len(path)-1):
                        # find the next joint
                        next_joint = urdf._G.get_edge_data(path[kdx], path[kdx+1])['joint']
                        if next_joint.name in name2joints_idx.keys():
                            next_joint_idx = name2joints_idx[next_joint.name]
                            parent_idx[joint_idx+1] = next_joint_idx+1
                            break
                    break
    
    #link_map = {}
    #for idx,link in enumerate(urdf._reverse_topo):
    #    link_map[link.name] = idx
    #parent_idx = []
    #for idx,link in enumerate(urdf._reverse_topo):
    #    path = urdf._paths_to_base[link]
        #if len(path)>1:
        #    if ball_joint and idx%3!=1:continue
        #    parent = path[1]
        #    if ball_joint:
        #        parent_idx.append( (link_map[parent.name]+2)//3 )
        #    else:
        #        parent_idx.append( link_map[parent.name] )
        #else:
        #    parent_idx.append(-1)
    return parent_idx

def get_joints(urdf,device="cpu"):
    """
    return joint locations wrt parent link
    a root joint of (0,0,0) is added
    joints: physical joints, B
    name2joints_idx, name to joint idx
    angle_names, registered angle predictions
    only called in nnutils/robot.py
    """
    ball_joint=urdf.ball_joint
    counter = 0
    name2joints_idx = {}
    name2query_idx = {}
    joints = []
    angle_names = []
    for idx,joint in enumerate(urdf.joints):
        if joint.joint_type == 'fixed':continue
        angle_names.append(joint.name)
        if ball_joint and idx%3!=2: continue
        name2query_idx[joint.name] = counter
        counter += 1
    counter = 0
    for idx,joint in enumerate(urdf.joints):
        if joint.joint_type == 'fixed':continue
        if ball_joint and idx%3!=0: continue
        # for ball joints, only the first has non zero center
        name2joints_idx[joint.name] = counter
        origin = torch.Tensor(joint.origin[:3,3]).to(device)
        joints.append(origin)
        counter += 1
    
    joints = torch.stack(joints,0)
    urdf.name2joints_idx = name2joints_idx # contain all physics joints
    urdf.name2query_idx = name2query_idx # contain all physics joints
    urdf.angle_names = angle_names # contain all dofs
    return joints 

def compute_bone_from_joint(model, is_init, vid=None):
    """
    vid, bs
        if none, use canonical bone length 
    Returns
        bones at rest pose
    """
    # compute bones as center of links
    urdf = model.robot.urdf
    device = model.device

    # get canonical sim3 or video specific se3
    sim3 = model.robot.sim3

    # get joint centers
    joints,angles = model.nerf_body_rts.forward_abs(vid=vid)
    joints = joints.view(1,-1,12)
    rmat = joints[:,:,:9].view(1,-1,3,3)
    tmat = joints[:,:,9:].view(1,-1,3,1)
    fk = torch.cat([rmat, tmat],-1)
    fk = fk[0].to(device)

    # update joint to link centers
    center=[]
    scale=[]
    orient=[]
    idx = 0
    for link in urdf._reverse_topo:
        path = urdf._paths_to_base[link]
        if len(path)>1:
            joint = urdf._G.get_edge_data(path[0], path[1])['joint']
            if joint.name not in urdf.name2query_idx:
                continue
        if len(link.visuals)>0:
            link_bounds = link.visuals[0].geometry.meshes[0].bounds
            # scale factor
            link_scale = torch.Tensor(link_bounds[1] - link_bounds[0]).to(device)
            link_scale = link_scale * 5
            link_scale = link_scale * sim3[7:].exp().to(device)

            # bone center
            link_corners =trimesh.bounds.corners(link_bounds)
            link_corners += link.visuals[0].origin[:3,3][None]
            link_corners = torch.Tensor(link_corners).to(device) * \
                           sim3[7:].exp().to(device)[None]
            link_corners = link_corners.matmul(fk[idx][:3,:3].T) + fk[idx][:3,3][None]
            link_center = link_corners.mean(0)
        else:
            link_scale = torch.Tensor([1,1,1]).to(device)*np.exp(-3.5) 
            link_center = fk[idx][:3,3]
        link_orient = fk[idx][:3,:3]
        link_orient = transforms.matrix_to_quaternion(link_orient)
        idx+=1

        center.append( link_center)
        orient.append( link_orient)
        scale. append( link_scale.log() )

    center = torch.stack(center,0)
    orient = torch.stack(orient,0)
    if is_init:
        scale =  torch.stack(scale, 0)
    else:
        scale = model.bones[:,7:10]
    bones =  torch.cat([center, orient, scale],-1)
    #from utils.io import save_bones
    #save_bones(bones.detach().cpu().numpy(), 0.1, 'tmp/0.obj')
    return bones

def angle_to_rts(robot, joints, angles, sim3):
    # forward kinematics
    # angles, -1, B
    bs = angles.shape[0]

    # current estimate
    joint_cfg = {}
    for idx, angle_name in enumerate(robot.angle_names):
        joint_cfg[angle_name] = angles[:,idx]
    fk = link_fk(robot, joints, joint_cfg, sim3)
    #fk0 = link_fk_v0(robot, joints, joint_cfg, sim3)
    #print((fk-fk0).abs().mean())
    return fk

def vis_joints(robot):
    # deprecated: origin is the transform from the parent link to the child link.
    pts = []
    for joint in robot.joints:
        pts.append(joint.origin[:3,3])
    pts = np.stack(pts,0)
    trimesh.Trimesh(pts).export('tmp/0.obj')

def link_fk(urdf, joints, joint_cfg, sim3):
    """
    pytorch version of link_fk function
    computes transformed coorrdinates given angles
    return -1,B,4,4 bones/links
    """
    # Compute forward kinematics in reverse topological order
    #torch.cuda.synchronize()
    #start_time = time.time()
    fk = []
    fk_dict = {}
    bs     = joint_cfg[list(joint_cfg.keys())[0]].shape[0]
    device = joint_cfg[list(joint_cfg.keys())[0]].device
    zero_mat = torch.zeros(bs,3).to(device)

    se3_log_list = []
    joint_center_list = []
    se3_log_kv = {}
    for idx,joint in enumerate(urdf.joints):
        # get joint angle
        if joint.name in joint_cfg.keys():
            angle = joint_cfg[joint.name].view(-1,1)
        else:
            angle = zero_mat
        axis = torch.Tensor(joint.axis).to(device).view(1,3)
        se3_log = torch.cat([zero_mat, angle*axis],1)
        se3_log_list.append(se3_log)
        se3_log_kv[joint.name] = idx

        # get joint center
        origin = torch.Tensor(joint.origin).to(device)
        #origin = torch.tensor(joint.origin.astype(np.float32), device=device)
        origin = origin.view(1,4,4).repeat(bs,1,1) # bs, k
        if joint.name in urdf.name2joints_idx.keys():
            origin[...,:3,3] = joints[:,urdf.name2joints_idx[joint.name]]
        joint_center_list.append(origin)

    se3_log_list = torch.stack(se3_log_list, 0).view(-1,6) # J,N,6
    joint_center_list = torch.stack(joint_center_list, 0) # J,N,4,4
    se3_exp_list = transforms.se3_exp_map(se3_log_list).view(-1,bs,4,4)
    se3_exp_list = se3_exp_list.permute(0,1,3,2) # to be compatible with urdfpy
    se3_exp_list = joint_center_list @ se3_exp_list

    #torch.cuda.synchronize()
    #print('(((fk1 time:%.6f'%(time.time()-start_time))

    pose_identity = torch.eye(4).to(device)[None].repeat(bs,1,1)
    for link in urdf._reverse_topo:
        path = urdf._paths_to_base[link] # path of links to root
        pose = pose_identity.clone()
        for i in range(len(path) - 1):
            child = path[i]
            parent = path[i + 1]
            joint = urdf._G.get_edge_data(child, parent)['joint']

            # get joint se3
            se3_exp = se3_exp_list[se3_log_kv[joint.name]]

            pose = se3_exp @ pose # N,Bx4x4

            # Check existing FK to see if we can exit early
            if parent in fk_dict:
                pose = fk_dict[parent].matmul(pose)
                break
        fk_dict[link] = pose
        # assign rtk to link
        if len(path)>1:
            parent_joint = urdf._G.get_edge_data(path[0], path[1])['joint']
            if parent_joint.name in urdf.name2query_idx.keys():
                fk.append( pose )
        else:
            fk.append( pose )
    #torch.cuda.synchronize()
    #print('(((fk2 time:%.6f'%(time.time()-start_time))
    fk = torch.stack(fk,1)

    # TODO write as sim3 transform
    # transform to banmo coordinate
    center, orient, scale = vec_to_sim3(sim3)

    fk[:,:,:3,3] *= scale[:,None] # bs
    se3 = torch.eye(4)[None].repeat(bs,1,1).to(device)
    se3[:,:3,:3] = orient # torch.Tensor([-1,0,0,0,0,1,0,1,0]).to(device).view(3,3)
    se3[:,:3, 3] = center # fk[:,:,1,3] += 0.1

    fk = se3[:,None].matmul(fk)
    return fk

# angles to config
def angles2cfg(robot, angles):
    cfg = {}
    for idx,name in enumerate(robot.angle_names):
        cfg[name] = angles[idx].cpu().numpy()
    return cfg

def visualize_joints(model, query_time=None, robot_save_path="tmp/robot.jpg"):
    """
    query_time: a scalar value
    This function is only used at eval mode
    """
    with torch.no_grad():
        joints,angles = model.nerf_body_rts.forward_abs(x=query_time)
    joints = joints.view(1,-1,12)
    angles = angles[0]

    # save to vec joints
    rmat = joints[:,:,:9].view(1,-1,3,3)
    tmat = joints[:,:,9:].view(1,-1,3,1)
    joints = torch.cat([rmat, tmat],-1)
    joints = se3exp_to_vec(joints[0])[None]

    cfg = angles2cfg(model.robot.urdf, angles)

    # save to robot renderings
    #robot_rendered,robot_mesh = render_robot(model.robot.urdf,
    #        robot_save_path,cfg=cfg, use_collision=False)
    robot_rendered  = np.zeros((256,256,3))
    robot_mesh = articulate_robot(model.robot.urdf, cfg=cfg, use_collision=False)

    # transform to canonical space
    sim3 = model.nerf_body_rts.sim3
    if query_time is not None:
        vid, _ = fid_reindex(query_time, model.num_vid, model.data_offset)
        vid = vid.view(-1)
        sim3 = sim3 + model.nerf_body_rts.sim3_vid[vid[0]]
    sim3 = sim3.detach().cpu()
    center, orient, scale = vec_to_sim3(sim3)
    robot_mesh.vertices = robot_mesh.vertices*scale[None].numpy()
    robot_mesh.vertices = robot_mesh.vertices@orient.T.numpy()
    robot_mesh.vertices = robot_mesh.vertices+center[None].numpy()

    # get kps by link center
    kp_dict = model.robot.urdf.link_fk(cfg=cfg) # body frame
    kp = []
    for kp_link, kp_T in kp_dict.items():
        if kp_link.name in model.robot.urdf.kp_links:
            kp.append(kp_T[:,3]) 
    kp = np.stack(kp,0) # k,4
    kp[:,:3] = kp[:,:3]*scale[None].numpy()
    kp[:,:3] = kp[:,:3]@orient.T.numpy()
    kp[:,:3] = kp[:,:3]+center[None].numpy()
    kp = kp.T # 4,k

    return joints, angles, robot_rendered, robot_mesh, kp

def query_kps(model, query_time):
    """
    query_time: T
    """
    with torch.no_grad():
        _,angles = model.nerf_body_rts.forward_abs(x=query_time)

    kps = []
    for angles_t in angles:
        cfg = angles2cfg(model.robot.urdf, angles_t)

        # get kps by link center
        kp_dict = model.robot.urdf.link_fk(cfg=cfg) # body frame
        kp = []
        for kp_link, kp_T in kp_dict.items():
            if kp_link.name in model.robot.urdf.kp_links:
                kp.append(kp_T[:,3]) 
        kp = np.stack(kp,0) # k,4
        kps.append( kp )
    kps = np.stack(kps, 0) # T,k,4

    # transform to canonical space
    sim3 = model.nerf_body_rts.sim3
    vid, _ = fid_reindex(query_time, model.num_vid, model.data_offset)
    vid = vid.view(-1)
    sim3 = sim3 + model.nerf_body_rts.sim3_vid[vid[0]]
    sim3 = sim3.detach().cpu()
    center, orient, scale = vec_to_sim3(sim3)

    kps[...,:3] = kps[...,:3]*scale[None,None].numpy()
    kps[...,:3] = kps[...,:3]@orient.T[None].numpy()
    kps[...,:3] = kps[...,:3]+center[None, None].numpy()
    kps = np.transpose(kps, [0,2,1]) # T,4,k
    return kps

def get_visual_origin(urdf):
    lfk = urdf.link_fk()
 
    rt = {}
    for link in lfk:
        for visual in link.visuals:
            if len(visual.geometry.meshes)>0:
                for mesh in visual.geometry.meshes:
                    rt[mesh] = visual.origin
    return rt

def get_collision_origin(urdf):
    lfk = urdf.link_fk()
 
    rt = {}
    for link in lfk:
        for visual in link.collisions:
            if len(visual.geometry.meshes)>0:
                for mesh in visual.geometry.meshes:
                    rt[mesh] = visual.origin
    return rt

def articulate_robot_rbrt_batch(robot, rbrt):
    """
    Note: this assumes rbrt is a torch tensor
    robot: urdfpy object
    rbrt: ...,13,7, first is the root pose instead of the base link
    returns a mesh of the articulated robot
    """
    ndim = rbrt.ndim
    device = rbrt.device
    fk = get_collision_origin(robot)

    # store a mesh
    verts_all=[]
    faces_single=[]
    face_base = 0
    count = 0
    for it,item in enumerate(fk.items()):
        if it not in robot.unique_body_idx:
            continue
        tm,pose = item
        pose = np.reshape(pose, (1,)*(ndim-2)+(4,4))
        pose = torch.Tensor(pose).to(device)

        pose = se3_vec2mat(rbrt[...,count,:]) @ pose
        #pose = se3_vec2mat(rbrt[...,it,:]) @ pose
        rmat, tmat = se3_mat2rt(pose) # ...,3,3
    
        verts = np.reshape(tm.vertices, (1,)*(ndim-2)+(-1,3,))
        verts = torch.Tensor(verts).to(device) # ...,3
        permute_tup = tuple(range(ndim-2))
        verts = verts @ torch.permute(rmat, permute_tup+(-1,-2)) + tmat[...,None,:]
        verts_all.append(verts)
        # add faces of each part        
        faces = tm.faces
        faces += face_base
        face_base += verts.shape[-2] 
        faces_single.append(faces)
        count += 1
    verts_all = torch.cat(verts_all, -2)
    faces_single = np.concatenate(faces_single, -2)
    return verts_all, faces_single

def articulate_robot_rbrt(robot, rbrt, gforce=None, com=None):
    """
    robot: urdfpy object
    rbrt: 13,7
    returns a mesh of the articulated robot
    """
    fk = get_collision_origin(robot)
    #fk = get_visual_origin(robot)

    # store a mesh
    meshes=[]
    count = 0
    for it,item in enumerate(fk.items()):
        if it not in robot.unique_body_idx:
            continue
        tm,pose = item
        pose = se3_vec2mat(rbrt[count]) @ pose
        #pose = se3_vec2mat(rbrt[it]) @ pose
        rmat, tmat = se3_mat2rt(pose)

        faces = tm.faces
        faces -= faces.min()
        tm = trimesh.Trimesh(tm.vertices, tm.faces) # ugly workaround for uv = []
        tm=tm.copy()
        tm.vertices = tm.vertices.dot(rmat.T) + tmat[None]
    
        # add arrow mesh
        if gforce is not None:
            force = gforce[count, 3:6].cpu().numpy()
            mag = np.linalg.norm(force, 2,-1)
            if mag>10: # a threshold
            #if mag>0:
                orn = force/mag
                orth1 = np.cross(orn, [0,0,1])
                orth2 = np.cross(orn, orth1)
                transform = np.eye(4)
                transform[:3,3] = tm.vertices.mean(0)
                transform[:3,2] = orn
                transform[:3,1] = orth1 / np.linalg.norm(orth1)
                transform[:3,0] = -orth2 / np.linalg.norm(orth2)

                arrow = get_arrow(mag, transform)

                tm = trimesh.util.concatenate([tm, arrow])
                tm.visual.vertex_colors[:,0] = 255
                tm.visual.vertex_colors[:,1] = 0
                tm.visual.vertex_colors[:,2] = 0
            else:
                tm.visual.vertex_colors[:,0] = 255
                tm.visual.vertex_colors[:,1] = 255
                tm.visual.vertex_colors[:,2] = 255

        meshes.append(tm)
        count += 1
    vertex_colors = np.concatenate([i.visual.vertex_colors for i in meshes],0)
    meshes = trimesh.util.concatenate(meshes)
    meshes.visual.vertex_colors = vertex_colors

    # com
    if com is not None:
        transform = np.eye(4)
        transform[:3,3] = com
        transform[:3,2] = [0,-1,0]
        transform[:3,1] = [1,0,0]
        transform[:3,0] = [0,0,-1]
        arrow = get_arrow(60, transform)
        arrow.visual.vertex_colors[:,0] = 0# np.clip(255*mag, 128, 255)
        arrow.visual.vertex_colors[:,1] = 255
        arrow.visual.vertex_colors[:,2] = 0
        meshes = trimesh.util.concatenate([meshes, arrow])

    return meshes

def get_arrow(mag, transform):
    mag = np.clip( mag/200, 0,1)
    box = trimesh.primitives.Box(extents=[0.05,0.05,1*mag])
    con = trimesh.creation.cone(0.05, 0.1)
    con.vertices[:,2] += 0.5 * mag
    arrow = trimesh.util.concatenate([box, con])
    arrow.vertices[:,2] += 0.5 * mag # 0,0,1 direction

    arrow.vertices = arrow.vertices @ transform[:3,:3].T + transform[:3,3][None]
    return arrow

def articulate_robot(robot, cfg=None, use_collision=False):
    """
    robot: urdfpy object
    returns a mesh of the articulated robot in its original scale
    """
    if cfg is not None and type(cfg) is not dict:
        cfg = torch.Tensor(cfg)
        cfg = angles2cfg(robot, cfg)
    if use_collision:
        fk = robot.collision_trimesh_fk(cfg=cfg)
    else:
        fk = robot.visual_trimesh_fk(cfg=cfg)

    # store a mesh
    meshes=[]
    for tm in fk:
        pose = fk[tm]
        faces = tm.faces
        faces -= faces.min()
        tm = trimesh.Trimesh(tm.vertices, tm.faces) # ugly workaround for uv = []
        tm=tm.copy()
        tm.vertices = tm.vertices.dot(pose[:3,:3].T) + pose[:3,3][None]
        meshes.append(tm)
    meshes = trimesh.util.concatenate(meshes)
    return meshes

def render_robot(robot, save_path, cfg=None, use_collision=False):
    """Visualize the URDF in a given configuration.
    modified from urdfpy
    Parameters
    ----------
    cfg : dict
        A map from joints or joint names to configuration values for
        each joint. If not specified, all joints are assumed to be
        in their default configurations.
    use_collision : bool
        If True, the collision geometry is visualized instead of
        the visual geometry.
    """
    import os
    os.environ["PYOPENGL_PLATFORM"] = "egl" # for offscreen rendering
    import pyrender  # Save pyrender import for here for CI

    meshes = articulate_robot(robot, cfg=cfg, use_collision=use_collision)

    # add mesh
    scene = pyrender.Scene(ambient_light=0.4*np.asarray([1.,1.,1.,1.]))
    scene.add(pyrender.Mesh.from_trimesh(meshes, smooth=False))
    #pyrender.Viewer(scene, use_raymond_lighting=True)
    # add camera and lighting
    cam = pyrender.OrthographicCamera(xmag=.5, ymag=.5)
    cam_pose = np.eye(4);
    cam_pose[:3,:3] = cv2.Rodrigues(np.asarray([0.,-np.pi/4*2,0.]))[0]
    cam_pose[0,3]=-1
    cam_node = scene.add(cam, pose=cam_pose)
    direc_l = pyrender.DirectionalLight(color=np.ones(3), intensity=6.0)
    light_pose = np.eye(4);
    light_pose[:3,:3] = cv2.Rodrigues(np.asarray([-np.pi/8*3.,-np.pi/8*5,0.]))[0]
    light_pose[0,3]=-1
    direc_l_node = scene.add(direc_l, pose=light_pose)

    r = pyrender.OffscreenRenderer(256,256)
    color,_ = r.render(scene,\
      flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL | \
            pyrender.RenderFlags.SKIP_CULL_FACES)
    r.delete()
    cv2.imwrite(save_path, color)
    return color, meshes
