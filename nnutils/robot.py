import numpy as np
import pdb
import torch
import torch.nn as nn
from pytorch3d import transforms
import trimesh
from urdfpy import URDF
from nnutils.urdf_utils import get_joints, robot2parent_idx, compute_bone_from_joint

class URDFRobot(nn.Module):
    def __init__(self, urdf_path):
        super(URDFRobot, self).__init__()
        self.urdf = URDF.load(urdf_path)
        robot_name = urdf_path.split('/')[-1][:-5]
        self.urdf.robot_name = robot_name
        if robot_name=='wolf' or robot_name=='human' or robot_name=='human_amp'\
            or robot_name=='wolf_mod' or robot_name=='human_mod':
            self.urdf.ball_joint = True # whether has sperical joints
        else: self.urdf.ball_joint = False
        joints = get_joints(self.urdf) # joints: joint location, for training
        self.urdf.parent_idx = robot2parent_idx(self.urdf) # visualization only

        # order of operations: scale => rotate => translate
        if robot_name=='a1':
            sim3 = torch.Tensor([0,0,0, \
                                0.5,-0.5,-0.5,-0.5, \
                                -1.61,-1.61,-1.61]) # center, orient, scale
            self.num_dofs = joints.shape[0]
            rest_angles = torch.zeros(1,joints.shape[0])
            rest_angles[0,2] = -.8
            rest_angles[0,5] = -.8
            rest_angles[0,8] = -.8
            rest_angles[0,11]= -.8
        elif robot_name=='laikago':
            sim3 = torch.Tensor([0,0,0, \
                                1,0,0,0, \
                                -1.61,-1.61,-1.61]) # center, orient, scale
            self.num_dofs = joints.shape[0]
            rest_angles = torch.zeros(1,joints.shape[0])
            rest_angles[0,2] = -.8
            rest_angles[0,5] = -.8
            rest_angles[0,8] = -.8
            rest_angles[0,11]= -.8
        elif robot_name=='laikago_toes_zup_joint_order' or \
             robot_name=='laikago_mod':
            sim3 = torch.Tensor([0,0,0, \
                                0.5,-0.5,-0.5,-0.5, \
                                -1.61,-1.61,-1.61]) # center, orient, scale
            self.num_dofs = joints.shape[0]
            rest_angles = torch.zeros(1,joints.shape[0])
            rest_angles[0,2] = -.8
            rest_angles[0,5] = -.8
            rest_angles[0,8] = -.8
            rest_angles[0,11]= -.8
        elif robot_name=='wolf':
            sim3 = torch.Tensor([0,0.02,-0.02, \
                                0.5,0.6,0,0, \
                                -2.2,-2.2,-2.2]) # center, orient, scale
            self.num_dofs = joints.shape[0]*3
            rest_angles = torch.zeros(1,self.num_dofs) # ball joints
            self.urdf.kp_links = ['link_155_Vorderpfote_R_Y', 'link_150_Vorderpfote_L_Y', 
                             'link_170_Pfote2_R_Y', 'link_165_Pfote2_L_Y']
        elif robot_name=='wolf_mod':
            sim3 = torch.Tensor([0,0.01,-0.04, \
                                0.5,0.6,0,0, \
                                -3.1,-3.1,-3.1]) # center, orient, scale
            self.num_dofs = joints.shape[0]*3
            rest_angles = torch.zeros(1,self.num_dofs) # ball joints
            self.urdf.kp_links = ['link_155_Vorderpfote_R_Y', 'link_150_Vorderpfote_L_Y', 
                             'link_170_Pfote2_R_Y', 'link_165_Pfote2_L_Y']
        elif robot_name=='human':
            sim3 = torch.Tensor([0,0,0, \
                                1,0,0,0, \
                                -2.5,-2.5,-2.5]) # center, orient, scale
            self.num_dofs = joints.shape[0]*3
            rest_angles = torch.zeros(1,self.num_dofs) # ball joints
            self.urdf.kp_links = ['link_24_mixamorig:RightFoot_Y', 'link_19_mixamorig:LeftFoot_Y']
        elif robot_name=='human_mod':
            sim3 = torch.Tensor([0,0,0, \
                                1,0,0,0, \
                                -3.2,-3.2,-3.2]) # center, orient, scale
            self.num_dofs = joints.shape[0]*3
            rest_angles = torch.zeros(1,self.num_dofs) # ball joints
            self.urdf.kp_links = ['link_24_mixamorig:RightFoot_Y', 'link_19_mixamorig:LeftFoot_Y']
        elif robot_name=='human_amp':
            sim3 = torch.Tensor([0,0,0, \
                                1,0,0,0, \
                                -2.5,-2.5,-2.5]) # center, orient, scale
            self.num_dofs = joints.shape[0]*3
            rest_angles = torch.zeros(1,self.num_dofs) # ball joints

        self.sim3 = sim3# [:8] # 10 => 8 
        #self.sim3 = sim3[:8] # 10 => 8 
        self.joints = joints
        self.rest_angles = rest_angles
        self.num_bones = len(self.joints)+1

        # map from body to unique bodies
        unique_body_idx = list(range(len(self.urdf.links)))
        if self.urdf.ball_joint:
            unique_body_idx = unique_body_idx[0:1] + unique_body_idx[3::3]
        self.urdf.unique_body_idx = unique_body_idx
