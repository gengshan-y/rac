from __future__ import print_function
import sys
sys.path.insert(0,'../')
import cv2
import pdb
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import time
from flowutils.io import mkdir_p
from flowutils.util_flow import write_flow, save_pfm
from flowutils.flowlib import point_vec
from flowutils.dydepth import warp_flow
import glob
from auto_gen import flow_inference
cudnn.benchmark = False

parser = argparse.ArgumentParser(description='VCN+expansion')
parser.add_argument('--datapath', default='/ssd/kitti_scene/training/',
                    help='dataset path')
parser.add_argument('--loadmodel', default=None,
                    help='model path')
parser.add_argument('--testres', type=float, default=1,
                    help='resolution')
parser.add_argument('--maxdisp', type=int ,default=256,
                    help='maxium disparity. Only affect the coarsest cost volume size')
parser.add_argument('--fac', type=float ,default=1,
                    help='controls the shape of search grid. Only affect the coarse cost volume size')
parser.add_argument('--dframe', type=int ,default=1,
                    help='how many frames to skip')
parser.add_argument('--flow_threshold', type=float ,default=0.05,
                    help='flow threshold that controls frame skipping')
args = parser.parse_args()


mean_L = [[0.33,0.33,0.33]]
mean_R = [[0.33,0.33,0.33]]

# construct model, VCN-expansion
from models.VCNplus import VCN
from models.VCNplus import WarpModule, flow_reg
model = VCN([1, 256, 256], md=[int(4*(args.maxdisp/256)),4,4,4,4], fac=args.fac)
model = nn.DataParallel(model, device_ids=[0])
model.cuda()

if args.loadmodel is not None:
    pretrained_dict = torch.load(args.loadmodel)
    mean_L=pretrained_dict['mean_L']
    mean_R=pretrained_dict['mean_R']
    pretrained_dict['state_dict'] =  {k:v for k,v in pretrained_dict['state_dict'].items()}
    model.load_state_dict(pretrained_dict['state_dict'],strict=False)
else:
    print('dry run')
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

seqname = args.datapath.strip().split('/')[-3]
dframe = args.dframe

mkdir_p('../../tmp/%s/images_flt/'%seqname)

test_left_img = sorted(glob.glob('%s/*'%(args.datapath)))

def main():
    model.eval()
    inx=0;jnx=1
    while True:
        print('%s/%s'%(test_left_img[inx],test_left_img[jnx]))
        imgL_o = cv2.imread(test_left_img[inx])[:,:,::-1]
        imgR_o = cv2.imread(test_left_img[jnx])[:,:,::-1]
        flowfw, occfw = flow_inference(imgL_o, imgR_o, max_res=600*600)
        height,width,_ = imgL_o.shape

        flowfw_normed = np.concatenate( [flowfw[:,:,:1]/width, flowfw[:,:,1:2]/height],-1 )
        medflow = np.max(np.linalg.norm(flowfw_normed,2,-1))
        print('%.3f'%(medflow))
       
        if medflow > args.flow_threshold:
            if inx==0:
                cv2.imwrite('../../tmp/%s/images_flt/%05d.jpg'% (seqname,inx), imgL_o[:,:,::-1])
            cv2.imwrite('../../tmp/%s/images_flt/%05d.jpg'% (seqname,jnx), imgR_o[:,:,::-1])
            inx=jnx
        jnx+=1

if __name__ == '__main__':
    main()

