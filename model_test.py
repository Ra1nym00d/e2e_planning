import os
import numpy as np
import cv2
import time
from common.transformations.camera import get_view_frame_from_road_frame
import utils_comma2k19.coordinates as coord

import laika.lib.orientation as orient
from utils import get_poses, reshape_yuv, get_hevc_np, get_calib_extrinsic, ComModel, PlanModel
from utils import MEDMDL_INSMATRIX, C3_INSMATRIX, C2_INSMATRIX, \
                  ANCHOR_TIME, TRAJECTORY_TIME, LABEL_LEN

import torch
from models.resnet18 import PlanningModel, MTPLoss
import utils_comma2k19.orientation as orient1
from utils import Simulator, draw_path
import matplotlib.pylab as plt
from check_label import SegDataset
import argparse
# import mplfinance as plt


# parameters
parser = argparse.ArgumentParser(description='e2e_demo model test .')
parser.add_argument('--closeloop', action='store_true')
parser.add_argument('--batsz', type=int, default=6)
args = parser.parse_args()


def get_model_input_and_label(plan_model, seg_path, seq_len):
  bat_sz=1
  batch_ds = []

  fig = plt.figure(figsize=( 12, 4) , dpi=80)
  output = cv2.VideoWriter('e2e_mdl_test.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 20, (960, 320))

  # cm_model = ComModel()
  for i in range(bat_sz):
    batch_ds.append(SegDataset(seg_path, seq_len=seq_len) )
    if not batch_ds[i].data_valid:
      print("seg dataset is not valid, go to next batch!")
      return  None
  
  for frame_cnt in range(seq_len):
    feed_imgs = np.zeros((bat_sz, 6, 128, 256), dtype=np.uint8)
    label_traj = np.zeros((bat_sz, 33,3), dtype=np.float32)
    label_pose = np.zeros((bat_sz, 32), dtype=np.float32)

    ## get devialation with simulator, generate feed img and label traj
    for i in range(bat_sz):
      # feedback position and heading error
      pos = batch_ds[i].simulator.error_pos
      theta = batch_ds[i].simulator.error_heading
      
      # image and label
      img_raw = batch_ds[i].next_image()
      if img_raw is None:
        print("can not read image from segment")
        return
      
      new_calib_matrix = batch_ds[i].get_calib_matrix(pos, theta)
      img_bgr = cv2.warpPerspective(src=img_raw, M=new_calib_matrix, dsize=(512,256), flags=cv2.WARP_INVERSE_MAP)
      img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV_I420)
      feed_imgs[i] = reshape_yuv(img)
      # img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
      # img_rgb = cv2.resize(img_rgb, (256, 128), interpolation=cv2.INTER_LINEAR)
        
      # ##print(img_rgb.shape)
      # feed_imgs[i] = img_rgb.transpose(2,0,1)

      label_traj_xyz =  batch_ds[i].get_label_traj_with_deviate(frame_cnt, pos, -1*theta)
      label_traj[i] = label_traj_xyz
      
      # xyz velocity
      label_pose[i,0:3] = batch_ds[i].local_velocity[frame_cnt]

      print("gt velocity ", label_pose[0,0] )
      # xyz angular speed, rad 2 deg
      label_pose[i,3:6] = (180./np.pi) * batch_ds[i].frame_angular_velocities[frame_cnt]

    # use last image and current image to feed model, scalar to [0 1]
    # big_imgs = np.concatenate([feed_imgs, last_bat_img], axis=1)
    
    st = time.time()
    traj_batch = plan_model.run(feed_imgs.astype(np.float32))


    ## figure axis setting
    spec = fig.add_gridspec(1, 2)  # H, W
    ax1 = fig.add_subplot(spec[ 0,  0])
    ax2 = fig.add_subplot(spec[ 0, 1])
  
    ## draw pred multi-trajectory
    best_idx = 0
    best_prob = 0.0
    idx = 0
    for traj in traj_batch:
      print("traj[0]", traj[0])
      ax1.plot(traj[2], traj[1], marker='o', c='r', linewidth='2', label='pred prob-{:.2f}'.format(traj[0]), alpha=max(traj[0], 0.1))
      ax1.fill_betweenx(traj[1], traj[2] + traj[4], traj[2] - traj[4], color='green', alpha=traj[0]) 
      if traj[0] > best_prob:
        best_idx = idx
        best_prob = traj[0]
      idx +=1
    # best_idx = 0
    ax1.legend(loc='upper left')
    ax1.plot(label_traj[0,:,1], label_traj[0,:,0], marker='o', c='b', linewidth='2')
    ax1.set_ylim([-0, 120])
    ax1.set_xlim([-8, 8])

    device_path = np.concatenate([traj_batch[best_idx][1].reshape(1,33), traj_batch[best_idx][2].reshape(1,33), np.zeros((1,33))], axis=0)
    batch_ds[0].draw_path(device_path.T, img_bgr, width=1, height=1.22)

    # gt_path = np.concatenate(label_traj_xyz.T, axis=0)
    batch_ds[0].draw_path(label_traj_xyz, img_bgr, width=1, height=1.22, fill_color=(122,122,122))
    ax2.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) )

    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8,)
    img = np.clip(img, 0, 255)
    print(img.shape)
    out_img  = img.reshape((320, 960) + (3,))

    if frame_cnt < 800:
      # cv2.imwrite(f"./images/img_{frame_cnt}.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR) )
      # print(out_img.shape)
      output.write(cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
      print(f'frame_{frame_cnt} has been saved!')
      fig.savefig(f"./images/img_{frame_cnt}.jpg") 
    fig.clf()

    # closed loop
    if args.closeloop:
      batch_ds[i].simulator.update(True, batch_ds[i].local_velocity[frame_cnt][0], -1*batch_ds[i].frame_angular_velocities[frame_cnt][2], 
                                traj_batch[best_idx][1], -1*traj_batch[best_idx][2], ANCHOR_TIME)
# load model and set eval mode
test_path = "./dataset/realdata/Chunk_cm7/99c94dc769b5d96e_2018-07-20--16-07-46/11"
# test_path = "/media/michael/a944f99c-b5cc-44fb-9ac8-d0b263759080/home/michael/Desktop/lcq__work/end2end_lateral/dataset/realdata/training/zs_c3_data/08131/2023-08-13--15-41-23--35" #32
#os.system("py model_convert.py")
plan_model = PlanModel(cuda=False)
get_model_input_and_label(plan_model, test_path, 900)
