#!/usr/bin/python
import sys
import os
sys.path.append(os.getcwd())
import cv2
import numpy as np
import glob 
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
from tqdm import tqdm
import random

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait, ALL_COMPLETED
import threading
import multiprocessing

from common.transformations.camera import get_view_frame_from_road_frame
import utils_comma2k19.coordinates as coord

import laika.lib.orientation as orient
from utils import get_poses, reshape_yuv, get_hevc_np, get_calib_extrinsic
from utils import MEDMDL_INSMATRIX, C3_INSMATRIX, C2_INSMATRIX, \
                  ANCHOR_TIME, TRAJECTORY_TIME, LABEL_LEN
from models.resnet18 import PlanningModel, MTPLoss
import utils_comma2k19.orientation as orient1
from utils import Simulator, draw_path

#from dataset.dataset import DatasetGeneratorV2
from h5_generator import client_generator
from check_label import SegDataset
from utils import valid_segment_slice


from torch.utils.tensorboard import SummaryWriter  
writer = SummaryWriter('./runs/e2e')  # 日志存储路径  


cv2.ocl.setUseOpenCL(True)

with open("train_config.json", 'r') as f:
	conf = json.load(f)

## parameters
P_dvc = torch.device('cuda')
P_optim_per_n_steps = conf['optim_per_steps']
P_seq_length = conf['sequence_len']
P_bs = conf['batch_sz']
P_train_datadir = conf['training_datadir']
P_train_epoch = conf['epoch']
P_lr = conf['lr']
P_usewandb = conf['use_wandb']
P_mtp_alpha = conf['mtp_alpha']
P_grad_clip = conf['grad_clip']

# init wandb
if P_usewandb:
  import wandb
  runner_w = wandb.init(project="e2e_demo")
  wandb.config = {
    "learning_rate": 0.0001,
    "epochs": 200,
    "batch_size": 128

  }
  runner = wandb.Artifact( "e2e_demo", type="dataset", description="test artifact")
  

datadir = "dataset/realdata/training/"
## sequese model and loss
if conf['retrain']:
  plan_model = torch.load('out/planning_model.pt')
  print("load pre-trained weight!")

  ## if the we get good pretrain traj, we started MTP loss tringing.
  # with torch.no_grad():
  #   # initial weights
  #   for name, m in plan_model.named_modules():
  #     # print(name=='plan_head.6', type(m))
  #     # if isinstance(m, nn.Conv2d):
  #     #     nn.init.kaiming_normal_(m.weight, mode="fan_out")
  #     #     if m.bias is not None:
  #     #         nn.init.zeros_(m.bias)
  #     # elif isinstance(m, nn.BatchNorm2d):
  #     #     nn.init.ones_(m.weight)
  #     #     nn.init.zeros_(m.bias)
  #     # elif isinstance(m, nn.Linear) and name=='plan_head.6':
  #     #     nn.init.normal_(m.weight, 0, 0.1)
  #     #     nn.init.zeros_(m.bias)
  #     # elif isinstance(m, nn.Linear):
  #     #     nn.init.normal_(m.weight, 0, 0.01)
  #     #     nn.init.zeros_(m.bias)
  #     if isinstance(m, nn.Linear) and name=='plan_head.2':
  #       m.weight[5+33*3:5+33*3*2, :] = m.weight[5:5+33*3*1]
  #       m.weight[5+33*3*2:5+33*3*3, :] = m.weight[5:5+33*3*1]
  #       m.weight[5+33*3*3:5+33*3*4, :] = m.weight[5:5+33*3*1]
  #       m.weight[5+33*3*4:5+33*3*5, :] = m.weight[5:5+33*3*1]

else:
  plan_model = PlanningModel().to(P_dvc)
## log model
if P_usewandb:
  wandb.watch(plan_model, log="gradients",  log_freq=10)



e2e_optim = torch.optim.SGD(
    [ {"params": plan_model.enc.parameters(), "lr": P_lr,  "momentum": 0.9, "weight_decay": 1e-3},
      {"params": plan_model.plan_head.parameters(), "lr": 5*P_lr,  "momentum": 0.85, "weight_decay": 1e-3},
      {"params": plan_model.context_gru.parameters(), "lr": 2*P_lr,  "momentum": 0.9, "weight_decay": 1e-3},
      {"params": plan_model.pose_head.parameters(), "lr": P_lr,  "momentum": 0.9, "weight_decay": 1e-3},
      {"params": plan_model.feat_head_res.parameters(), "lr": P_lr,  "momentum": 0.9, "weight_decay": 1e-3},
])


# e2e_optim = torch.optim.SGD(plan_model.parameters(), lr=P_lr,  momentum=0.85, weight_decay=1e-3)
# # e2e_optim = torch.optim.Adam(plan_model.parameters(), lr=P_lr, weight_decay=1e-2)  # optimize all cnn parameters
# scheduler = torch.optim.lr_scheduler.StepLR(e2e_optim, step_size=5, gamma=0.8, last_epoch=-1, verbose=False)

# AdamW + Warmup + Cosine 衰减（工业级方案）
mtp_loss = MTPLoss()
# e2e_optim = torch.optim.AdamW(
#     plan_model.parameters(),
#     lr=P_lr,               # 初始学习率
#     weight_decay=0.01,      # 权重衰减强度
#     betas=(0.85, 0.999),    # 动量参数
#     eps=1e-8               # 数值稳定性
# )
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(e2e_optim, T_max=100)  # 假设总epoch=100


video_names = glob.glob(f"{datadir}/*/*/*/fcamera.hevc") + glob.glob(f"{datadir}/*/*/*/video.hevc")
train_step = 0
total_step_one_epoch = len(video_names)*P_seq_length/(P_bs*P_optim_per_n_steps)

def warn_cosin_learningRate(epoch, warmup_epochs, optimizer, scheduler):
    if epoch < warmup_epochs:
        lr = P_lr * (epoch + 1) / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        scheduler.step()

def traing_simulator(batch_ds, pbar, ep, seq_len=900, bat_sz=4):

  global train_step
  # every batch data include segments with 1 minites
  train_loss = 0
  loss_cls = 0
  loss_reg = 0  
  loss_x = 0 
  loss_y = 0  
  loss_angle = 0
  loss_vx = 0
  loss_wz = 0

  ## reset with every new mini batch  ,init last image
  hidden_st = torch.zeros((bat_sz, 512), dtype=torch.float32).to(P_dvc)#.cuda() 
  # feat_buff = torch.zeros((bat_sz, 20, 64), dtype=torch.float32).to(P_dvc)
  last_bat_img = None

  
  st = time.time()
  for frame_cnt in range(seq_len):
    feed_imgs = np.zeros((bat_sz, 3, 128, 256), dtype=np.uint8)
    label_traj = np.zeros((bat_sz, 33,3), dtype=np.float32)
    label_pose = np.zeros((bat_sz, 32), dtype=np.float32)
    # gpu_img = cv2.cuda_GpuMat()

    ## get devialation with simulator, generate feed img and label traj
    st1 = time.time() 
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
      #img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV_I420)
      #feed_imgs[i] = reshape_yuv(img)
      
      img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
      img_rgb = cv2.resize(img_rgb, (256, 128), interpolation=cv2.INTER_LINEAR)
        
      ##print(img_rgb.shape)
      feed_imgs[i] = img_rgb.transpose(2,0,1)

        
      label_traj_xyz =  batch_ds[i].get_label_traj_with_deviate(frame_cnt, pos, -1*theta)
      label_traj[i] = label_traj_xyz
      
      # xyz velocity
      label_pose[i,0:3] = batch_ds[i].local_velocity[frame_cnt]

      # xyz angular speed, rad 2 deg
      label_pose[i,3:6] = (180./np.pi) * batch_ds[i].frame_angular_velocities[frame_cnt]

    
    if frame_cnt == 0:
      last_bat_img = torch.from_numpy(feed_imgs).to(P_dvc)#.cuda()
      continue
    
    ## mdoel update, and simulator update
    input_bev = torch.from_numpy(np.ascontiguousarray(feed_imgs)).to(P_dvc)#.cuda()
    gt_traj = torch.from_numpy(label_traj).to(P_dvc)#.cuda()
    gt_pose = torch.from_numpy(label_pose).to(P_dvc)#.cuda()
   
    # use last image and current image to feed model, scalar to [-1 1]
    X_in = torch.cat([last_bat_img, input_bev], dim=1).float()
    
    out_preds = plan_model(X_in, hidden_st)

    # print('image gen cost time is : ', P_optim_per_n_steps*(time.time() - st1))

    feat_vec = out_preds[:, 0:512]
    hidden_st = out_preds[:, 512:1024]
    preds_buffer = out_preds[:, 1024:]
   
    # feat_buff = feat_buff.roll(shifts=-1, dims=1)
    # feat_buff[:, -1, :]   = next_feat.clone().detach()    
    last_bat_img = input_bev.clone()
    
    # caculate valid object loss, order is loss_obj, loss_reg_obj,  loss_cls_obj, 
    # loss_x_obj, loss_y_obj, loss_v_obj, loss_heading_obj  
    ret_loss, best_traj = mtp_loss.forward(P_mtp_alpha, preds_buffer, gt_traj, gt_pose, feat_vec, ep)

    # sum loss
    train_loss    += ret_loss[0]/P_optim_per_n_steps
    loss_reg      += ret_loss[1]/P_optim_per_n_steps
    loss_cls      += ret_loss[2]/P_optim_per_n_steps
    loss_x        += ret_loss[3]/P_optim_per_n_steps
    loss_y        += ret_loss[4]/P_optim_per_n_steps
    loss_angle    += ret_loss[5]/P_optim_per_n_steps

    loss_vx       += ret_loss[6]/P_optim_per_n_steps
    loss_wz       += ret_loss[7]/P_optim_per_n_steps
    
    
    # ToDO, first not close cloop training, after we get a open-loop better model
    ## feedback vehicle control model, to update heading and position bias
    for i in range(bat_sz):
      best_traj_np = best_traj[i].cpu().detach().numpy()
      pred_traj_x = best_traj_np[:,0]
      pred_traj_y = best_traj_np[:,1] 
      # after 2 epoch ,we get a relative good init plan, we start close loop
      if ep > 5:
        batch_ds[i].simulator.update(True, batch_ds[i].local_velocity[frame_cnt][0], -1*batch_ds[i].frame_angular_velocities[frame_cnt][2], 
                            pred_traj_x, -1*pred_traj_y, ANCHOR_TIME)
      
      ## for debug plot
      # gt_traj[i][:,2] = 0
      # best_traj_np[:,2] = 0
      # best_traj_np[:,2:3] = np.ones((33,1))
      # draw_path(gt_traj[i].cpu().detach().numpy(), img_bgr, width=0.7, height=1.22, fill_color=(255/2.,0,255))
      # draw_path(best_traj_np[:,0:3], img_bgr, width=0.6, height=1.22, fill_color=(255,255,255))
      # cv2.imshow("test", img_bgr)
      # cv2.waitKey(1)
    
    ## update model loss, backward
    # every optim_per_n_steps, we optimize model parameters
    if (frame_cnt+1) % P_optim_per_n_steps == 0:
      train_step+=1
      ## first log loss
      # if P_usewandb and train_step % 20 ==0:
        # wandb.log({"loss_train":      train_loss.cpu().detach().numpy(), 
        #             "reg_loss_train": loss_reg.cpu().detach().numpy(), 
        #             "cls_loss_train": loss_cls.cpu().detach().numpy(),  
        #             "x_loss":         loss_x.cpu().detach().numpy(), 
        #             "y_loss":         loss_y.cpu().detach().numpy(), 
        #             "loss_angle":     loss_angle.cpu().detach().numpy(),
        #             "vx_loss":        loss_vx.cpu().detach().numpy(), 
        #             "wz_loss":        loss_wz.cpu().detach().numpy(), 
        #             "lr":             e2e_optim.param_groups[0]['lr'] } )
      if train_step % 20 ==0:
        writer.add_scalar("loss_train", train_loss.cpu().detach().numpy(), train_step//20.)
        writer.add_scalar("reg_loss_train", loss_reg.cpu().detach().numpy(), train_step//20.)
        writer.add_scalar("cls_loss_train", loss_cls.cpu().detach().numpy(), train_step//20.)
        writer.add_scalar("x_loss", loss_x.cpu().detach().numpy(), train_step//20.)
        writer.add_scalar("y_loss", loss_y.cpu().detach().numpy(), train_step//20.)
        writer.add_scalar("loss_angle", loss_angle.cpu().detach().numpy(), train_step//20.)
        writer.add_scalar("vx_loss", loss_vx.cpu().detach().numpy(), train_step//20.)
        writer.add_scalar("wz_loss", loss_wz.cpu().detach().numpy(), train_step//20.)
        writer.add_scalar("lr", e2e_optim.param_groups[0]['lr'], train_step//20.)

      if train_step % 300 ==0:
        for name, param in plan_model.named_parameters():  
            # 记录权重直方图（按层名分类：conv1/weight）  
            writer.add_histogram(f'{name}/weight', param, train_step // 200)  
            # 记录梯度直方图（需在backward后获取）  
            writer.add_histogram(f'{name}/grad', param.grad, train_step // 200)  

        
      # update pbar
      pbar.update(1)
      pbar.set_description(f"epoch:{ep:.2f}, step:{train_step:.2f}, loss:{train_loss:.2f}, loss_cls:{loss_cls:.2f},loss_x:{loss_x:.2f}, loss_y:{loss_y:.2f},loss_vx:{loss_vx:.2f}, loss_angle:{loss_angle:.2f}")
                              
      ## then backward optimize
      hidden_st = hidden_st.clone().detach()
      e2e_optim.zero_grad()  # clear gradients for this training step
      train_loss.backward()  # back propagation, compute gradients
      torch.nn.utils.clip_grad_norm_(parameters=plan_model.parameters(), max_norm= P_grad_clip)
      e2e_optim.step()
      
      ## reset loss
      train_loss = 0
      loss_cls = 0
      loss_reg = 0  
      loss_x = 0 
      loss_y = 0  
      loss_angle = 0
      loss_vx = 0
      loss_wz = 0
      ## save model with 100 steps
      if train_step % 120 ==0:
        torch.save(plan_model, 'out/planning_model.pt')
    
  # if loss conter is not integral multipy of optim_per_n_steps, just proceed optimizer step
  if not isinstance(train_loss, int):
    e2e_optim.zero_grad()  # clear gradients for this training step
    train_loss.backward()  # back propagation, compute gradients
    torch.nn.utils.clip_grad_norm_(parameters=plan_model.parameters(), max_norm= P_grad_clip)
    e2e_optim.step()


## training loop
for ep in range(P_train_epoch):

  # first update lr
  warn_cosin_learningRate(ep, 5, e2e_optim, scheduler)
    
  files_length = len(video_names)
  sample_list = [i for i in range(files_length)]
  print(f'************************** trainig: {ep+1} **************************\n')
  pbar = tqdm(total=int(total_step_one_epoch), desc='training process: ', colour='WHITE')

  while True:
    # print(len(sample_list))
    if len(sample_list) < P_bs:
      break
    valid_count = 0
    # print(sample_list, random.sample(sample_list, 1))
    gt_paths = []
    batch_ds =  []
    
    # can not get P_bs ds or datasets length is zero, break the loop
    while valid_count < P_bs and len(sample_list) > 0:
      # get one segments
      file_index = random.sample(sample_list, 1)[0]
      sample_list.remove(file_index)

      # get names
      video_name = video_names[file_index]
      tmp1 = video_name.split('/')

      tmp1.pop()
      seg_path = '/'.join(tmp1)
      gt_paths.append(seg_path)

      st1 = time.time()
      dd = SegDataset(seg_path, seq_len=P_seq_length)

      print(time.time() - st1)
      if dd.data_valid:
        batch_ds.append(dd)
        valid_count +=1
      else:
        print("seg dataset is not valid, go to next sample!")
        continue
    
    # print(gt_paths)
    if valid_count == P_bs:
      traing_simulator(batch_ds, pbar, ep, seq_len=P_seq_length, bat_sz=P_bs)  
    else:
      print(" valid counter is less then P_bs")
   