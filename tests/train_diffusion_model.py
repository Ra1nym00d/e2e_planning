#!/usr/bin/env python3
import sys
import os
sys.path.append(os.getcwd())
import glob
import numpy as np
import random
import cv2
from tqdm import tqdm

from common.transformations.camera import get_view_frame_from_road_frame

import torch.nn as nn
import torch
import torch.nn.functional as F

    
img_datasets = glob.glob(f"big_imgs/*.png")
P_bs = 32


def reshape_yuv(frames):
  H = (frames.shape[0]*2)//3
  W = frames.shape[1]
  in_img1 = np.zeros((6, H//2, W//2), dtype=np.uint8)

  in_img1[0] = frames[0:H:2, 0::2]
  in_img1[1] = frames[1:H:2, 0::2]
  in_img1[2] = frames[0:H:2, 1::2]
  in_img1[3] = frames[1:H:2, 1::2]
  in_img1[4] = frames[H:H+H//4].reshape((H//2, W//2))
  in_img1[5] = frames[H+H//4:H+H//2].reshape((H//2, W//2))
  return in_img1

def to_rgb(img):
    H = 256
    W = 512

    frames = np.zeros((H*3//2, W), dtype=np.uint8)
    frames[0:H:2, 0::2] = img[0]
    frames[1:H:2, 0::2] = img[1]
    frames[0:H:2, 1::2] = img[2]
    frames[1:H:2, 1::2] = img[3]
    frames[H:H+H//4] = img[4].reshape((-1, H//4, W))
    frames[H+H//4:H+H//2] = img[5].reshape((-1, H//4, W))

    return cv2.cvtColor(frames, cv2.COLOR_YUV2BGR_I420)


def get_calib_matrix(is_c2=False):

  if is_c2:
    rot_angle =  [-0.0, 0.05, -0.0, 1.22, -0.]
    cam_insmatrixs = np.array([[910.0,  0.0,   0.5 * 1164],
                            [0.0,  910.0,   0.5 * 874],
                            [0.0,  0.0,     1.0]])
  else:

    rot_angle =  [-0.0005339412952404624, 0.0434282135994484, -0.02439150642603636, 1.35, -0.06]
    cam_insmatrixs = np.array([[2648.0,   0.,   1928/2.],
                            [0.,  2648.0,  1208/2.],
                            [0.,    0.,     1.]])

  ang_x = rot_angle[0]
  ang_y = rot_angle[1]
  ang_z = rot_angle[2] + np.clip(np.random.normal(loc=0.0, scale=0.005), -0.02, 0.02)

  dev_height = rot_angle[3]
  lat_bias = rot_angle[4] + np.clip(np.random.normal(loc=0.0, scale=0.1), -0.4, 0.4)
    

  MEDMDL_INSMATRIX = np.array([[910.0,  0.0,   0.5 * 512],
                                  [0.0,  910.0,   47.6],
                                  [0.0,  0.0,     1.0]])
  camera_frame_from_ground = np.dot(cam_insmatrixs,
                                      get_view_frame_from_road_frame(ang_x, ang_y, ang_z, dev_height, lat_bias))[:, (0, 1, 3)]
  calib_frame_from_ground = np.dot(MEDMDL_INSMATRIX,
                                      get_view_frame_from_road_frame(0, 0, 0, 1.22))[:, (0, 1, 3)]
  calib_msg = np.dot(camera_frame_from_ground, np.linalg.inv(calib_frame_from_ground))


  return calib_msg


def datagen():
  files_length = len(img_datasets)
  sample_list = [i for i in range(files_length)]

  while True:

    if len(sample_list) < P_bs:
      print(len(sample_list))
      sample_list = [i for i in range(files_length)]

    bat_imgs = np.zeros((P_bs, 3, 64, 128), dtype=np.float32)
    count = 0
    while count < P_bs:
      # get one segments
      file_index = random.sample(sample_list, 1)[0]
      sample_list.remove(file_index)

      # get names
      img_name = img_datasets[file_index]

      img = cv2.imread(img_name)

      is_c2 = img_name.split("/")[1][0]=='1'

      # print(img_name.split("/")[1][0])
      new_calib_matrix = get_calib_matrix(is_c2)
      img_bgr = cv2.warpPerspective(src=img, M=new_calib_matrix, dsize=(512,256), flags=cv2.WARP_INVERSE_MAP)

      img_bgr = img_bgr[:,::-1,:] if np.random.rand() > 0.5 else  img_bgr # random rever image with u axis
      img_bgr = cv2.resize(img_bgr, [ 128, 64], interpolation = cv2.INTER_AREA)


      # img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV_I420)

      bat_imgs[count] = img_bgr.transpose(2,0,1) #reshape_yuv(img)
      # cv2.imshow("test", img_bgr)
      # cv2.waitKey(1000)
      count +=1

    yield bat_imgs, len(sample_list) < P_bs

from Train import generate_diffussion_target
from Train import sample
from diff_model import Diffusion
from Loss import DiffLoss
from torchvision.utils import save_image


batch_size = 256
epoches = 10

# Actually train the model
ema_decay = 0.999

# The number of timesteps to use when sampling
steps = 500

# The amount of noise to add each timestep when sampling
eta = 1.

# Classifier-free guidance scale (0 is unconditional, 1 is conditional)
guidance_scale = 2.


def main():
  # # self.load_model()

  train_steps = 0
  device = torch.device('mps')

  model = Diffusion().to(device)

  #mdoel optiomizter
  diffusion_optim = torch.optim.Adam([{'params': model.parameters(),}], lr=4e-4)

  scheduler = torch.optim.lr_scheduler.StepLR(diffusion_optim, step_size=5, gamma=0.8, last_epoch=-1, verbose=False)


  # load dataset
  generator = datagen()
  total_step_one_epoch = int(len(img_datasets)/P_bs)

  for ep in range(100):
    print(f'************************** trainig: {ep+1} **************************\n')
    # deal with one epoch data loop
    with tqdm(total=int(total_step_one_epoch), desc='training process: ', colour='GREEN') as pbar:

        while True:
            tx, is_not_valid = next(generator)

            # print(tx.shape, is_not_valid, total_step_one_epoch)
            if is_not_valid:
              print("this epoch data has been used done, will go to next epoch ***")
              break

            train_steps += 1
            ## two continious image
            images = torch.from_numpy(tx).to(device)/127.5 - 1
            labels = torch.zeros(P_bs).long()

            src_images, src_labels = images.to(device), labels.to(device)

            t, noised_src, src_recon_targets = generate_diffussion_target(src_images, src_labels)

            diffusion_optim.zero_grad()

            to_drop = torch.rand(src_labels.shape, device=src_labels.device).le(0.2)
            classes_drop = torch.where(to_drop, -torch.ones_like(src_labels), src_labels)

            output, _, _, _ = model(noised_src, t, classes_drop)
            loss = F.mse_loss(output, src_recon_targets)
            loss.backward()
            diffusion_optim.step()
            #update pbar
            pbar.update(1)
            pbar.set_description(f"epoch:{ep}, step:{train_steps}, loss:{loss}")

            if train_steps % 500 == 0:
                torch.save(model, 'out/diffusion_model.pt')
                noise = torch.randn([10, 3, 64, 128], device=device)
                fakes_classes = torch.zeros(10).long().to(device)
                fakes = sample(model, noise, steps, eta, fakes_classes, guidance_scale)
                fakes = (fakes + 1) / 2
                fakes = torch.clamp(fakes, min=0, max = 1)
                save_image(fakes.data, './output/%03d_train.png' % ep)
    scheduler.step() # update parameters scheduler 


if __name__=='__main__':
  main()