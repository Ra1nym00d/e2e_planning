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

 # 1.基础模块, 两次卷积, 然后跳跃连接相加
class BasicBlock(nn.Module):
    def __init__(self, in_channel):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channel),
        )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = F.relu(out + identity)
        return out

# 2.下采样模块, 第一次卷积缩小尺度扩大维度 第二次卷积不变, 跳跃连接使用1*1卷积缩小尺度扩大维度 与第二次卷积相加
class DownSample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DownSample, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channel),

        )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        identity = self.downsample(identity)
        out = F.relu(out + identity)
        return out


class Encoder(nn.Module):
    def __init__(self, in_channel=3, debug=False):
        super(Encoder, self).__init__()
        self.debug = debug

        # 最初的卷积和最大池化
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # 第一层
        self.layer1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            BasicBlock(64),
            # BasicBlock(64),
        )
        # 第三层
        self.layer2 = nn.Sequential(
            DownSample(64, 128),
            BasicBlock(128),
            # BasicBlock(128),
        )
        # 第四层
        self.layer3 = nn.Sequential(
            DownSample(128, 256),
            BasicBlock(256),
            # BasicBlock(256),
        )

        self.layer4 = nn.Sequential(
            DownSample(256, 512),
            BasicBlock(512),
            # BasicBlock(512),
            nn.Conv2d(512, 32, 1), #32 x8 x4
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.mu    = torch.nn.Linear(1024, 512)
        self.logsigma = torch.nn.Linear(1024, 512)

    def forward(self, x):
        x = self.conv1(x)
        if self.debug:
            print('layer0-------------', x.shape)

        x = self.layer1(x)
        if self.debug:
            print('layer1-------------', x.shape)

        x = self.layer2(x)
        if self.debug:
            print('layer2-------------', x.shape)

        x = self.layer3(x)
        if self.debug:
            print('layer3-------------', x.shape)

        x = self.layer4(x)
        if self.debug:
            print('layer4-------------', x.shape)

        # x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        # x = F.relu(self.fc1(x))
        # x = self.resnet_1d_1(x)
        # x = self.resnet_1d_2(x)
        return x, self.mu(x), self.logsigma(x)

"""
nn.Conv2d:
out_size = （in_size - K + 2P）/ S +1


nn.ConvTranspose2d
output = (input-1)stride+outputpadding -2padding+kernelsize
"""
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.upsample = nn.Sequential(
            nn.Linear(512, 512*4*8),
            nn.ReLU(),
            )

        self.generator = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, 2, 1, 1), 
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1), 
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 96, 5, 2, 2, 1), 
            nn.BatchNorm2d(96),
            nn.ReLU(),

            nn.ConvTranspose2d(96, 64, 5, 2, 2, 1), 
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 6, 5, 2, 2, 1), 
            # nn.Tanh(),
            nn.BatchNorm2d(6)
        )

    def forward(self, x):
        x = self.upsample(x)
        x = x.reshape(-1, 512, 4, 8)
        return self.generator(x)
    
    
img_datasets = glob.glob(f"big_imgs/*.png")
P_bs = 16


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

    bat_imgs = np.zeros((P_bs, 6,128,256), dtype=np.float32)
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
      img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV_I420)

      bat_imgs[count] = reshape_yuv(img)
      # cv2.imshow("test", img_bgr)
      # cv2.waitKey(1000)
      count +=1

    yield bat_imgs, len(sample_list) < P_bs



def main():
  # # self.load_model()

  train_steps = 0
  dvc = torch.device('mps')
  encoder = Encoder(6).to(dvc)
  #decoder = Generator().to(dvc)
  decoder = torch.load('out/decoder.pt')

  # import wandb
  # runner_w = wandb.init(project="e2e_demo")
  # wandb.config = {
  #   "learning_rate": 0.0001,
  #   "epochs": 200,
  #   "batch_size": 128

  # }
  # runner = wandb.Artifact( "e2e_demo", type="dataset", description="test artifact")

  # wandb.watch(encoder, log="gradients",  log_freq=10)
  # wandb.watch(decoder, log="gradients",  log_freq=10)
  
  #mdoel optiomizter
  vae_optim = torch.optim.Adam([{'params': encoder.parameters()},
                                {'params': decoder.parameters(),}], lr=2e-4, betas=(0.5, 0.999), weight_decay=1e-3)


  ref_loss_func = nn.SmoothL1Loss(reduction='none')

  # load dataset
  generator = datagen()
  total_step_one_epoch = int(len(img_datasets)/P_bs)

  for ep in range(100):
    print(f'************************** trainig: {ep+1} **************************\n')
    # deal with one epoch data loop
    kld_w = 1


    with tqdm(total=int(total_step_one_epoch), desc='training process: ', colour='GREEN') as pbar:

      while True:
        tx, is_not_valid = next(generator)

        # print(tx.shape, is_not_valid, total_step_one_epoch)

        if is_not_valid:
          print("this epoch data has been used done, will go to next epoch ***")
          break

        train_steps += 1
        ## two continious image
        X_in = torch.from_numpy(tx).to(dvc)/127.5 - 1
        # print(X_in.shape)

        ## first encoder it
        _, mu, log_var = encoder(X_in)

        std = torch.exp(0.5 * log_var)
        ## then sample gauss sdistribution
        # std = torch.exp(log_var/2)  # more smoother
        eps = torch.randn_like(mu)  #eps: bs,latent_size
        z_noise = mu + eps*std  #z: bs,latent_size

        ## then get generated image
        gen_img_out = decoder(z_noise)

        ## last we caculate loss
        reg_loss = (ref_loss_func(gen_img_out, X_in)).mean(dim=0).sum()
        kl_loss = ( -0.5 * (1 + log_var - mu*mu - log_var.exp())).mean(dim=0).sum()

        # print(reg_loss, kl_loss)
        # vae_loss = (reg_loss +  kl_loss * 512/(6*128*256)).mean()
        vae_loss = (reg_loss +  1*kl_loss).mean()

        # # update pbar
        pbar.update(1)
        pbar.set_description(f"epoch:{ep}, step:{train_steps}, loss:{vae_loss}, loss_kl:{kl_loss.mean()}, reg_loss:{reg_loss.mean()}")

        # ## first log loss
        # wandb.log({
        #           "miu": wandb.Histogram(mu.detach().cpu().numpy()),
        #           "std": wandb.Histogram(std.detach().cpu().numpy()),
        #           "vae_loss": vae_loss.cpu().detach().numpy(), 
        #           "reg_loss": reg_loss.mean().cpu().detach().numpy(), 
        #           "loss_KLD": kl_loss.mean().cpu().detach().numpy()})

        vae_optim.zero_grad()  # clear gradients for this training step
        vae_loss.backward()  # back propagation, compute gradients
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 10.0)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 10.0)
        vae_optim.step()

        ## save model with 100 steps
        if train_steps % 200==0:
          torch.save(encoder, 'out/encoder.pt')
          torch.save(decoder, 'out/decoder1.pt')
          z_noise = torch.randn(P_bs, 512).to(dvc)
          gen_img_bat = decoder(z_noise).reshape(P_bs, 6,128,256).cpu().detach().numpy()

          for j in range(P_bs):
            # print(gen_img_out.shape)
            gen_img_1 = gen_img_out[j,:,:,:].cpu().detach().numpy()
            gen_img_1 = np.clip((gen_img_1+1)*127.5, a_min=0, a_max = 255.)
            cv2.imwrite(f"./out/images_gen_vae/dec_img_{train_steps}_{j}.png", to_rgb(gen_img_1))

          for j in range(P_bs):
            gen_img_2 = gen_img_bat[j,:,:,:]
            gen_img_2 = np.clip((gen_img_2+1)*127.5, a_min=0, a_max = 255.)
            cv2.imwrite(f"./out/images_gen_vae/gen_img_{train_steps}_{j}.png", to_rgb(gen_img_2))


if __name__=='__main__':
  main()