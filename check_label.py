
import cv2
import os
import numpy as np
import subprocess
import time
import matplotlib.pyplot as plt
from tools.lib.logreader import LogReader
import laika.lib.coordinates as coord
import laika.lib.orientation as orient
from common.transformations.camera import get_view_frame_from_road_frame

cv2.ocl.setUseOpenCL(True) 

def read_poses(segment_dir, logfile='rlog'):
  logs = LogReader(os.path.join(segment_dir, logfile))
  kalman_msgs = [m.liveLocationKalman for m in logs if m.which() == 'liveLocationKalman']
  poses = {
      'positions_ecef': np.array([m.positionECEF.value for m in kalman_msgs]),
      # 'velocities_calib': np.array([m.velocityCalibrated.value for m in kalman_msgs]),
      'velocities_ecef': np.array([m.velocityECEF.value for m in kalman_msgs]),
      'accelerations_calib': np.array([m.accelerationCalibrated.value for m in kalman_msgs]),
      # 'accelerations_device': np.array([m.accelerationDevice.value for m in kalman_msgs]),
      'orientations_calib': np.array([m.calibratedOrientationECEF.value for m in kalman_msgs]),
      'orientations_ecef': np.array([m.orientationECEF.value for m in kalman_msgs]),
      'angular_velocities_calib': np.array([m.angularVelocityCalibrated.value for m in kalman_msgs]),
      # 'angular_velocities_device': np.array([m.angularVelocityDevice.value for m in kalman_msgs]),
      # 'times': np.array([m.unixTimestampMillis for m in kalman_msgs]),
  }
  if len(poses['orientations_ecef']) > 0:
      poses['orientations_euler'] = orient.euler2quat(poses['orientations_ecef'])

  status = {
      'positions': np.array([m.positionECEF.valid for m in kalman_msgs]),
      # 'velocities': np.array([m.velocityECEF.valid for m in kalman_msgs]),
      # 'accelerations': np.array([m.accelerationCalibrated.valid for m in kalman_msgs]),
      'orientations_calib': np.array([m.calibratedOrientationECEF.valid for m in kalman_msgs]),
      # 'angular_velocities_calib': np.array([m.angularVelocityCalibrated.valid for m in kalman_msgs]),
      'status': np.array([m.status for m in kalman_msgs]),
      'inputsOK': np.array([m.inputsOK for m in kalman_msgs]),
      'posenetOK': np.array([m.posenetOK for m in kalman_msgs]),
      'gpsOK': np.array([m.gpsOK for m in kalman_msgs]),
      'sensorsOK': np.array([m.sensorsOK for m in kalman_msgs]),
      # 'deviceStable': np.array([m.deviceStable for m in kalman_msgs]),
      # 'excessiveResets': np.array([m.excessiveResets for m in kalman_msgs]),
  }
  calib_msgs = [m.liveCalibration for m in logs if m.which() == 'liveCalibration']
  # model_msgs = [m.modelV2 for m in logs if m.which() == 'modelV2']
  return poses, status, np.array([m.rpyCalib for m in calib_msgs]), None




def get_poses(segment_path, kind='liveKalman'):
  if kind == 'liveKalman':
      poses, statuses , calib_msgs, model_msgs = read_poses(segment_path)
      frame_positions = poses['positions_ecef']
      frame_orientations_ecef = poses['orientations_calib']
      frame_velocities = poses['velocities_ecef'] 
      frame_orientations = orient.euler2quat(frame_orientations_ecef)
      frame_angular_velocities = poses['angular_velocities_calib']
  elif kind == 'comma2k19':
      frame_positions = np.load(example_segment + 'global_pose/frame_positions')
      frame_orientations = np.load(example_segment + 'global_pose/frame_orientations')
  elif kind == 'laika':
      raise NotImplementedError('laika')
  
  return frame_positions, frame_orientations, frame_velocities, calib_msgs, statuses, frame_angular_velocities, model_msgs


def get_calib_extrinsic(segment_dir, logfile='rlog'):
  st1 = time.time()
  logs = LogReader(os.path.join(segment_dir, logfile))
  print(time.time() - st1)

  calib_msgs = [m.liveCalibration for m in logs if m.which() == 'liveCalibration']

  # kalman_msgs = [m.liveLocationKalman for m in logs if m.which() == 'liveLocationKalman']
  # print(kalman_msgs)

  # trans_msgs = np.array([m.velocityCalibrated.value for m in kalman_msgs])
  # rot_msgs = np.array([m.angularVelocityCalibrated.value for m in kalman_msgs])

  # print([m.which() for m in logs])

  extrinsic_matrix = np.array(calib_msgs[0].extrinsicMatrix).reshape((3,4))
  # model_msgs = [m.modelV2 for m in logs if m.which() == 'modelV2']

  return extrinsic_matrix, None

def get_vector_angle(a, b):
  # 夹角cos值
  cos_ = np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
  # 夹角sin值
  sin_ = np.cross(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
  arctan2_ = np.arctan2(sin_, cos_)
  # print(arctan2_*180/np.pi)
  return arctan2_


from control.lat_mpc import LatMpc
import math



# test_seg = "/media/chaoqun/datadisk_2TB/08_end2end/e2e_lateral_planning/dataset/realdata/training/Chunk_cm6/99c94dc769b5d96e_2018-07-11--15-13-12/14/"
# get_calib_extrinsic(test_seg,  logfile='raw_log.bz2') 


class Simulator():
  def __init__(self, X, Y, Heading):
    self.out = 0.0
    self.controller = LatMpc()
    self.dy = 0.1
    self.dheading = 0.0
    self.dt = 0.05

    # self.fig = plt.figure(figsize=( 12, 9) , dpi=80)
    # plt.ion()

    self.g_heading = -0.00
    self.g_y = 0.0
    self.cnt = 0

    self.X = X
    self.Y = Y
    self.Heading = Heading

    self.error_pos = np.clip(np.random.normal(loc=0.0, scale=0.3), -0.6, 0.6)
    self.error_heading = 0.0

    self.cmds = [0 for i in range(10)]

    # self.fig = plt.figure(figsize=( 12, 9) , dpi=80)
    # plt.ion()

    self.pos_err_limit = 0.6
    self.heading_err_limit = 0.1



  def update(self, is_active, velocity, omiga, traj_x, traj_y, traj_t) -> float:
    pred_dx = np.clip(traj_x[1:33] - traj_x[0:32], 1e-1, 1e4) # avoid zero divide
    pred_dy = traj_y[1:33] - traj_y[0:32]
    traj_theta = np.array([math.atan2(y, x) for x,y in zip(pred_dx.flatten(), pred_dy.flatten())]) # rad

    # print("traj_theta", traj_theta[0])
    if self.cnt == 0:
      is_active = False
    yawrate_cmd = self.controller.update(is_active, velocity, omiga*180/np.pi, traj_x.flatten(), traj_y.flatten(), traj_theta.flatten(), traj_t.flatten())

    self.cmds[0:-1] = self.cmds[1:]
    self.cmds[-1] = yawrate_cmd

    # print(yawrate_cmd*180/np.pi)
    # # use velocity and wheel angle to get next frame offset
    # psi_ego = traj_theta[0]
    # delta_y = (velocity * np.sin(psi_ego)) * self.dt

    # motion model
    # delta_y = 0.5*dx*yawrate_cmd/velocity
    # delta_heading = yawrate_cmd * self.dt
    # delta_y = delta_heading * velocity

    self.g_heading += self.cmds[0] * self.dt
    dx = self.X[self.cnt+1] -  self.X[self.cnt]
    dy = self.Y[self.cnt+1] -  self.Y[self.cnt]
    ds = np.sqrt(dx*dx + dy*dy)

    self.g_y += np.sin(self.g_heading) *ds#* velocity * self.dt

    self.error_pos = self.Y[self.cnt] - self.g_y
    self.error_heading = self.Heading[self.cnt] - self.g_heading 

    self.error_pos = min(self.error_pos, self.pos_err_limit) if self.error_pos>0 \
                else max(self.error_pos, -self.pos_err_limit)
    self.error_heading = min(self.error_heading, self.heading_err_limit) if self.error_heading>0 \
                else max(self.error_heading, -self.heading_err_limit)

    # print(self.cnt, self.error_pos, self.g_y, self.Y[self.cnt], self.g_heading, self.Heading[self.cnt] , yawrate_cmd)

    # # need add coordinate transform
    # traj_points = np.concatenate((traj_x.reshape(33,1), traj_y.reshape(33,1), np.ones((33,1))), axis=1)
    # theta_bias = -1*self.g_heading

    # ## with comma2k dataset, gt path is rotate with yaw angle, tested!
    # transform_maxtrix = np.array([[np.cos(theta_bias), np.sin(theta_bias), self.X[self.cnt]], 
    #                               [-np.sin(theta_bias), np.cos(theta_bias), self.g_y],
    #                               [0,0,1]])

    # label_traj_xyz = (transform_maxtrix @ traj_points.T).T

    # traj_points1 = np.concatenate((np.array([0, 10,250]).reshape(3,1), np.array([0,0,0]).reshape(3,1), np.array([1,1,1]).reshape(3,1)), axis=1)
    # theta_bias = -1*self.g_heading

    # ## with comma2k dataset, gt path is rotate with yaw angle, tested!
    # transform_maxtrix = np.array([[np.cos(theta_bias), np.sin(theta_bias), self.X[self.cnt]], 
    #                               [-np.sin(theta_bias), np.cos(theta_bias), self.g_y],
    #                               [0,0,1]])

    # label_traj_xyz1 = (transform_maxtrix @ traj_points1.T).T


    # plt.clf()
    # plt.plot(traj_y, traj_x)
    # # plt.plot(self.X, self.Y)
    # # plt.plot(label_traj_xyz[:,0], label_traj_xyz[:,1], 'g', linewidth=4.0)
    # # plt.plot(label_traj_xyz1[:,0], label_traj_xyz1[:,1], 'r', linewidth=2.0)

    # # plt.plot(self.X[self.cnt], self.g_y, 'ro', linewidth=4.0)
    # # plt.xlim([self.X[self.cnt] - 40, self.X[self.cnt] + 250])
    # # plt.ylim([self.Y[self.cnt] - 10, self.Y[self.cnt] + 10])
    # plt.ylim([0, 250])
    # plt.xlim([-8, 8])
    # plt.show()
    # plt.pause(0.001)

    self.cnt+=1
      

import utils_comma2k19.orientation as orient_com

class SegDataset():
  def __init__(self, seg_path: str, seq_len=900, label_len=201):
    self.is_comma2k19 = 'Chunk_cm' in seg_path
    self.seg_len = seq_len
    self.label_len = label_len

    self.MEDMDL_INSMATRIX = np.array([[910.0,  0.0,   0.5 * 512],
                                [0.0,  910.0,   47.6],
                                [0.0,  0.0,     1.0]])

    self.C3_INSMATRIX = np.array([[2648.0,   0.,   1928/2.],
                            [0.,  2648.0,  1208/2.],
                            [0.,    0.,     1.]])

    self.C2_INSMATRIX = np.array([[910.0,  0.0,   0.5 * 1164],
                            [0.0,  910.0,   0.5 * 874],
                            [0.0,  0.0,     1.0]])
    try:
      if not self.is_comma2k19:

        # should we have a datacheck, such as segments length and gps valid 
        frame_positions, frame_orientations, frame_velocities, \
        calib_msgs , statuses, frame_angular_velocities, model_msgs = get_poses(seg_path)
        frame_valid =  (np.all(statuses['gpsOK'] >0)      and np.all(statuses['orientations_calib'] >0)  \
                  and  np.all(statuses['positions'] >0)  and np.all(statuses['status'] >0) and statuses['gpsOK'].shape[0]>1050)

        ang_x = np.mean(calib_msgs[:,0])
        ang_y = np.mean(calib_msgs[:,1])
        ang_z = np.mean(calib_msgs[:,2])
        rot_angle = [ang_x, ang_y, ang_z, 1.35, -0.06]

        # goto next valid dataset

        is_data_not_valid = (not frame_valid) or (statuses['gpsOK'].shape[0]<seq_len+label_len)
        # print("frame_valid", is_data_not_valid)
        cam_insmatrixs = self.C3_INSMATRIX
        video_data = cv2.VideoCapture(f"{seg_path}/fcamera.hevc")


      else:
        extrinsic_matrix, model_msgs = get_calib_extrinsic(seg_path, logfile='raw_log.bz2') 
        device_frame_from_view_frame = np.array([
          [ 0.,  0.,  1.],
          [ 1.,  0.,  0.],
          [ 0.,  1.,  0.]
        ])
        view_frame_from_device_frame = device_frame_from_view_frame.T
        # should we check the extrinsic_matrix
        is_reverse = 1 if extrinsic_matrix[1,3] >0 else -1
        M1 = is_reverse*extrinsic_matrix[0:3, 0:3]
        M2 = np.linalg.inv(view_frame_from_device_frame) @ M1
        M3 = M2 @ np.linalg.inv(np.diag([1, -1, -1]) )
        rot_angle = orient.euler_from_rot(M3).tolist()
        # rot_angle = [0,0,0]
        rot_angle += [1.22, 0.0]


        ## get label data
        frame_positions    = np.load(f"{seg_path}/global_pose/frame_positions")
        frame_orientations = np.load(f"{seg_path}/global_pose/frame_orientations")
        frame_velocities   = np.load(f"{seg_path}/global_pose/frame_velocities")

        # 10ms, we need to sample with 50ms, interp
        frame_angular_velocities_v = np.load(f"{seg_path}/processed_log/IMU/gyro/value")
        frame_angular_velocities_t0 = np.load(f"{seg_path}/processed_log/IMU/gyro/t")
        # start time is not zero
        frame_angular_velocities_t = frame_angular_velocities_t0 - frame_angular_velocities_t0[0] 
        
        frame_angular_velocities_tt = np.linspace(0,60,1201)
        frame_angular_velocities_0 = np.interp(frame_angular_velocities_tt, frame_angular_velocities_t, frame_angular_velocities_v[:,0].flatten())
        frame_angular_velocities_1 = np.interp(frame_angular_velocities_tt, frame_angular_velocities_t, frame_angular_velocities_v[:,1].flatten())
        frame_angular_velocities_2 = np.interp(frame_angular_velocities_tt, frame_angular_velocities_t, frame_angular_velocities_v[:,2].flatten())

        frame_angular_velocities = np.concatenate([np.expand_dims(frame_angular_velocities_0, 1), 
                                                  np.expand_dims(frame_angular_velocities_1, 1), 
                                                  np.expand_dims(frame_angular_velocities_2, 1)], axis=1)

        is_data_not_valid = frame_positions.shape[0]<seq_len+label_len

        cam_insmatrixs = self.C2_INSMATRIX
        video_data = cv2.VideoCapture(f"{seg_path}/video.hevc")
    except Exception as e:
      print("can not get label data!")
      print(e)
      is_data_not_valid = True

    self.data_valid = not is_data_not_valid 
    # if data is valid, we generate a good class menbers
    if self.data_valid:
      self.video_handler = video_data
      self.frame_positions = frame_positions
      self.frame_orientations = frame_orientations
      self.frame_angular_velocities = frame_angular_velocities
      self.frame_velocities = frame_velocities

      self.anchor_time = np.array([10 * i**2/32**2 for i in range(33)])
      self.rot_angle = rot_angle
      self.cam_insmatrixs = cam_insmatrixs

      # self.imgs = self.videos2numpy(f"{seg_path}/video.hevc") if self.is_comma2k19 else self.videos2numpy(f"{seg_path}/fcamera.hevc")
      local_xy_at_t0 = self.get_local_xyz_at_t0()
      local_heading_at_t0 = self.get_local_orientation_at_t0()
      p0 = [np.sin(local_heading_at_t0[0]), np.cos(local_heading_at_t0[0])]
      local_heading = np.array([-1*get_vector_angle(p0, [np.sin(heading), np.cos(heading)]) for heading in local_heading_at_t0] )

      self.local_velocity = self.get_local_velocity()
      self.simulator = Simulator(local_xy_at_t0[:, 0], -1*local_xy_at_t0[:, 1], -1*local_heading)


  def videos2numpy(self, path: str) -> np.array:
    """
      Func: we use faster conversion methods to get images numpy buffer from raw video file
    """
    if not self.is_comma2k19:
        H, W = 1208, 1928 # video dimensions
    else:
        H, W = 874, 1164 # video dimensions
    command = [ "ffmpeg", 
                '-loglevel', 'quiet',
                # '-vsync', '0',
                '-hwaccel', 'cuda', 
                '-hwaccel_output_format', 'cuda',
                '-i', path,
                '-s', f'{W}x{H}',
                '-pix_fmt', 'nv12',
                # '-c:v', 'h264_nvenc',
                '-c:v', 'libx265',
                '-vf', 'hwdownload,format=nv12' ,
                '-vcodec', 'rawvideo',
                '-f', 'rawvideo',
                '-preset', 'slow',
                'pipe:1' ]
    # run ffmpeg and load all frames into numpy array (num_frames, H, W, 3)
    st=time.time()
    pipe = subprocess.run(command, stdout=subprocess.PIPE, bufsize=10**8)
    print(time.time() - st)
    video = np.frombuffer(pipe.stdout, dtype=np.uint8).reshape(-1, int(H*1.5), W)
    return video


  def get_local_traj(self, frame_cnt: int) -> np.array:
    """
      Func: convert ecef coordinate to local coordinates with specific time anchor
    """
    ## get label trajectory
    ecef_from_local = orient.rot_from_quat(self.frame_orientations[frame_cnt])
    local_from_ecef = ecef_from_local.T
    frame_positions_local = np.einsum('ij,kj->ki', local_from_ecef, 
                                      self.frame_positions - self.frame_positions[frame_cnt]).astype(np.float32)

    traj_xyz = frame_positions_local[frame_cnt:frame_cnt+self.label_len]
    traj_t = [i*0.05 for i in range(self.label_len)]
    
    label_position_x = np.interp(self.anchor_time, traj_t, traj_xyz[:,0]).reshape(33,1)
    label_position_y = np.interp(self.anchor_time, traj_t, traj_xyz[:,1]).reshape(33,1)
    label_position_z = np.interp(self.anchor_time, traj_t, traj_xyz[:,2]).reshape(33,1)


    # # get label heading angle
    # euler_angles_ned_rad = orient_com.ned_euler_from_ecef(self.frame_positions[frame_cnt], orient.euler_from_quat(self.frame_orientations))

    # traj_heading = euler_angles_ned_rad[frame_cnt:frame_cnt+self.label_len][:,2]
    # label_heading = np.interp(self.anchor_time, traj_t, traj_heading).reshape(33,1)

    # print('label', euler_angles_ned_rad[:,2])

    return np.concatenate([label_position_x, label_position_y, label_position_z], axis=1)

  def get_local_velocity(self) -> np.array:
    ecef_from_local = orient.rot_from_quat(self.frame_orientations[0])
    local_from_ecef = ecef_from_local.T
    frame_velocities_local = np.einsum('ij,kj->ki', local_from_ecef, 
                                      self.frame_velocities).astype(np.float32)
    return frame_velocities_local

  def get_local_xyz_at_t0(self) -> np.array:
    """
      Func: translate all points to t0 axis,means start with [0,0,0]
    """
    ## get label trajectory
    ecef_from_local = orient.rot_from_quat(self.frame_orientations[0])
    local_from_ecef = ecef_from_local.T
    frame_positions_local = np.einsum('ij,kj->ki', local_from_ecef, 
                                      self.frame_positions - self.frame_positions[0]).astype(np.float32)

    # compensation yaw angle offset
    if self.is_comma2k19:
      frame_positions_local[:,1] = -1*frame_positions_local[:,1] # reverse
      # need add coordinate transform
      sz = frame_positions_local.shape[0]
      traj_points = np.concatenate((frame_positions_local[:,0].reshape(sz,1), frame_positions_local[:,1].reshape(sz,1), np.ones((sz,1))), axis=1)
      # print(traj_points.shape)

      ## with comma2k dataset, gt path is rotate with yaw angle, tested!
      theta_bias = 0.0
      pos_bias = 0.0
      compensation_theta = -1*self.rot_angle[2]
      transform_maxtrix = np.array([[np.cos(theta_bias+compensation_theta), np.sin(theta_bias+compensation_theta), 0], 
                                    [-np.sin(theta_bias+compensation_theta), np.cos(theta_bias+compensation_theta), pos_bias],
                                    [0,0,1]])

      label_traj_xyz = transform_maxtrix @ traj_points.T
      label_traj_xyz = label_traj_xyz.T
      label_traj_xyz[:,1] = -1*label_traj_xyz[:,1] # reverse

      frame_positions_local[:,0] = label_traj_xyz[:,0]
      frame_positions_local[:,1] = label_traj_xyz[:,1]


    return frame_positions_local

  def get_local_orientation_at_t0(self) -> np.array:
    """
      Func: get relatice heading angle from t0 
    """
    ## get label trajectory
    euler_angles_ned_rad = orient_com.ned_euler_from_ecef(self.frame_positions[0], orient.euler_from_quat(self.frame_orientations))

    # just using heading angle
    return euler_angles_ned_rad[:,2]

  def get_calib_matrix(self, pos_bias:float, theta_bias:float) -> np.array:
    """
      Func: if the camera heading angle of position has been changed, the trasform matrix is changed too.
    """
    # print(pos_bias, theta_bias)
    ang_x = self.rot_angle[0]
    ang_y = self.rot_angle[1]
    ang_z = self.rot_angle[2] + theta_bias

    dev_height = self.rot_angle[3]
    lat_bias = self.rot_angle[4] + pos_bias
    

    camera_frame_from_ground = np.dot(self.cam_insmatrixs,
                                        get_view_frame_from_road_frame(ang_x, ang_y, ang_z, dev_height, lat_bias))[:, (0, 1, 3)]
    calib_frame_from_ground = np.dot(self.MEDMDL_INSMATRIX,
                                        get_view_frame_from_road_frame(0, 0, 0, 1.22))[:, (0, 1, 3)]
    calib_msg = np.dot(camera_frame_from_ground, np.linalg.inv(calib_frame_from_ground))

    return calib_msg


  def get_label_traj_with_deviate(self, frame_cnt: int, pos_bias:float, theta_bias:float) -> np.array:
    """
      Func: while the adc postion is deviated, the labeled trajectory should alse changed with new view matrix.
    """
    original_traj = self.get_local_traj(frame_cnt)
    original_traj[:,1]  *= -1 #reverse

    # need add coordinate transform
    traj_points = np.concatenate((original_traj[:,0:1], original_traj[:,1:2], np.ones((33,1))), axis=1)

    ## with comma2k dataset, gt path is rotate with yaw angle, tested!
    compensation_theta = -1*self.rot_angle[2] if self.is_comma2k19 else 0.0
    transform_maxtrix = np.array([[np.cos(theta_bias+compensation_theta), np.sin(theta_bias+compensation_theta), 0], 
                                  [-np.sin(theta_bias+compensation_theta), np.cos(theta_bias+compensation_theta), pos_bias],
                                  [0,0,1]])

    label_traj_xyz = transform_maxtrix @ traj_points.T
    label_traj_xyz = label_traj_xyz.T
    label_traj_xyz[:,1] = -1*label_traj_xyz[:,1] #reverse back
    label_traj_xyz[:,2:3] = original_traj[:,2:3]
    return label_traj_xyz


  def draw_path(self, device_path, img, width=1, height=1.22, fill_color=(255,255,255), line_color=(0,255,0)) -> None:
    bs = device_path.shape[0]
    device_path_l = device_path + np.array([0, 0, height])                                                                    
    device_path_r = device_path + np.array([0, 0, height]) 

    # calib frame to raod frame                                                         
    device_path_l[:,1] -= width                                                                                               
    device_path_r[:,1] += width
    device_path_l[:,2] = -1*device_path_l[:,2]  
    device_path_l[:,1] = -1*device_path_l[:,1] 
    device_path_r[:,2] = -1*device_path_r[:,2]  
    device_path_r[:,1] = -1*device_path_r[:,1] 

    m1 = get_view_frame_from_road_frame(0, 0, 0, 0)
    calib_pts = np.vstack((device_path_l.T, np.ones((1,bs)) ))
    view_pts = m1 @ calib_pts
    for i in range(bs):
      view_pts[0,i] = view_pts[0,i]/max(view_pts[2,i], 2)
      view_pts[1,i] = view_pts[1,i]/max(view_pts[2,i], 2)
      view_pts[2,i] = view_pts[2,i]/max(view_pts[2,i], 2)

    img_pts_l = self.MEDMDL_INSMATRIX @ view_pts
    img_pts_l = img_pts_l.astype(int)
    calib_pts = np.vstack((device_path_r.T, np.ones((1,bs)) ))
    view_pts = m1 @ calib_pts
    for i in range(bs):
      view_pts[0,i] = view_pts[0,i]/max(view_pts[2,i], 2)
      view_pts[1,i] = view_pts[1,i]/max(view_pts[2,i], 2)
      view_pts[2,i] = view_pts[2,i]/max(view_pts[2,i], 2)

    img_pts_r = self.MEDMDL_INSMATRIX @ view_pts
    img_pts_r = img_pts_r.astype(int)
    for i in range(1, img_pts_l.shape[1]):
      #check valid
      if img_pts_l[2,i] >0 and img_pts_r[2,i] >0:
        u1 = img_pts_l[0, i-1]
        v1 = img_pts_l[1, i-1]
        u2 = img_pts_r[0, i-1]
        v2 = img_pts_r[1, i-1]
        u3 = img_pts_l[0, i]
        v3 = img_pts_l[1, i]
        u4 = img_pts_r[0, i]
        v4 = img_pts_r[1, i]
        pts = np.array([[u1,v1],[u2,v2],[u4,v4],[u3,v3]], np.int32).reshape((-1,1,2))
        cv2.fillPoly(img,[pts],fill_color)
        cv2.polylines(img,[pts],True,line_color)

  def next_image(self):
    try:
      ret, image = self.video_handler.read()
    except Exception as e:
      print("read video image failed!")
      print(e)
      ret = False
    if not ret:
      return None
    else:
      return image



# seg_path = "/media/chaoqun/Faster_DDs/03_zs_dd/2.zs_c3/1/0811/2023-08-12--09-45-54--91/"
# # seg_path = "/media/chaoqun/Faster_DDs/03_zs_dd/1.comma_2k/Chunk_8/99c94dc769b5d96e--2018-09-19--16-09-16/20/"
# # seg_path = "/media/chaoqun/datadisk_2TB/08_end2end/e2e_lateral_planning/dataset/realdata/training/Chunk_cm8/99c94dc769b5d96e--2018-09-19--16-09-16/25/"


# sd = SegDataset(seg_path)


# local_xy_at_t0 = sd.get_local_xyz_at_t0()
# local_heading_at_t0 = sd.get_local_orientation_at_t0()

# p0 = [np.sin(local_heading_at_t0[0]), np.cos(local_heading_at_t0[0])]
# local_heading = np.array([-1*get_vector_angle(p0, [np.sin(heading), np.cos(heading)]) for heading in local_heading_at_t0] )

# # print("local_heading_at_t0", local_heading)
# # plt.plot(local_xy_at_t0[:,0], local_xy_at_t0[:,1], 'ro')
# # plt.show()
# # plt.plot(local_heading)
# # plt.plot(local_heading_at_t0 - local_heading_at_t0[0] - 2*np.pi)

# # dx = local_xy_at_t0[1:, 0] - local_xy_at_t0[0:-1, 0]
# # dy = local_xy_at_t0[1:, 1] - local_xy_at_t0[0:-1, 1]
# # local_heading = [math.atan2(y, x) for x,y in zip(dx, dy)]
# # plt.plot(angle)
# # plt.show()


# local_velocity = sd.get_local_velocity()
# simulator = Simulator(local_xy_at_t0[:, 0], -1*local_xy_at_t0[:, 1], -1*local_heading)

# for i in range(900):
#   img = cv2.cvtColor(sd.imgs[i], cv2.COLOR_YUV2BGR_NV12)
#   # get feed model image with yuv format

#   pos = simulator.error_pos
#   theta = simulator.error_heading
#   new_calib_matrix = sd.get_calib_matrix(pos, theta)
#   img = cv2.warpPerspective(src=img, M=new_calib_matrix, dsize=(512,256), flags=cv2.WARP_INVERSE_MAP)
#   path = sd.get_label_traj_with_deviate(i, pos, -1*theta)
#   sd.draw_path(path, img, width=0.5)
#   cv2.imshow("test", img)
#   cv2.waitKey(1)

#   print(sd.rot_angle[2])


#   local_heading_at_t = local_heading[i:i+201] - local_heading[i-1] + theta
#   traj_t = [i*0.05 for i in range(201)]
#   traj_heading = np.interp(sd.anchor_time, np.array(traj_t), local_heading_at_t)

#   simulator.update(True, local_velocity[i,0],-1*sd.frame_angular_velocities[i][2], path[:,0], -1*path[:,1], -1*traj_heading, sd.anchor_time)


