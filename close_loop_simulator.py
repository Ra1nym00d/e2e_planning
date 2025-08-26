import cv2
import numpy as np
import copy
import torch
import torch.nn as nn
import math
import time
import matplotlib.pyplot as plt

import utils_comma2k19.coordinates as coord
import utils_comma2k19.orientation as orient
import laika.lib.orientation as orient1
from utils import C2_INSMATRIX, C3_INSMATRIX, MEDMDL_INSMATRIX, LABEL_LEN, TRAJECTORY_TIME, ANCHOR_TIME, \
			get_poses, get_view_frame_from_road_frame, get_calib_extrinsic, reshape_yuv, draw_path, Simulator


from lat_mpc import LatMpc


class Simulator1():
	def __init__(self, X, Y, Heading):
		self.out = 0.0
		self.controller = LatMpc()
		self.dy = 0.1
		self.dheading = 0.0
		self.dt = 0.05

		# self.fig = plt.figure(figsize=( 12, 9) , dpi=80)
		# plt.ion()

		self.g_heading = -0.02
		self.g_y = 0.8
		self.cnt = 0

		self.X = X
		self.Y = Y
		self.Heading = Heading

		self.error_pos = 0
		self.error_heading = 0

		self.cmds = [0 for i in range(10)]

		self.fig = plt.figure(figsize=( 12, 9) , dpi=80)
		plt.ion()

		self.pos_err_limit = 0.8
		self.heading_err_limit = 0.08



	def update(self, is_active, velocity, omiga, traj_x, traj_y, traj_t) -> float:
		# first get front wheel angle
		# print("traj_y", traj_y)
		# print("traj_y", traj_y)
		# print("traj_t", traj_t)

		# gt_heading_local = self.Heading[self.cnt:self.cnt+LABEL_LEN] - self.Heading[self.cnt]
		# traj_theta = 1*np.interp(ANCHOR_TIME, TRAJECTORY_TIME, gt_heading_local).reshape(33,1)[0:32]

		p = np.polyfit(traj_x.flatten()[0:24], traj_y.flatten()[0:24], 3)
		

		dx = velocity*self.dt
		# next_step_dtheta = p[-2] + 2*p[-3]*dx + 3*p[-4]*dx*dx

		pred_dx = np.clip(traj_x[1:33] - traj_x[0:32], 1e-3, 1e4) # avoid zero divide
		pred_dy = traj_y[1:33] - traj_y[0:32]
		traj_theta = np.array([math.atan(x) for x in pred_dy.flatten()/pred_dx.flatten()]) # deg


		# print("traj_theta", traj_theta[0])

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
		self.g_y += np.sin(self.g_heading) * velocity * self.dt

		self.error_pos = self.Y[self.cnt] - self.g_y
		self.error_heading = self.g_heading -  self.Heading[self.cnt]

		self.error_pos = min(self.error_pos, self.pos_err_limit) if self.error_pos>0 \
							  else max(self.error_pos, -self.pos_err_limit)
		self.error_heading = min(self.error_heading, self.heading_err_limit) if self.error_heading>0 \
							  else max(self.error_heading, -self.heading_err_limit)

		print(self.cnt, self.error_pos, self.g_y, self.Y[self.cnt], self.g_heading, self.Heading[self.cnt] , yawrate_cmd)

		# next_step_dy = np.interp(0.05, traj_t, traj_y.flatten())
		# next_step_dtheta = np.interp(0.05, traj_t[0:-1], traj_theta.flatten())
		# # print(next_step_dtheta, np.interp(0.05, traj_t[0:-1], traj_theta.flatten()))

		
		# print("next step ", next_step_dy, next_step_dtheta)
		# print("estimate", delta_y, delta_heading)
		# gap_dy     =  next_step_dy/2.
		# gap_dtheta =  delta_heading - next_step_dtheta

		# print('---------------------------------', next_step_dy)

		# self.dy = gap_dy
		# self.dheading = gap_dtheta
		# print("offset", self.dy, self.dheading, delta_heading, next_step_dtheta)


		# plt.clf()
		# plt.plot(traj_x, traj_y)
		# plt.xlim([0,250])
		# plt.ylim([-20, 20])
		# plt.show()
		# plt.pause(0.02)

		self.cnt+=1

		# plt.clf()
		# plt.plot(self.X, self.Y)
		# plt.plot(self.X[self.cnt], self.g_y, 'o')
		# plt.ylim([-20, 20])
		# plt.show()
		# plt.pause(0.001)



# class TestCloseLoop():
# 	def __init__(self):
# 		self.euler_angles_ned_rad = orient.ned_euler_from_ecef(frame_positions[0], orient.euler_from_quat(frame_orientations))

# 	def


# cc.update(True, 10, 0.01, np.linspace(0,100,33), np.linspace(0.2, 5, 33),  np.linspace(0., 10, 33))


test_segment = "/media/chaoqun/datadisk_2TB/08_end2end/e2e_lateral_planning/dataset/realdata/training/Chunk_cm3/99c94dc769b5d96e--2018-05-02--11-42-52/10/"

hevc_video = cv2.VideoCapture(test_segment + "/video.hevc")
extrinsic_matrix,_ = get_calib_extrinsic(test_segment, logfile='raw_log.bz2') 

device_frame_from_view_frame = np.array([
	[ 0.,  0.,  1.],
	[ 1.,  0.,  0.],
	[ 0.,  1.,  0.]
])
view_frame_from_device_frame = device_frame_from_view_frame.T

# should we check the extrinsic_matrix
print(extrinsic_matrix)
is_reverse = 1 if extrinsic_matrix[1,3] >0 else -1
M1 = is_reverse*extrinsic_matrix[0:3, 0:3]
M2 = np.linalg.inv(view_frame_from_device_frame) @ M1
M3 = M2 @ np.linalg.inv(np.diag([1, -1, -1]) )
rot_angle = orient1.euler_from_rot(M3).tolist()

ang_x = rot_angle[0]
ang_y = rot_angle[1]
ang_z = rot_angle[2]

## get label data
frame_positions    = np.load(f"{test_segment}/global_pose/frame_positions")
frame_orientations = np.load(f"{test_segment}/global_pose/frame_orientations")
frame_velocities   = np.load(f"{test_segment}/global_pose/frame_velocities")
frame_angular_velocities = np.load(f"{test_segment}/processed_log/IMU/gyro/value")[::5]

# test_segment = "/home/chaoqun/Videos/2023-09-20--07-56-52--40/"
# test_segment = "/media/chaoqun/datadisk_2TB/08_end2end/e2e_lateral_planning/dataset/realdata/training/Chunk_4/0811/2023-08-12--09-45-54--94/"

# hevc_video = cv2.VideoCapture(test_segment + "/fcamera.hevc")

# frame_positions, frame_orientations, frame_velocities, calib_msgs , statuses, frame_angular_velocities,_ = get_poses(test_segment)

# print(frame_orientations.shape, frame_positions.shape)
euler_angles_ned_rad = orient.ned_euler_from_ecef(frame_positions[0], orient.euler_from_quat(frame_orientations))

## get global  trajectory
ecef_from_local = orient.rot_from_quat(frame_orientations[0])
local_from_ecef = ecef_from_local.T
frame_positions_local = np.einsum('ij,kj->ki', local_from_ecef, 
                                  frame_positions - frame_positions[0]).astype(np.float32)

frame_velocities_local = np.einsum('ij,kj->ki', local_from_ecef, 
                                  frame_velocities).astype(np.float32)

# compensation yaw angle offset

frame_positions_local[:,1] = -1*frame_positions_local[:,1] # reverse
# need add coordinate transform
traj_points = np.concatenate((frame_positions_local[:,0].reshape(1200,1), frame_positions_local[:,1].reshape(1200,1), np.ones((1200,1))), axis=1)
# print(traj_points.shape)

## with comma2k dataset, gt path is rotate with yaw angle, tested!
theta_bias = 0.0
compensation_theta = -1*rot_angle[2]
transform_maxtrix = np.array([[np.cos(theta_bias+compensation_theta), np.sin(theta_bias+compensation_theta), 0], 
															[-np.sin(theta_bias+compensation_theta), np.cos(theta_bias+compensation_theta), 0.0],
															[0,0,1]])

label_traj_xyz = transform_maxtrix @ traj_points.T
label_traj_xyz = label_traj_xyz.T
label_traj_xyz[:,1] = -1*label_traj_xyz[:,1] # reverse

frame_positions_local[:,0] = label_traj_xyz[:,0]
frame_positions_local[:,1] = label_traj_xyz[:,1]

cc = Simulator(frame_positions_local[:, 0], -1*frame_positions_local[:, 1], -1*(euler_angles_ned_rad[:,2] - euler_angles_ned_rad[0,2]))


frame_cnt = 0

output = cv2.VideoWriter('test_closeloop.avi', 
								cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, (512, 256))

# load model and set eval mode
plan_model = torch.load('out/planning_model.pt', map_location=torch.device("cpu"))
plan_model.eval()
plan_model.cpu()


hidden_st = torch.zeros((2,1, 512)).cpu()
last_bat_img = torch.zeros((6,128,256)).cpu()
prob_func = nn.Softmax(dim=0)
plt.ion()
while True:

	ret, img = hevc_video.read()

	if not ret:
		break


	## get label trajectory
	ecef_from_local = orient.rot_from_quat(frame_orientations[frame_cnt])
	local_from_ecef = ecef_from_local.T
	frame_positions_local = np.einsum('ij,kj->ki', local_from_ecef, 
	                                  frame_positions - frame_positions[frame_cnt]).astype(np.float32)

	frame_velocities_local = np.einsum('ij,kj->ki', local_from_ecef, 
	                                  frame_velocities).astype(np.float32)

	traj_xyz = frame_positions_local[frame_cnt:frame_cnt+LABEL_LEN]


	
	pos_bias =  cc.error_pos
	theta_bias = cc.error_heading

	# extrinsic_matrix_1 = copy.deepcopy(extrinsic_matrix)
	# # add noise bias 
	# extrinsic_matrix_1[0,3] += -1*rand_bias
	# # print(extrinsic_matrix)

	# camera_frame_from_ground = np.dot(C2_INSMATRIX, extrinsic_matrix_1[:, (0, 1, 3)] )
	# calib_frame_from_ground = np.dot(MEDMDL_INSMATRIX,
	#                                   get_view_frame_from_road_frame(0, 0, 0, 1.22))[:, (0, 1, 3)]
	# calib_frame_from_camera_frame = np.dot(camera_frame_from_ground, np.linalg.inv(calib_frame_from_ground))

	# extrinsic_matrix, model_msgs = get_calib_extrinsic(seg_path, logfile='raw_log.bz2') 



	# ang_x = np.mean(calib_msgs[:,0])
	# ang_y = np.mean(calib_msgs[:,1]) 
	# ang_z = np.mean(calib_msgs[:,2]) - theta_bias


	# print(get_view_frame_from_road_frame(ang_x, ang_y, ang_z, 1.32, -0.06 + rand_bias))
	camera_frame_from_ground = np.dot(C2_INSMATRIX,
	                                    get_view_frame_from_road_frame(ang_x, ang_y, ang_z- theta_bias, 1.22, pos_bias))[:, (0, 1, 3)]
	calib_frame_from_ground = np.dot(MEDMDL_INSMATRIX,
	                                    get_view_frame_from_road_frame(0, 0, 0, 1.22))[:, (0, 1, 3)]

	calib_frame_from_camera_frame = np.dot(camera_frame_from_ground, np.linalg.inv(calib_frame_from_ground))



	label_position_x = np.interp(ANCHOR_TIME, TRAJECTORY_TIME, traj_xyz[:,0]).reshape(33,1)

	direction = 1.

	label_position_y = (np.interp(ANCHOR_TIME, TRAJECTORY_TIME, traj_xyz[:,1]).reshape(33,1)- 0) * direction

	label_position_y *=-1

	# print(label_position_y)
	label_position_z = np.interp(ANCHOR_TIME, TRAJECTORY_TIME, traj_xyz[:,2]).reshape(33,1)
	# print(label_position_z)

	# need add coordinate transform
	traj_points = np.concatenate((label_position_x, label_position_y, np.ones((33,1))), axis=1)
	# print(traj_points.shape)


	theta = theta_bias

	# transform_maxtrix = np.array([[np.cos(theta), np.sin(theta), 0], 
	# 							[-np.sin(theta), np.cos(theta), pos_bias],
	# 							[0,0,1]])

	# new_points = transform_maxtrix @ traj_points.T
	# new_points = new_points.T

	## with comma2k dataset, gt path is rotate with yaw angle, tested!
	compensation_theta = 0.0

	compensation_theta = -1*rot_angle[2]
	transform_maxtrix = np.array([[np.cos(theta_bias+compensation_theta), np.sin(theta_bias+compensation_theta), 0], 
																[-np.sin(theta_bias+compensation_theta), np.cos(theta_bias+compensation_theta), pos_bias],
																[0,0,1]])

	new_points = transform_maxtrix @ traj_points.T
	new_points = new_points.T

	# print(theta, new_points[:, 1], label_position_y)

	crop_img = cv2.warpPerspective(src=img, M=calib_frame_from_camera_frame, dsize=(512,256), flags=cv2.WARP_INVERSE_MAP)


	# crop_img = cv2.warpPerspective(src=img, M=calib_frame_from_camera_frame, dsize=(512,256), flags=cv2.WARP_INVERSE_MAP)
	feed_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2YUV_I420)
	feed_img = reshape_yuv(feed_img)

	with torch.no_grad():
		# use last image and current image to feed model, scalar to [0 1]
		input_bev = torch.from_numpy(feed_img.copy()).cpu()
		X_in = torch.cat([last_bat_img, input_bev], dim=0).float()
		preds_buffer, hidden_st, feat_vec = plan_model(X_in.reshape(1,12,128,256), hidden_st)

		# print(feat_vec.cpu().detach().numpy())
		# plt.clf()
		# plt.plot(np.linspace(0,1024,1024), feat_vec.cpu().detach().numpy().flatten())
		# plt.ylim([-1, 2])
		# plt.show()
		# plt.pause(0.001)
		last_bat_img = input_bev.clone().detach()


		pred_cls_obj = preds_buffer[:,0:5]
		pred_traj_obj = preds_buffer[:,5:5+5*33*6].reshape(-1, 5, 33, 6)

		pred_pose = preds_buffer[:, 5+5*33*6:5+5*33*6+12]
		# print(pred_pose[0])

		## get loss
		traj_cls = torch.squeeze(pred_cls_obj)
		traj_pred = torch.squeeze(pred_traj_obj)
		traj_prob = prob_func(traj_cls).cpu().detach().numpy()


		best_idx = np.argmax(traj_prob)
		best_traj_x = traj_pred[best_idx,:,0].cpu().detach().numpy()
		best_traj_y = traj_pred[best_idx,:,1].cpu().detach().numpy()
		best_traj_z = np.zeros((33,1))

		best_traj_std = 1 + traj_pred[best_idx,:,3].cpu().detach().numpy()
		print("best_traj_y", best_traj_y)
		# print("gt_traj_y", -1*new_points[:,1])

		best_traj = np.concatenate((best_traj_x.reshape(33,1), best_traj_y.reshape(33,1), best_traj_z.reshape(33,1)), axis=1)

	draw_path(best_traj[:,:], crop_img, width=1.0, height=1.22, fill_color=(255,255,255))


	# gt_traj = np.concatenate((best_traj_x.reshape(33,1), best_traj_y.reshape(33,1), best_traj_z.reshape(33,1)), axis=1)

	# print(new_points)

	new_points[:,1] = -1*new_points[:,1]
	new_points[:,2:3] = 0

	draw_path(new_points, crop_img, width=1.0, height=1.22, fill_color=(122,122,122))

	cv2.imshow("test", crop_img)
	cv2.waitKey(1)
	# time.sleep(1)

	output.write(crop_img)

	frame_cnt+=1

	# print("label_position_x", label_position_x)
	# cc.update(True, frame_velocities_local[frame_cnt][0], -1*frame_angular_velocities[frame_cnt][2], new_points[:,0], new_points[:, 1], ANCHOR_TIME)


	cc.update(True, frame_velocities_local[frame_cnt][0], -1*frame_angular_velocities[frame_cnt][2], best_traj_x, -1*best_traj_y, ANCHOR_TIME)
