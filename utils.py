import cv2
import numpy as np
import h5py
import os
import argparse
from common.transformations.camera import get_view_frame_from_road_frame
from tools.lib.logreader import LogReader
import time
import laika.lib.coordinates as coord
import laika.lib.orientation as orient
# import av
import subprocess
import onnxruntime as ort


MEDMDL_INSMATRIX = np.array([[910.0,  0.0,   0.5 * 512],
                            [0.0,  910.0,   47.6],
                            [0.0,  0.0,     1.0]])

C3_INSMATRIX = np.array([[2648.0,   0.,   1928/2.],
                        [0.,  2648.0,  1208/2.],
                        [0.,    0.,     1.]])

C2_INSMATRIX = np.array([[910.0,  0.0,   0.5 * 1164],
                        [0.0,  910.0,   0.5 * 874],
                        [0.0,  0.0,     1.0]])


ANCHOR_TIME = np.array((0.        ,  0.00976562,  0.0390625 ,  0.08789062,  0.15625   ,
                        0.24414062,  0.3515625 ,  0.47851562,  0.625     ,  0.79101562,
                        0.9765625 ,  1.18164062,  1.40625   ,  1.65039062,  1.9140625 ,
                        2.19726562,  2.5       ,  2.82226562,  3.1640625 ,  3.52539062,
                        3.90625   ,  4.30664062,  4.7265625 ,  5.16601562,  5.625     ,
                        6.10351562,  6.6015625 ,  7.11914062,  7.65625   ,  8.21289062,
                        8.7890625 ,  9.38476562, 10.))
MDL_TIME = 0.05
LABEL_LEN = 201
TRAJECTORY_TIME  = [MDL_TIME*i for i in range(LABEL_LEN)]





def sigmoid(x):
    return 1 / (1 + np.exp(-x))



# class PlanModel():
#     def __init__(self):
#         options = ort.SessionOptions()
#         provider = 'CUDAExecutionProvider'
#         self.session = ort.InferenceSession(f'plannine.onnx', options, [provider])

#         # print shapes
#         input_shapes = {i.name: i.shape for i in self.session.get_inputs()}
#         output_shapes = {i.name: i.shape for i in self.session.get_outputs()}
#         print('input shapes : ', input_shapes)
#         print('output shapes: ', output_shapes)

#         self.recurrent_state = np.zeros((1, 512)).astype(np.float32)

# def my_softmax(x):
#     exp_x = np.exp(x)
#     return exp_x/np.sum(exp_x)
    
# class PlanModel():
#     def __init__(self):
#         options = ort.SessionOptions()
#         provider = 'CPUExecutionProvider'
#         self.session = ort.InferenceSession(f'./out/planning_model.onnx', options, [provider])

#         # print shapes
#         input_shapes = {i.name: i.shape for i in self.session.get_inputs()}
#         output_shapes = {i.name: i.shape for i in self.session.get_outputs()}
#         print('input shapes : ', input_shapes)
#         print('output shapes: ', output_shapes)

#         self.recurrent_state = np.zeros((1, 1024)).astype(np.float32)


#     def run(self, feed_img):
#         img = feed_img

#         model_out = self.session.run(None, {'big_imgs': img, 'hiddenst_in': self.recurrent_state})
#         preds_buffer, self.recurrent_state = model_out[0], model_out[1]
#         pred_cls_obj = preds_buffer[:,0:5]
#         pred_traj_obj = preds_buffer[:,5:5+5*33*6].reshape(5, 33, 6)

#         print("predict velocity is : ",  (preds_buffer[:,5+5*33*6]) )
#         traj_prob = my_softmax(pred_cls_obj)
#         traj_batch = []
#         for j in range(5):
#             traj_x  = (pred_traj_obj[j, :, 0]) 
#             traj_y  =  (pred_traj_obj[j, :, 1] )
#             traj_xstd  = (pred_traj_obj[j, :, 2]) 
#             traj_ystd  =  (pred_traj_obj[j, :, 3] )
#             traj_batch.append((traj_prob[0,j],traj_x, traj_y, traj_xstd, traj_ystd ))

#         return traj_batch


## onnx planning demo model runner, get images, pred trajectory!
class PlanModel():
	def __init__(self, cuda=True):
		options = ort.SessionOptions()
		provider = 'CUDAExecutionProvider' if cuda else 'CPUExecutionProvider'
		self.session = ort.InferenceSession(f'./out/planning_model.onnx', options, [provider])

		# print shapes
		input_shapes = {i.name: i.shape for i in self.session.get_inputs()}
		output_shapes = {i.name: i.shape for i in self.session.get_outputs()}
		print('input shapes : ', input_shapes)
		print('output shapes: ', output_shapes)

		self.recurrent_state = np.zeros((1, 20, 128)).astype(np.float32)
		self.last_feed_imgs = np.zeros((1, 6, 128, 256)).astype(np.float32)

	@staticmethod
	def my_softmax(x):
			exp_x = np.exp(x)
			return exp_x/np.sum(exp_x)
	@staticmethod
	def elu(x, alpha=1.0):
		return 1.0 + np.where(x < 0, alpha*(np.exp(x) - 1.0), x)
		# return np.exp(x)

	def run(self, feed_img):
		big_imgs = np.concatenate([self.last_feed_imgs, feed_img], axis=1)
		model_out = self.session.run(None, {'big_imgs': big_imgs, 'hiddenst_in': self.recurrent_state})
			
		preds_buffer, next_feat = model_out[0][:, 512+128:], model_out[0][:, 512:512+128]
		pred_cls_obj = preds_buffer[:,0:5]
		pred_traj_obj = preds_buffer[:,5:5+5*33*3].reshape(5, 33, 3)
		pred_traj_obj_std = preds_buffer[:,5+5*33*3:5+5*33*3*2].reshape(5, 33, 3)

		self.recurrent_state = np.roll(self.recurrent_state, -1, axis=1)
		self.recurrent_state[:, -1, :]   = next_feat  
		print("predict velocity is : ",  (preds_buffer[:,5+5*33*6]),  PlanModel.elu(preds_buffer[:,5+5*33*6 + 6]))
		traj_prob = PlanModel.my_softmax(pred_cls_obj)
		traj_batch = []
		for j in range(5):
			traj_x  = (pred_traj_obj[j, :, 0]) 
			traj_y  =  (pred_traj_obj[j, :, 1] )
			traj_xstd  = PlanModel.elu(pred_traj_obj_std[j, :, 0]) 
			traj_ystd  =  PlanModel.elu(pred_traj_obj_std[j, :, 1] )
			traj_batch.append((traj_prob[0,j],traj_x, traj_y, traj_xstd, traj_ystd ))
			print("traj_x", traj_x)
			print("traj_xstd", traj_xstd)
		self.last_feed_imgs = feed_img
		return traj_batch
     

        
class ComModel():
    def __init__(self):
        options = ort.SessionOptions()
        provider = 'CPUExecutionProvider'
        self.session = ort.InferenceSession(f'./out/supercombo.onnx', options, [provider])

        # print shapes
        input_shapes = {i.name: i.shape for i in self.session.get_inputs()}
        output_shapes = {i.name: i.shape for i in self.session.get_outputs()}
        print('input shapes : ', input_shapes)
        print('output shapes: ', output_shapes)

        self.recurrent_state = np.zeros((1, 512)).astype(np.float32)


    def extract_preds(self, outputs, best_plan_only=False):
        # N is batch_size

        plan_start_idx = 0
        plan_end_idx = 4955

        lanes_start_idx = plan_end_idx
        lanes_end_idx = lanes_start_idx + 528

        lane_lines_prob_start_idx = lanes_end_idx
        lane_lines_prob_end_idx = lane_lines_prob_start_idx + 8

        road_start_idx = lane_lines_prob_end_idx
        road_end_idx = road_start_idx + 264

        # plan
        plan = outputs[:, plan_start_idx:plan_end_idx]  # (N, 4955)
        plans = plan.reshape((-1, 5, 991))  # (N, 5, 991)
        plan_probs = plans[:, :, -1]  # (N, 5)
        

        plans = plans[:, :, :-1].reshape(-1, 5, 2, 33, 15)  # (N, 5, 2, 33, 15)
        best_plan_idx = np.argmax(plan_probs, axis=1)[0]  # (N,)
        print("best_plan_idx", best_plan_idx)
        best_plan = plans[:, best_plan_idx, ...]  # (N, 2, 33, 15)

        # lane lines
        lane_lines = outputs[:, lanes_start_idx:lanes_end_idx]  # (N, 528)
        lane_lines_deflat = lane_lines.reshape((-1, 2, 264))  # (N, 2, 264)
        lane_lines_means = lane_lines_deflat[:, 0, :]  # (N, 264)
        lane_lines_means = lane_lines_means.reshape(-1, 4, 33, 2)  # (N, 4, 33, 2)

        outer_left_lane = lane_lines_means[:, 0, :, :]  # (N, 33, 2)
        inner_left_lane = lane_lines_means[:, 1, :, :]  # (N, 33, 2)
        inner_right_lane = lane_lines_means[:, 2, :, :]  # (N, 33, 2)
        outer_right_lane = lane_lines_means[:, 3, :, :]  # (N, 33, 2)

        # lane lines probs
        lane_lines_probs = outputs[:, lane_lines_prob_start_idx:lane_lines_prob_end_idx]  # (N, 8)
        lane_lines_probs = lane_lines_probs.reshape((-1, 4, 2))  # (N, 4, 2)
        lane_lines_probs = sigmoid(lane_lines_probs[:, :, 1])  # (N, 4), 0th is deprecated

        outer_left_prob = lane_lines_probs[:, 0]  # (N,)
        inner_left_prob = lane_lines_probs[:, 1]  # (N,)
        inner_right_prob = lane_lines_probs[:, 2]  # (N,)
        outer_right_prob = lane_lines_probs[:, 3]  # (N,)

        # road edges
        road_edges = outputs[:, road_start_idx:road_end_idx]
        road_edges_deflat = road_edges.reshape((-1, 2, 132))  # (N, 2, 132)
        road_edge_means = road_edges_deflat[:, 0, :].reshape(-1, 2, 33, 2)  # (N, 2, 33, 2)
        road_edge_stds = road_edges_deflat[:, 1, :].reshape(-1, 2, 33, 2)  # (N, 2, 33, 2)

        left_edge = road_edge_means[:, 0, :, :]  # (N, 33, 2)
        right_edge = road_edge_means[:, 1, :, :]
        left_edge_std = road_edge_stds[:, 0, :, :]  # (N, 33, 2)
        right_edge_std = road_edge_stds[:, 1, :, :]

        batch_size = best_plan.shape[0]

        result_batch = []

        for i in range(batch_size):
            lanelines = [outer_left_lane[i], inner_left_lane[i], inner_right_lane[i], outer_right_lane[i]]
            lanelines_probs = [outer_left_prob[i], inner_left_prob[i], inner_right_prob[i], outer_right_prob[i]]
            road_edges = [left_edge[i], right_edge[i]]
            road_edges_probs = [left_edge_std[i], right_edge_std[i]]

            if best_plan_only:
                plan = best_plan[i]
            else:
                plan = (plans[i], plan_probs[i])

            result_batch.append(((lanelines, lanelines_probs), (road_edges, road_edges_probs), plan))

        return result_batch


    def update(self, feed_img):
        print(feed_img.shape)
        img = feed_img
        # recurrent_state = np.zeros((1, 512)).astype(np.float32)
        desire = np.zeros((1, 8)).astype(np.float32)
        tc = np.array([[0, 1]]).astype(np.float32)

        outs = self.session.run(None, {'input_imgs': img, 'desire': desire, 
                                       'traffic_convention': tc, 'initial_state': self.recurrent_state})[0]
        # print(outs)

        self.recurrent_state = outs[:, -512:]
        return self.extract_preds(outs)
    

def draw_path(device_path, img, width=1, height=1.22, fill_color=(255,255,255), line_color=(0,255,0)):
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


    m1 = get_view_frame_from_road_frame(0, 0, 0, 0, 0)

    calib_pts = np.vstack((device_path_l.T, np.ones((1,bs)) ))

    view_pts = m1 @ calib_pts

    for i in range(bs):
        view_pts[0,i] = view_pts[0,i]/max(view_pts[2,i], 2)
        view_pts[1,i] = view_pts[1,i]/max(view_pts[2,i], 2)
        view_pts[2,i] = view_pts[2,i]/max(view_pts[2,i], 2)


    img_pts_l = MEDMDL_INSMATRIX @ view_pts
    img_pts_l = img_pts_l.astype(int)


    calib_pts = np.vstack((device_path_r.T, np.ones((1,bs)) ))

    view_pts = m1 @ calib_pts

    for i in range(bs):
        view_pts[0,i] = view_pts[0,i]/max(view_pts[2,i], 2)
        view_pts[1,i] = view_pts[1,i]/max(view_pts[2,i], 2)
        view_pts[2,i] = view_pts[2,i]/max(view_pts[2,i], 2)


    img_pts_r = MEDMDL_INSMATRIX @ view_pts
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
    model_msgs = [m.modelV2 for m in logs if m.which() == 'modelV2']

    # print(calib_msgs[0].rpyCalib)
        
    return poses, status, np.array([m.rpyCalib for m in calib_msgs]), model_msgs


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
    logs = LogReader(os.path.join(segment_dir, logfile))

    calib_msgs = [m.liveCalibration for m in logs if m.which() == 'liveCalibration']
    # print(calib_msgs[0])

    kalman_msgs = [m.liveLocationKalman for m in logs if m.which() == 'liveLocationKalman']

    trans_msgs = np.array([m.velocityCalibrated.value for m in kalman_msgs])
    rot_msgs = np.array([m.angularVelocityCalibrated.value for m in kalman_msgs])

    # print([m.which() for m in logs])

    extrinsic_matrix = np.array(calib_msgs[0].extrinsicMatrix).reshape((3,4))
    model_msgs = [m.modelV2 for m in logs if m.which() == 'modelV2']

    return extrinsic_matrix, model_msgs


def get_hevc_np(path):
    is_comma2k19 = 'Chunk_cm' in path

    if not is_comma2k19:
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
    # st=time.time()
    pipe = subprocess.run(command, stdout=subprocess.PIPE, bufsize=10**8)
    # print(time.time() - st)
    video = np.frombuffer(pipe.stdout, dtype=np.uint8).reshape(-1, int(H*1.5), W)
    return video


def valid_segment_slice(model_msgs, frame_orientations, frame_positions, max_frame):
  # plt.ion()
  valid_seg = []

  last_valid_index = 0

  frame_cnt = 0

  last_divate_status = False


  while frame_cnt<max_frame:
    # ret, img = cap.read()
    # # print(img)

    # img = cv2.resize(img, (480,240))
    # cv2.imshow("test", img)
    # cv2.waitKey(10)

    ## get label trajectory
    ecef_from_local = orient.rot_from_quat(frame_orientations[frame_cnt])
    local_from_ecef = ecef_from_local.T
    frame_positions_local = np.einsum('ij,kj->ki', local_from_ecef, 
                                      frame_positions - frame_positions[frame_cnt]).astype(np.float32)

    traj_xyz = frame_positions_local[frame_cnt:frame_cnt+LABEL_LEN]

    # get lanelines related
    lanelines_l = model_msgs[frame_cnt].laneLines[1]
    lanelines_r = model_msgs[frame_cnt].laneLines[2]

    lanelines_ll = model_msgs[frame_cnt].laneLines[0]
    lanelines_rr = model_msgs[frame_cnt].laneLines[3]

    # print(lanelines_0.x)
    centerline_y = (np.array(lanelines_l.y) + np.array(lanelines_r.y))/2.
    # print(centerline_y)
    future_time = 4 # sec
    future_index = 4*20

    traj = traj_xyz[0:future_index]

    bound_y_l = np.interp(traj[:,0], np.array(lanelines_l.x), np.array(lanelines_l.y))
    bound_y_r = np.interp(traj[:,0], np.array(lanelines_r.x), np.array(lanelines_r.y))

    center_y  = np.interp(traj[:,0], np.array(lanelines_r.x), centerline_y)

    # print(bound_y_l)

    is_current_lane_ok = [msg.laneLineProbs[1] > 0.5 \
                        and msg.laneLineProbs[2] > 0.5  for msg in model_msgs[frame_cnt:frame_cnt+future_index] ]
    
    # print(all(is_current_lane_ok))
    divate_factor = 0.8
    # if we start a lane change, all path points should diviate from center laneline
    is_ego_traj_deviate = is_current_lane_ok and np.any(traj[:,1] < divate_factor*bound_y_l) or np.any(traj[:,1] > divate_factor*bound_y_r) 

    # we need early find last centerlast_divate_status keep drive point time
    valid_idx = 0

    # reset last valid index
    if last_divate_status and not is_ego_traj_deviate:
      last_valid_index = frame_cnt
      print(last_valid_index)

    # if we get a deviate event, we add a valid seg
    if is_ego_traj_deviate and not last_divate_status:
      print(frame_cnt)
      for i in range(LABEL_LEN):
        # check future 1s is all divate or not
        is_future_1s_deviate = np.all(traj[i:i+20,1] - center_y[i:i+20] > 0.3) or np.all(traj[i:i+20,1] - center_y[i:i+20] < -0.3)
        if is_future_1s_deviate:
          valid_idx = i
          # print(valid_idx)
          break

      # valid segment size should always larger than LABEL_LEN+40, means least 2s valid trajectory
      if frame_cnt + valid_idx>0 and frame_cnt + valid_idx - last_valid_index > LABEL_LEN+40:
        valid_seg.append((last_valid_index, frame_cnt + valid_idx))

      # once we have a devialte, we should inhibit for least 8*20 frames, means 8s
      frame_cnt += 8*20

    # update frame counters
    frame_cnt+=1
    last_divate_status = is_ego_traj_deviate

    # plt.clf()
    # plt.plot(traj_xyz[:,0], traj_xyz[:,1])
    # plt.plot(traj_xyz[:,0], traj_xyz[:,1])

    # plt.plot(lanelines_l.x, lanelines_l.y)
    # plt.plot(lanelines_r.x, lanelines_r.y)
    # plt.plot(lanelines_ll.x, lanelines_ll.y)
    # plt.plot(lanelines_rr.x, lanelines_rr.y)
    # plt.xlim([0, 300])
    # plt.ylim([-10, 10])
    # plt.show()
    # plt.pause(0.001)


  if not is_ego_traj_deviate:
    valid_seg.append((last_valid_index, max_frame))

  print(valid_seg)

  return valid_seg

from control.lat_mpc import LatMpc
import math

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

		self.pos_err_limit = 0.8
		self.heading_err_limit = 0.1



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

		# print(self.cnt, self.error_pos, self.g_y, self.Y[self.cnt], self.g_heading, self.Heading[self.cnt] , yawrate_cmd)

		self.cnt+=1