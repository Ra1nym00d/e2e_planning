"""
This file is named after `dask` for historical reasons. We first tried to
use dask to coordinate the hdf5 buckets but it was slow and we wrote our own
stuff.
"""
import numpy as np
import h5py
import time
import logging
import traceback
import glob
from tqdm import tqdm
import random
import os
import logging
import numpy
import zmq
from numpy.lib.format import header_data_from_array_1_0
import cv2
import six


from common.transformations.camera import get_view_frame_from_road_frame
import laika.lib.coordinates as coord
import laika.lib.orientation as orient
from utils import get_poses, reshape_yuv, get_hevc_np, get_calib_extrinsic
from utils import MEDMDL_INSMATRIX, C3_INSMATRIX, C2_INSMATRIX, \
                  ANCHOR_TIME, TRAJECTORY_TIME, LABEL_LEN

if six.PY3:
  buffer_ = memoryview
else:
  buffer_ = buffer  # noqa

logger = logging.getLogger(__name__)


def send_arrays(socket, arrays, stop=False):
  """Send NumPy arrays using the buffer interface and some metadata.

  Parameters
  ----------
  socket : :class:`zmq.Socket`
  The socket to send data over.
  arrays : list
  A list of :class:`numpy.ndarray` to transfer.
  stop : bool, optional
  Instead of sending a series of NumPy arrays, send a JSON object
  with a single `stop` key. The :func:`recv_arrays` will raise
  ``StopIteration`` when it receives this.

  Notes
  -----
  The protocol is very simple: A single JSON object describing the array
  format (using the same specification as ``.npy`` files) is sent first.
  Subsequently the arrays are sent as bytestreams (through NumPy's
  support of the buffering protocol).

  """
  if arrays:
    # The buffer protocol only works on contiguous arrays
    arrays = [numpy.ascontiguousarray(array) for array in arrays]
  if stop:
    headers = {'stop': True}
    socket.send_json(headers)
  else:
    headers = [header_data_from_array_1_0(array) for array in arrays]
    socket.send_json(headers, zmq.SNDMORE)
    for array in arrays[:-1]:
      socket.send(array, zmq.SNDMORE)
    socket.send(arrays[-1])


def recv_arrays(socket):
  """Receive a list of NumPy arrays.

  Parameters
  ----------
  socket : :class:`zmq.Socket`
  The socket to receive the arrays on.

  Returns
  -------
  list
  A list of :class:`numpy.ndarray` objects.

  Raises
  ------
  StopIteration
  If the first JSON object received contains the key `stop`,
  signifying that the server has finished a single epoch.

  """
  headers = socket.recv_json()
  if 'stop' in headers:
    raise StopIteration
  arrays = []
  for header in headers:
    data = socket.recv()
    buf = buffer_(data)
    array = numpy.frombuffer(buf, dtype=numpy.dtype(header['descr']))
    array.shape = header['shape']
    if header['fortran_order']:
      array.shape = header['shape'][::-1]
      array = array.transpose()
    arrays.append(array)
  return arrays


def client_generator(port=5557, host="localhost", hwm=20):
  """Generator in client side should extend this generator

  Parameters
  ----------

  port : int
  hwm : int, optional
  The `ZeroMQ high-water mark (HWM)
  <http://zguide.zeromq.org/page:all#High-Water-Marks>`_ on the
  sending socket. Increasing this increases the buffer, which can be
  useful if your data preprocessing times are very random.  However,
  it will increase memory usage. There is no easy way to tell how
  many batches will actually be queued with a particular HWM.
  Defaults to 10. Be sure to set the corresponding HWM on the
  receiving end as well.
  """
  context = zmq.Context()
  socket = context.socket(zmq.REQ)
  socket.set_hwm(hwm)
  socket.connect("tcp://{}:{}".format(host, port))
  logger.info('client started')
  while True:
    socket.send(b"A message")
    data = recv_arrays(socket)
    yield tuple(data)


def start_server(data_stream, port=5557, hwm=20):
  """Start a data processing server.

  This command starts a server in the current process that performs the
  actual data processing (by retrieving data from the given data stream).
  It also starts a second process, the broker, which mediates between the
  server and the client. The broker also keeps a buffer of batches in
  memory.

  Parameters
  ----------
  data_stream : generator
  The data stream to return examples from.
  port : int, optional
  The port the server and the client (training loop) will use to
  communicate. Defaults to 5557.
  hwm : int, optional
  The `ZeroMQ high-water mark (HWM)
  <http://zguide.zeromq.org/page:all#High-Water-Marks>`_ on the
  sending socket. Increasing this increases the buffer, which can be
  useful if your data preprocessing times are very random.  However,
  it will increase memory usage. There is no easy way to tell how
  many batches will actually be queued with a particular HWM.
  Defaults to 10. Be sure to set the corresponding HWM on the
  receiving end as well.
  """
  logging.basicConfig(level='INFO')

  context = zmq.Context()
  socket = context.socket(zmq.REP)
  socket.set_hwm(hwm)
  socket.bind('tcp://*:{}'.format(port))

  # it = itertools.tee(data_stream)
  it = data_stream

  logger.info('server started')
  while True:
    try:
      data = next(it)
      stop = False
      logger.debug("sending {} arrays".format(len(data)))
    except StopIteration:
      it = data_stream
      data = None
      stop = True
      logger.debug("sending StopIteration")
    message = socket.recv()
    send_arrays(socket, data, stop=stop)

    
# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
first = True


from utils import valid_segment_slice
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait, ALL_COMPLETED
import threading
cv2.ocl.setUseOpenCL(True)


def get_data(video_sets, bat_labels, calib_msgs, bat_bias, batch_models, batch_size, seq_len):
  def task(video_name, gt_label, calib_msg, bat_bia, batch_model):
    # convert h265 video to image numpy buffer
    hevc_np = get_hevc_np(video_name)

    # label data
    frame_positions = gt_label[0]
    frame_orientations = gt_label[1]
    frame_velocities = gt_label[2]
    frame_angular_speed = gt_label[3]
    bev_d = np.zeros((seq_len+1, 6, 128, 256), dtype=np.uint8)
    label_d = np.zeros((seq_len+1, 33,3), dtype=np.float32)
    pose_d = np.zeros((seq_len+1, 32), dtype=np.float32)

    # # first we check about lane change or lane diviate , temp we not trainging this data, can make model confusing
    # valid_seg = valid_segment_slice(batch_model, frame_orientations, frame_positions, seq_len+LABEL_LEN)
    # print(valid_seg)
    valid_seg = [(0, 1101)]

    # index
    frame_cnt = 0

    # random flip operation
    is_flip = np.random.rand() > 0.5
    while frame_cnt<seq_len+1:
      img = cv2.cvtColor(hevc_np[frame_cnt], cv2.COLOR_YUV2BGR_NV12)
      # get feed model image with yuv format
      img = cv2.warpPerspective(src=img, M=calib_msg, dsize=(512,256), flags=cv2.WARP_INVERSE_MAP)
      # cv2.imshow("test", img)
      # cv2.waitKey(10)
      # img = img[::2,::2,:]
      if is_flip:
        img = img[:,::-1,:] #[h,w,3]
      img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420)
      img = reshape_yuv(img)
      bev_d[frame_cnt] = img

      ## get label trajectory
      ecef_from_local = orient.rot_from_quat(frame_orientations[frame_cnt])
      local_from_ecef = ecef_from_local.T
      frame_positions_local = np.einsum('ij,kj->ki', local_from_ecef, 
                                        frame_positions - frame_positions[frame_cnt]).astype(np.float32)

      traj_xyz = frame_positions_local[frame_cnt:frame_cnt+LABEL_LEN]
      
      label_position_x = np.interp(ANCHOR_TIME, TRAJECTORY_TIME, traj_xyz[:,0]).reshape(33,1)

      direction = -1. if is_flip else 1.

      label_position_y = (np.interp(ANCHOR_TIME, TRAJECTORY_TIME, traj_xyz[:,1]).reshape(33,1)- bat_bia) * direction

      # print(label_position_y)
      label_position_z = np.interp(ANCHOR_TIME, TRAJECTORY_TIME, traj_xyz[:,2]).reshape(33,1)

      label_traj_xyz = np.hstack((label_position_x, label_position_y, label_position_z))
      label_d[frame_cnt] = label_traj_xyz


      frame_velocities_local = np.einsum('ij,kj->ki', local_from_ecef, 
                                        frame_velocities).astype(np.float32)


      # xyz velocity
      pose_d[frame_cnt][0:3] = frame_velocities_local[frame_cnt] 
      # print(frame_angular_speed[frame_cnt])

      # xyz angular speed, rad 2 deg
      pose_d[frame_cnt][3:6] = (180./np.pi) * frame_angular_speed[frame_cnt] 

      # angular should also reverse
      pose_d[frame_cnt][5] = pose_d[frame_cnt][5]*direction # wz
      pose_d[frame_cnt][3] = pose_d[frame_cnt][3]*direction # wx

      # vy should reverse, 0 is vx, 1 is vy, 2 is vz
      pose_d[frame_cnt][1] = pose_d[frame_cnt][1]*direction 

      # print(frame_velocities_local[frame_cnt])

      # # check data converage
      if np.any( np.abs(label_position_y) > 100.):
        # print()
        raise Exception("label data is not valid")

      # xyz 
      frame_cnt +=1

    return [bev_d, label_d, pose_d, valid_seg]

  # multiprocess for batchsize video decoder
  start_time = time.time()
  t = ThreadPoolExecutor(2)
  cmds = []
  dd=[]

  # print(bat_bias)
  for i in range(batch_size):
    cmds.append((video_sets[i], bat_labels[i], calib_msgs[i], bat_bias[i], batch_models[i]))

  for cmd in cmds:
    dd.append(t.submit(task, cmd[0], cmd[1], cmd[2], cmd[3], cmd[4]))

  # block until the all threading task done!
  wait(dd, return_when=ALL_COMPLETED)
  print(f'one pack cost time：{time.time()-start_time}')
  return [d.result() for d in dd]


def datagen(datadir, time_len=1, batch_size=6, seq_len=900, ignore_goods=False):
  """
  Parameters:
  -----------
  leads : bool, should we use all x, y and speed radar leads? default is false, uses only x
  """
  global first
  assert time_len > 0

  # get h5 buckert from datadir
  # filter_names = glob.glob(f"{datadir}/*/*/*/*.h5") 
  video_names = glob.glob(f"{datadir}/*/*/*/fcamera.hevc") + glob.glob(f"{datadir}/*/*/*/video.hevc")

  logger.info("Loading {} hdf5 buckets.".format(len(video_names)))
  print(video_names)

  # create segments sample lists
  h5_datasets = []
  print("start loading dataset-")

  files_length = len(video_names)

  X_batch    = np.zeros((batch_size, seq_len+1, 6, 128, 256), dtype='uint8')
  traj_batch = np.zeros((batch_size, seq_len+1, 33, 3), dtype='float32')
  pose_batch = np.zeros((batch_size, seq_len+1, 32), dtype='float32')
  valid_seg_batch = []

  sample_list = [i for i in range(files_length)]
  logger.info(f"total file length is : {files_length}" )
  logger.info(f"steps with one epoch is : {int(files_length/batch_size)}")

  while files_length>0:
    try:
      t = time.time()

      count = 0
      start = time.time()

      if len(sample_list) < batch_size:
        print(len(sample_list))
        sample_list = [i for i in range(files_length)]

      batch_index = []
      batch_calib = []
      batch_label = []
      batch_bias = []
      batch_model = []

      while count < batch_size:
        ## we give bias
        rand_bias = np.random.normal(loc=0.0, scale=0.3)

        # get one segments
        file_index = random.sample(sample_list, 1)[0]
        sample_list.remove(file_index)

        # get names
        video_name = video_names[file_index]
        tmp1 = video_name.split('/')

        tmp1.pop()
        seg_path = '/'.join(tmp1)
        print(seg_path)

        # if we can not find the log file ,just continue
        if not os.path.exists(seg_path+'/rlog')  and  not os.path.exists(seg_path+'/raw_log.bz2'):
          continue

        is_comma2k19 = 'Chunk_cm' in video_name

        if not is_comma2k19:
          # should we have a datacheck, such as segments length and gps valid 
          frame_positions, frame_orientations, frame_velocities, \
          calib_msgs , statuses, frame_angular_velocities, model_msgs = get_poses(seg_path)
          frame_valid =  (np.all(statuses['gpsOK'] >0)      and np.all(statuses['orientations_calib'] >0)  \
                     and  np.all(statuses['positions'] >0)  and np.all(statuses['status'] >0) and statuses['gpsOK'].shape[0]>1050)

          ang_x = np.mean(calib_msgs[:,0])
          ang_y = np.mean(calib_msgs[:,1])
          ang_z = np.mean(calib_msgs[:,2])

          # print(get_view_frame_from_road_frame(ang_x, ang_y, ang_z, 1.32, -0.06 + rand_bias))
          camera_frame_from_ground = np.dot(C3_INSMATRIX,
                                              get_view_frame_from_road_frame(ang_x, ang_y, ang_z, 1.32, -0.06 + rand_bias))[:, (0, 1, 3)]
          calib_frame_from_ground = np.dot(MEDMDL_INSMATRIX,
                                              get_view_frame_from_road_frame(0, 0, 0, 1.22))[:, (0, 1, 3)]
          calib_frame_from_camera_frame = np.dot(camera_frame_from_ground, np.linalg.inv(calib_frame_from_ground))

          # goto next valid dataset
          if not frame_valid and statuses['gpsOK'].shape[0]<seq_len+LABEL_LEN:
            continue 

        else:
          # st = time.time()
          extrinsic_matrix, model_msgs = get_calib_extrinsic(seg_path, logfile='raw_log.bz2') 
          # print(extrinsic_matrix, )
          # add noise bias 
          extrinsic_matrix[0,3] += -1*rand_bias
          # print(extrinsic_matrix)

          camera_frame_from_ground = np.dot(C2_INSMATRIX, extrinsic_matrix[:, (0, 1, 3)] )
          calib_frame_from_ground = np.dot(MEDMDL_INSMATRIX,
                                              get_view_frame_from_road_frame(0, 0, 0, 1.22))[:, (0, 1, 3)]
          calib_frame_from_camera_frame = np.dot(camera_frame_from_ground, np.linalg.inv(calib_frame_from_ground))

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


          ## if the sequeece length is less then 1100, goto next valid dataset
          if frame_positions.shape[0]<seq_len+LABEL_LEN:
            continue 

        # ecef_from_local = orient.rot_from_quat(frame_orientations[0])
        # local_from_ecef = ecef_from_local.T
        # frame_velocities_local = np.einsum('ij,kj->ki', local_from_ecef, 
        #                                   frame_velocities).astype(np.float32)
        
        # print(frame_velocities_local[0])
        # print(frame_positions - frame_positions[0])
        batch_index.append(file_index)
        batch_calib.append(calib_frame_from_camera_frame)
        batch_label.append([frame_positions, frame_orientations, frame_velocities, frame_angular_velocities])
        batch_bias.append(rand_bias)
        batch_model.append(model_msgs)
        count += 1

      ## get model feed and label data
      video_sets = [video_names[idx] for idx in batch_index]
      dd = get_data(video_sets, batch_label, batch_calib, batch_bias, batch_model, batch_size, seq_len)
      for i in range(batch_size):
        X_batch[i]    = dd[i][0]
        traj_batch[i] = dd[i][1]
        pose_batch[i] = dd[i][2]
        valid_seg_batch.append(dd[i][3])

      # sanity check
      assert X_batch.shape == (batch_size, seq_len+1, 6, 128, 256)

      # logging.info("load image : {}s".format(time.time()-t))
      print("load image : %5.2f ms" % ((time.time()-start)*1000.0), batch_index, len(sample_list))

      if first:
        print("X", X_batch.shape)
        print("angle", traj_batch.shape) 
        first = False

      yield (X_batch, traj_batch, pose_batch, len(sample_list) < batch_size , files_length)

    except KeyboardInterrupt:
      raise
    except:
      traceback.print_exc()
      pass


