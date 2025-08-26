# import  cv2
# import time
# test_path = "/media/michael/a944f99c-b5cc-44fb-9ac8-d0b263759080/home/michael/Desktop/lcq__work/end2end_lateral/dataset/realdata/training/zs_c3_data/08131/2023-08-13--15-41-23--32/fcamera.hevc" #32
# cap = cv2.VideoCapture(test_path)
# gpu_output = cv2.cuda_GpuMat()
# st = time.time()
# while True:
#   gpu_img = cv2.cuda_GpuMat()
#   st = time.time()
#   ret, img = cap.read()
#   gpu_img.upload(img)
#   print((time.time() - st)*1)
#   if not ret:
#     break


import cv2
import time

# 用CPU读取视频
def read_video_cpu(video_path):
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # 在这里进行你希望执行的操作
        # ...
    cap.release()


# 用GPU读取视频
def read_video_gpu(video_path):
    if not cv2.cuda.getCudaEnabledDeviceCount():
        print("CUDA is not available. Please make sure CUDA drivers are installed.")
        return

    # gpu_id = 0
    # device = cv2.cuda.Device(gpu_id)
    # ctx = device.createContext()

    cap = cv2.VideoCapture(video_path)

    cap.set(cv2.CAP_PROP_CUDA_MPS, 1)

    while True:
        ret, frame = cap.read(cv2.CAP_CUDA)

        if not ret:
            break

        # 在这里进行你希望执行的操作
        # ...

    cap.release()


# 视频文件路径
video_path = "path/to/video/file.mp4"

read_video_gpu(video_path)

