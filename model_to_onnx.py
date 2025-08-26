import torch
import pickle
import numpy as np
import torch
import torch.nn as nn
import re
import os

# # create temp file , after create onnx model and trt model, then remove it
# new_f_name = 'models/pred_model_ego_tmp.py'
# temp_f = open(new_f_name, 'w')


# f = open("models/pred_model_ego.py",'r')
# # con = f.read()
# new_str = ""
# for line in f:
# 	print(line)
# 	if "pred_traj_tmp[:,:,:,0:1].exp()" in line:
# 		new_str += line.replace("pred_traj_tmp[:,:,:,0:1].exp()", "pred_traj_tmp[:,:,:,0:1]")

# 	elif "pred_traj_tmp[:,:,:,1:2].sinh()" in line:
# 		new_str += line.replace("pred_traj_tmp[:,:,:,1:2].sinh()", "pred_traj_tmp[:,:,:,1:2]")

# 	else:
# 		new_str += line
# f.close()

# # print(con)
# temp_f.write(new_str)
# temp_f.close()

# os.system("mv models/pred_model_ego.py models/pred_model_ego.py.bak")
# os.system("mv models/pred_model_ego_tmp.py models/pred_model_ego.py")

# from models.pred_model_ego import PredModel
from onnxmltools.utils import float16_converter
import onnx
model_path = r'./out/planning_model.pt'                           # 模型参数路径  
model_input_ego = torch.randn(1, 50, 6).float().contiguous().cpu()

model_input_bev = torch.randn(1, 6, 128, 256).float().contiguous().cpu() 

model_input_hd_bev = torch.randn(1,  512).float().contiguous().cpu()

# model_input_hd_bev = torch.randn(1, 20, 64).float().contiguous().cpu() 


#model = LSTM(519, 9)                         # 定义模型结构，此处是我自己设计的模型
#checkpoing = torch.load(model_path)  # 导入模型参数
#model.load_state_dict(checkpoing)           # 将模型参数赋予自定义的模型


model = torch.load(model_path, map_location='cpu')  # 导入模型参数
# model = torch.load(model_path)  # 导入模型参数
model.cpu()
model.eval()

for name, p in model.named_parameters():
    print(name, p.requires_grad)
    # print()

# with torch.no_grad():
# 	# initial weights
# 	for name, m in model.named_modules():
# 		# print(name=='plan_head.6', type(m))
# 		# if isinstance(m, nn.Conv2d):
# 		#     nn.init.kaiming_normal_(m.weight, mode="fan_out")
# 		#     if m.bias is not None:
# 		#         nn.init.zeros_(m.bias)
# 		# elif isinstance(m, nn.BatchNorm2d):
# 		#     nn.init.ones_(m.weight)
# 		#     nn.init.zeros_(m.bias)
# 		# elif isinstance(m, nn.Linear) and name=='plan_head.6':
# 		#     nn.init.normal_(m.weight, 0, 0.1)
# 		#     nn.init.zeros_(m.bias)
# 		# elif isinstance(m, nn.Linear):
# 		#     nn.init.normal_(m.weight, 0, 0.01)
# 		#     nn.init.zeros_(m.bias)
# 		if isinstance(m, nn.Linear) and name=='plan_head.2':
# 			m.weight[5+33*3:5+33*3*2, :] = m.weight[5:5+33*3*1]
# 			m.weight[5+33*3*2:5+33*3*3, :] = m.weight[5:5+33*3*1]
# 			m.weight[5+33*3*3:5+33*3*4, :] = m.weight[5:5+33*3*1]
# 			m.weight[5+33*3*4:5+33*3*5, :] = m.weight[5:5+33*3*1]
        
torch.onnx.export(model, (model_input_bev , model_input_hd_bev ),  "./out/planning_model.onnx", 
						input_names=["big_imgs", "hiddenst_in"], 
						output_names=["pred_buffer"],  opset_version=15, 
						verbose=True) # 将模型保存成.onnx格式


onnx_model = onnx.load_model( "./out/planning_model.onnx")
trans_model = float16_converter.convert_float_to_float16(onnx_model,keep_io_types=True)
onnx.save_model(trans_model, "./out/planning_model_f16.onnx")


# os.system("rm models/pred_model_ego.py")
# os.system("mv models/pred_model_ego.py.bak models/pred_model_ego.py")


# # convert onnx to trt model
# os.system("trtexec --onnx=out/model_best.onnx --saveEngine=out/model_best.trt --fp16")
