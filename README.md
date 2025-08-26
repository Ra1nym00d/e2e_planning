# end2end_planning
using deep learning model to predict planning path
![Watch the video](./e2e_mdl_test.gif)
## directory tree 

  ├── ./check_label.py--------------> dataset label generator help file<br>
  ├── ./close_loop_simulator.py-----> based close loop simulator for e2e<br>
  ├── ./model_test.py---------------> test closed-loop e2e model, generater traj and follow it, then update env <br>
  ├── ./close_loop_train.py---------> closed loop training training pipeline <br>
  ├── ./common----------------------> common file, api for commaai based datasets <br>
  ├── ./control---------------------> mpc controller for path follower<br>
  ├── ./dataset---------------------> e2e main datsets directory<br>
  ├── ./diffusion-------------------> diffusion and VAE based model, useful for next world simulation model<br>
  ├── ./e2e_mdl_test.mp4------------> video for generated e2e model traj<br>
  ├── ./h5_generator.py-------------> h5py pipeline data generator, hardward-accel for training pipelie <br>
  ├── ./images----------------------> generated images for e2e trajectory.<br>
  ├── ./lib-------------------------> ulits library. no much work<br>
  ├── ./mbd-------------------------> model-based for generate e2e traj with history state(v, w)<br>
  ├── ./models----------------------> vison based models, like resnet, efficient net and transformer<br>
  ├── ./model_to_onnx.py------------> torch model to onnx or maybe tensorRt later.<br>
  ├── ./out-------------------------> model checkpoints<br>
  ├── ./README.md<br>
  ├── ./runs------------------------> tensorbord visualization runs <br>
  ├── ./tests-----------------------> tests files<br>
  ├── ./train_config.json-----------> training config, lr, optimizer and something else. <br>
  ├── ./tree.txt<br>
  ├── ./utils_comma2k19<br>
  ├── ./utils.py<br>
  └── ./wandb<br>

# for training 
python close_loop_train.py<br>


# for e2e test.
python model_to_onnx.py<br>
python model_test.py --closeloop<br>


