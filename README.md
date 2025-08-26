# end2end_planning
using deep learning model to predict planning path

# directory tree 
.
├── ./check_label.py              # dataset label generator help file
├── ./close_loop_simulator.py     # based close loop simulator for e2e
├── ./model_test.py               # test closed-loop e2e model, generater traj and follow it, then update env
├── ./close_loop_train.py         # closed loop training training pipeline
├── ./common                      # common file, api for commaai based datasets
├── ./control                     # mpc controller for path follower
├── ./dataset                     # e2e main datsets directory
├── ./diffusion                   # diffusion and VAE based model, useful for next world simulation model
├── ./e2e_mdl_test.avi            # video for generated e2e model traj
├── ./h5_generator.py             # h5py pipeline data generator, hardward-accel for training pipelie 
├── ./images                      # generated images for e2e trajectory.
├── ./lib                         # ulits library. no much work
├── ./mbd                         # model-based for generate e2e traj with history state(v, w)
├── ./models                      # vison based models, like resnet, efficient net and transformer
├── ./model_to_onnx.py            # torch model to onnx or maybe tensorRt later.
├── ./out                         # model checkpoints
├── ./README.md
├── ./runs                        # tensorbord visualization runs 
├── ./tests                       # tests files
├── ./train_config.json           # training config, lr, optimizer and something else.
├── ./tree.txt
├── ./utils_comma2k19
├── ./utils.py
└── ./wandb

# for training 
python close_loop_train.py


# for e2e test.
python model_to_onnx.py
python model_test.py --closeloop


![Watch the video](./e2e_mdl_test.avi)
