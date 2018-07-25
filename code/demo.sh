# RDN BI model (x2)
#python3.6 main.py --scale 2 --save RDN_D16C8G64_BIx2 --model RDN --epochs 200 --batch_size 16 --n_val 5 --patch_size 64 --reset
# RDN BI model (x3)
#python3.6 main.py --scale 3 --save RDN_D16C8G64_BIx3 --model RDN --epochs 200 --batch_size 16 --n_val 5 --patch_size 96 --reset
# RDN BI model (x4)
#python3.6 main.py --scale 4 --save RDN_D16C8G64_BIx4 --model RDN --epochs 200 --batch_size 16 --n_val 5 --patch_size 128 --reset

# EDSR baseline model (x2)
#python main.py --model EDSR --scale 2 --save EDSR_baseline_x2 --reset

# EDSR baseline model (x3) - from EDSR baseline model (x2)
#python main.py --model EDSR --scale 3 --save EDSR_baseline_x3 --reset --pre_train ../experiment/model/EDSR_baseline_x2.pt

# EDSR baseline model (x4) - from EDSR baseline model (x2)
#python main.py --model EDSR --scale 4 --save EDSR_baseline_x4 --reset --pre_train ../experiment/model/EDSR_baseline_x2.pt

# EDSR in the paper (x2)
#python main.py --model EDSR --scale 2 --save EDSR_x2 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset

# EDSR in the paper (x3) - from EDSR (x2)
#python main.py --model EDSR --scale 3 --save EDSR_x3 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset --pre_train ../experiment/EDSR_x2/model/model_best.pt

# EDSR in the paper (x4) - from EDSR (x2)
#python main.py --model EDSR --scale 4 --save EDSR_x4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset --pre_train ../experiment/EDSR_x2/model/model_best.pt

# MDSR baseline model
#python main.py --template MDSR --model MDSR --scale 2+3+4 --save MDSR_baseline --reset --save_models

# MDSR in the paper
#python main.py --template MDSR --model MDSR --scale 2+3+4 --n_resblocks 80 --save MDSR --reset --save_models

# Standard benchmarks (Ex. EDSR_baseline_x4)
#python main.py --data_test Set5 --scale 4 --pre_train ../experiment/model/EDSR_baseline_x4.pt --test_only --self_ensemble
#python main.py --data_test Set14 --scale 4 --pre_train ../experiment/model/EDSR_baseline_x4.pt --test_only --self_ensemble
#python main.py --data_test B100 --scale 4 --pre_train ../experiment/model/EDSR_baseline_x4.pt --test_only --self_ensemble
#python main.py --data_test Urban100 --scale 4 --pre_train ../experiment/model/EDSR_baseline_x4.pt --test_only --self_ensemble
#python main.py --data_test DIV2K --ext img --n_val 100 --scale 4 --pre_train ../experiment/model/EDSR_baseline_x4.pt --test_only --self_ensemble

#python main.py --data_test Set5 --scale 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --pre_train ../experiment/model/EDSR_x4.pt --test_only --self_ensemble
#python main.py --data_test Set14 --scale 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --pre_train ../experiment/model/EDSR_x4.pt --test_only --self_ensemble
#python main.py --data_test B100 --scale 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --pre_train ../experiment/model/EDSR_x4.pt --test_only --self_ensemble
#python main.py --data_test Urban100 --scale 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --pre_train ../experiment/model/EDSR_x4.pt --test_only --self_ensemble
#python main.py --data_test DIV2K --ext img --n_val 100 --scale 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --pre_train ../experiment/model/EDSR_x4.pt --test_only --self_ensemble

# Test your own images
#python3 main.py --data_test Demo --scale 4 --pre_train ../experiment/model/EDSR_baseline_x4.pt --test_only --save_results
#python3 main.py --scale 4 --loss 1*L1+1*MSE+1*GAN --pre_train ../experiment/model/EDSR_x4.pt --save_results --print_every 10 --n_GPUs 6 --batch_size 32 --save EDSR_custom_3 --lr 2.5e-5 --n_feats 256 --res_scale 0.1 --n_resblocks 32 --chop



##first one to run
#python3 main.py --scale 4 --loss 1*L1+1*MSE+0.001*APRX-GAN --pre_train ../experiment/model/EDSR_x4.pt --save_results --print_every 10 --n_GPUs 5 --batch_size 8 --save APRX_trainer_test --lr 2.5e-6 --n_feats 256 --res_scale 0.1 --n_resblocks 32 --chop --aprx_training_dir /usr/project/xtmp/superresoluter/EDSR_patches 

##second one to run
#python3 main.py --scale 4 --loss 1*L1+1*MSE+200*GAN --load ../experiment/APRX_trainer_test --save_results --print_every 10 --n_GPUs 5 --batch_size 8 --save APRX_user_test --lr 2.5e-6 --n_feats 256 --res_scale 0.1 --n_resblocks 32 --chop

##combined one to run
#python3 main.py --scale 4 --loss 1*L1+1*MSE+1*GAN --pre_train ../experiment/model/EDSR_x4.pt --save_results --print_every 10 --n_GPUs 5 --batch_size 8 --save APRX_trainer_test --lr 2.5e-6 --n_feats 256 --res_scale 0.1 --n_resblocks 32 --chop --aprx_training_dir /usr/project/xtmp/superresoluter/EDSR_patches --aprx_epochs 100
#python3 main.py --scale 4 --loss 1*L1+1*MSE+5*GAN --pre_train ../experiment/model/EDSR_x4.pt --save_results --print_every 30 --n_GPUs 5 --batch_size 16 --save joint_test_d1 --lr 1e-9 --n_feats 256 --res_scale 0.1 --n_resblocks 32 --chop --aprx_training_dir /usr/project/xtmp/superresoluter/EDSR_patches --aprx_epochs 1200 --save_models

##test only one to run
python3 main.py --scale 4 --pre_train ../experiment/joint_test_c1/model/model_latest.pt --test_only --save_results --n_GPUs 3 --batch_size 8 --lr 2.5e-6 --n_feats 256 --res_scale 0.1 --n_resblocks 32 --chop --data_test Demo



# Advanced - Test with JPEG images 
#python main.py --model MDSR --data_test Demo --scale 2+3+4 --pre_train ../experiment/model/MDSR_baseline_jpeg.pt --test_only --save_results

# Advanced - Training with adversarial loss
#python main.py --template GAN --scale 4 --save EDSR_GAN --reset --patch_size 96 --loss 5*VGG54+0.15*GAN --pre_train ../experiment/model/EDSR_baseline_x4.pt

# RCAN_BIX2_G10R20P48, input=48x48, output=96x96
# pretrained model can be downloaded from https://www.dropbox.com/s/mjbcqkd4nwhr6nu/models_ECCV2018RCAN.zip?dl=0
#python main.py --model RCAN --save RCAN_BIX2_G10R20P48 --scale 2 --n_resgroups 10 --n_resblocks 20 --n_feats 64  --reset --chop --save_results --print_model --patch_size 96
# RCAN_BIX3_G10R20P48, input=48x48, output=144x144
#python main.py --model RCAN --save RCAN_BIX3_G10R20P48 --scale 3 --n_resgroups 10 --n_resblocks 20 --n_feats 64  --reset --chop --save_results --print_model --patch_size 144 --pre_train ../experiment/model/RCAN_BIX2.pt
# RCAN_BIX4_G10R20P48, input=48x48, output=192x192
#python main.py --model RCAN --save RCAN_BIX4_G10R20P48 --scale 4 --n_resgroups 10 --n_resblocks 20 --n_feats 64  --reset --chop --save_results --print_model --patch_size 192 --pre_train ../experiment/model/RCAN_BIX2.pt
# RCAN_BIX8_G10R20P48, input=48x48, output=384x384
#python main.py --model RCAN --save RCAN_BIX8_G10R20P48 --scale 8 --n_resgroups 10 --n_resblocks 20 --n_feats 64  --reset --chop --save_results --print_model --patch_size 384 --pre_train ../experiment/model/RCAN_BIX2.pt
