








# CUDA_VISIBLE_DEVICES=0,1,2,3 python origin_pix2pixhd_mpv_train.py \
#                         --phase train \
#                         --name origin_pix2pixhd_mpv \
#                         --gpu_ids 0,1,2,3 \
#                         --model origin_pix2pixHD_mpv \
#                         --no_instance \
#                         --checkpoints_dir ../pix2pix_results/checkpoint_origin_pix2pixhd_mpv_global \
#                         --dataroot /data/lgq/database/mpv \
#                         --gmm_path /data/lgq/new_graduate/results/images_new_train_cloth_points/gmm_final.pth/ \

CUDA_VISIBLE_DEVICES=0,1,2,3 python origin_pix2pixhd_mpv_train.py \
                        --phase train \
                        --name origin_pix2pixhd_mpv \
                        --gpu_ids 0,1,2,3 \
                        --model origin_pix2pixHD_mpv \
                        --no_instance \
                        --checkpoints_dir ../pix2pix_results/checkpoint_origin_pix2pixhd_mpv_local \
                        --dataroot /data/lgq/database/mpv \
                        --gmm_path /data/lgq/new_graduate/results/images_new_train_cloth_points/gmm_final.pth/ \
                        --netG local --ngf 32 --num_D 3 \
                        --load_pretrain ../pix2pix_results/checkpoint_origin_pix2pixhd_mpv_global/origin_pix2pixhd_mpv --niter_fix_global 20