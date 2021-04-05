



CUDA_VISIBLE_DEVICES=0 python semantic_mpv_train.py \
                        --phase train \
                        --name semantic_pix2pixhd_mpv \
                        --gpu_ids 0 \
                        --model semantic_mpv \
                        --no_instance \
                        --niter 20 \
                        --niter_decay 20 \
                        --checkpoints_dir ../pix2pix_mpv_results/checkpoint_long_train_semantic_pix2pixhd_mpv \
                        --dataroot /data/lgq/database/mpv \
                        --batchSize 1
# CUDA_VISIBLE_DEVICES=1 python semantic_mpv_test.py \
#                         --phase test \
#                         --name semantic_pix2pixhd_mpv \
#                         --gpu_ids 0 \
#                         --model semantic_mpv \
#                         --no_instance \
#                         --checkpoints_dir ../pix2pix_results/checkpoint_semantic_pix2pixhd_mpv \
#                         --results_dir ../debug_results/images_semantic_pix2pixhd_mpv \
#                         --dataroot /data/lgq/database/mpv \
#                         --batchSize 1 \
#                         --how_many 40000