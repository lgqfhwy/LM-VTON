



# CUDA_VISIBLE_DEVICES=0,1 python semantic_mpv_train.py \
#                         --phase train \
#                         --name semantic_pix2pixhd_mpv \
#                         --gpu_ids 0 \
#                         --model semantic_mpv \
#                         --no_instance \
#                         --checkpoints_dir ../pix2pix_results/checkpoint_no_background_semantic_pix2pixhd_mpv \
#                         --dataroot /data/lgq/database/mpv \
#                         --batchSize 1 \
#                         --no_background

CUDA_VISIBLE_DEVICES=0,1 python semantic_mpv_test.py \
                        --phase test \
                        --name semantic_pix2pixhd_mpv \
                        --gpu_ids 0 \
                        --model semantic_mpv \
                        --no_instance \
                        --checkpoints_dir ../pix2pix_results/checkpoint_no_background_semantic_pix2pixhd_mpv \
                        --results_dir ../pix2pix_results/images_no_background_semantic_pix2pixhd_mpv \
                        --dataroot /data/lgq/database/mpv \
                        --batchSize 1 \
                        --how_many 40000 \
                        --no_background