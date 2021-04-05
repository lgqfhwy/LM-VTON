



# CUDA_VISIBLE_DEVICES=6 python semantic_mpv_train.py \
#                         --dataset_name viton_semantic \
#                         --phase train \
#                         --name semantic_pix2pixhd \
#                         --gpu_ids 0 \
#                         --model semantic_mpv \
#                         --no_instance \
#                         --checkpoints_dir ../pix2pix_viton_results/checkpoint_retrain_semantic_pix2pixhd_viton \
#                         --dataroot /data/lgq/database/viton_resize \
#                         --batchSize 1
CUDA_VISIBLE_DEVICES=1 python semantic_mpv_test.py \
                        --dataset_name viton_semantic \
                        --phase train \
                        --name semantic_pix2pixhd \
                        --gpu_ids 0 \
                        --model semantic_mpv \
                        --no_instance \
                        --checkpoints_dir ../pix2pix_viton_results/checkpoint_semantic_pix2pixhd_viton \
                        --results_dir ../pix2pix_viton_results/images_retest_semantic_pix2pixhd_viton \
                        --dataroot /data/lgq/database/viton_resize \
                        --batchSize 1 \
                        --how_many 40000
CUDA_VISIBLE_DEVICES=1 python semantic_mpv_test.py \
                        --dataset_name viton_semantic \
                        --phase test \
                        --name semantic_pix2pixhd \
                        --gpu_ids 0 \
                        --model semantic_mpv \
                        --no_instance \
                        --checkpoints_dir ../pix2pix_viton_results/checkpoint_semantic_pix2pixhd_viton \
                        --results_dir ../pix2pix_viton_results/images_retest_semantic_pix2pixhd_viton \
                        --dataroot /data/lgq/database/viton_resize \
                        --batchSize 1 \
                        --how_many 40000                        