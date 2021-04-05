



# CUDA_VISIBLE_DEVICES=1 python gmm_tps_content_fusion_mpv_train.py \
#                         --phase train \
#                         --name gmm_tps_content_fusion_pix2pixhd_mpv \
#                         --gpu_ids 0 \
#                         --model gmm_tps_vton_mpv \
#                         --vgg_weight 1 \
#                         --no_instance \
#                         --checkpoints_dir ../pix2pix_refined_results/checkpoint_gmm_tps_content_fusion_pix2pixhd_mpv \
#                         --dataroot /data/lgq/database/mpv \
#                         --batchSize 1 \
#                         --dataset_name gmm_tps_content_fusion_mpv \
#                         --gmm_path /data/lgq/new_graduate/results/images_refined_gmm_png_train_cloth_points/refined_gmm_final.pth \
#                         --semantic_path /data/lgq/new_graduate/pix2pix_results/images_semantic_pix2pixhd_mpv/semantic_pix2pixhd_mpv



CUDA_VISIBLE_DEVICES=1 python gmm_tps_content_fusion_mpv_test.py \
                        --phase test \
                        --name gmm_tps_content_fusion_pix2pixhd_mpv \
                        --gpu_ids 0 \
                        --model gmm_tps_vton_mpv \
                        --vgg_weight 1 \
                        --no_instance \
                        --checkpoints_dir ../pix2pix_refined_results/checkpoint_gmm_tps_content_fusion_pix2pixhd_mpv \
                        --results_dir ../pix2pix_refined_results/images_gmm_tps_vton_pix2pixhd_mpv \
                        --dataroot /data/lgq/database/mpv \
                        --batchSize 1 \
                        --dataset_name gmm_tps_content_fusion_mpv \
                        --gmm_path /data/lgq/new_graduate/results/images_refined_gmm_png_train_cloth_points/refined_gmm_final.pth \
                        --semantic_path /data/lgq/new_graduate/pix2pix_results/images_semantic_pix2pixhd_mpv/semantic_pix2pixhd_mpv \
                        --how_many 40000 \