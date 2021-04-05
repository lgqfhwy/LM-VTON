



# CUDA_VISIBLE_DEVICES=1 python content_fusion_mpv_train.py \
#                         --phase train \
#                         --name attention_content_fusion_pix2pixhd_mpv \
#                         --gpu_ids 0 \
#                         --model attention_content_fusion_mpv \
#                         --vgg_weight 10 \
#                         --no_instance \
#                         --block_version v3 \
#                         --checkpoints_dir ../new_pix2pix_results/checkpoint_discriminator_attention_v3_all_content_fusion_pix2pixhd_mpv \
#                         --dataroot /data/lgq/database/mpv \
#                         --batchSize 1 \
#                         --dataset_name content_fusion_mpv \
#                         --gmm_path /data/lgq/new_graduate/new_cp_vton_results/images_new_atten_v3_refined_gmm_png_train_cloth_points/atten_refined_gmm_final.pth \
#                         --semantic_path /data/lgq/new_graduate/pix2pix_results/images_semantic_pix2pixhd_mpv/semantic_pix2pixhd_mpv


CUDA_VISIBLE_DEVICES=1 python content_fusion_mpv_test.py \
                        --phase test \
                        --name attention_content_fusion_pix2pixhd_mpv \
                        --gpu_ids 0 \
                        --model attention_content_fusion_mpv \
                        --vgg_weight 10 \
                        --no_instance \
                        --block_version v3 \
                        --checkpoints_dir ../new_pix2pix_results/checkpoint_discriminator_attention_v3_all_content_fusion_pix2pixhd_mpv \
                        --results_dir ../new_pix2pix_results/images_discriminator_new_attention_v3_all_content_fusion_pix2pixhd_mpv \
                        --dataroot /data/lgq/database/mpv \
                        --batchSize 1 \
                        --dataset_name content_fusion_mpv \
                        --gmm_path /data/lgq/new_graduate/new_cp_vton_results/images_new_atten_v3_refined_gmm_png_train_cloth_points/atten_refined_gmm_final.pth \
                        --semantic_path /data/lgq/new_graduate/pix2pix_results/images_semantic_pix2pixhd_mpv/semantic_pix2pixhd_mpv \
                        --how_many 40000 \