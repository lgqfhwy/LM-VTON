



# CUDA_VISIBLE_DEVICES=0 python content_fusion_mpv_train.py \
#                         --phase train \
#                         --name content_fusion_pix2pixhd_mpv \
#                         --gpu_ids 0 \
#                         --model content_fusion_mpv \
#                         --vgg_weight 10 \
#                         --no_instance \
#                         --no_skin_color_and_semantic \
#                         --checkpoints_dir ../new_pix2pix_mpv_results/checkpoint_train_no_skin_color_no_semantic_content_fusion_pix2pixhd_mpv \
#                         --dataroot /data/lgq/database/mpv \
#                         --batchSize 1 \
#                         --dataset_name content_fusion_mpv \
#                         --gmm_path /data/lgq/virtual_try_on/cp_vton_mpv_images/densepose_mpv_One_model_add_point_add_vgg_refined_gmm_final_400000_images/refined_gmm_final.pth \
#                         --semantic_path /data/lgq/new_graduate/pix2pix_results/images_semantic_pix2pixhd_mpv/semantic_pix2pixhd_mpv

CUDA_VISIBLE_DEVICES=1 python content_fusion_mpv_test.py \
                        --phase test \
                        --name content_fusion_pix2pixhd_mpv \
                        --gpu_ids 0 \
                        --model content_fusion_mpv \
                        --vgg_weight 10 \
                        --no_instance \
                        --checkpoints_dir ../new_pix2pix_mpv_results/checkpoint_train_no_skin_color_no_semantic_content_fusion_pix2pixhd_mpv \
                        --results_dir ../mpv_images_test_pix2pix_mpv_results/images_train_no_skin_color_no_semantic_content_fusion_pix2pixhd_mpv \
                        --dataroot /data/lgq/database/mpv \
                        --batchSize 1 \
                        --dataset_name content_fusion_mpv \
                        --gmm_path /data/lgq/virtual_try_on/cp_vton_mpv_images/densepose_mpv_One_model_add_point_add_vgg_refined_gmm_final_400000_images/refined_gmm_final.pth  \
                        --semantic_path /data/lgq/new_graduate/pix2pix_results/images_semantic_pix2pixhd_mpv/semantic_pix2pixhd_mpv \
                        --how_many 40000 \