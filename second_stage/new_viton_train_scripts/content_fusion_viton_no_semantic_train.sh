



# CUDA_VISIBLE_DEVICES=0 python content_fusion_mpv_train.py \
#                         --phase train \
#                         --name content_fusion_pix2pixhd_vitonn \
#                         --data_list train_pairs.txt \
#                         --no_semantic \
#                         --gpu_ids 0 \
#                         --model content_fusion_mpv \
#                         --vgg_weight 10 \
#                         --no_instance \
#                         --checkpoints_dir ../pix2pix_viton_results/checkpoint_retrain_viton_no_semantic_content_fusion_pix2pixhd_viton \
#                         --dataroot /data/lgq/database/viton_resize \
#                         --batchSize 1 \
#                         --dataset_name content_fusion_viton \
#                         --gmm_path /data/lgq/virtual_try_on/cp_vton_viton_images/images_densepose_add_point_loss_One_model_refined_gmm_200000_images/step_200000.pth \
#                         --semantic_path /data/lgq/new_graduate/pix2pix_viton_results/images_retest_semantic_pix2pixhd_viton/semantic_pix2pixhd \


CUDA_VISIBLE_DEVICES=0 python content_fusion_mpv_test.py \
                        --phase test \
                        --name content_fusion_pix2pixhd_vitonn \
                        --data_list test_pairs.txt \
                        --gpu_ids 0 \
                        --model content_fusion_mpv \
                        --vgg_weight 10 \
                        --no_instance \
                         --no_semantic \
                        --checkpoints_dir ../pix2pix_viton_results/checkpoint_retrain_viton_no_semantic_content_fusion_pix2pixhd_viton \
                        --results_dir ../pix2pix_viton_results/images_long_no_semantic_train_One_model_add_render_loss_refined_gmm_train_cloth_points \
                        --dataroot /data/lgq/database/viton_resize \
                        --batchSize 1 \
                        --dataset_name content_fusion_viton \
                        --gmm_path /data/lgq/virtual_try_on/cp_vton_viton_images/images_densepose_add_point_loss_One_model_refined_gmm_200000_images/step_200000.pth \
                        --semantic_path /data/lgq/new_graduate/pix2pix_viton_results/images_retest_semantic_pix2pixhd_viton/semantic_pix2pixhd \
                        --how_many 40000 \

CUDA_VISIBLE_DEVICES=0 python content_fusion_mpv_test.py \
                        --phase test \
                        --name content_fusion_pix2pixhd_vitonn \
                        --data_list test_pairs.txt \
                        --gpu_ids 0 \
                        --model content_fusion_mpv \
                        --vgg_weight 10 \
                        --no_instance \
                        --no_semantic \
                        --checkpoints_dir ../pix2pix_viton_results/checkpoint_retrain_viton_no_semantic_content_fusion_pix2pixhd_viton \
                        --results_dir ../pix2pix_viton_results/images_long_pair_no_semantic_train_One_model_add_render_loss_refined_gmm_train_cloth_points \
                        --dataroot /data/lgq/database/viton_resize \
                        --batchSize 1 \
                        --dataset_name content_fusion_viton \
                        --gmm_path /data/lgq/virtual_try_on/cp_vton_viton_images/images_densepose_add_point_add_vgg_warped_mask_pair_One_model_refined_gmm_final_400000_images/refined_gmm_final.pth \
                        --semantic_path /data/lgq/new_graduate/pix2pix_viton_results/images_retest_semantic_pix2pixhd_viton/semantic_pix2pixhd \
                        --how_many 40000 \
                        --is_pair \