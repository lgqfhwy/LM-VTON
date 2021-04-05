

# CUDA_VISIBLE_DEVICES=1 python viton_origin_refined_train_cloth_points.py --name RefinedGMM \
#                         --datamode train \
#                         --gpu_ids 0 \
#                         --stage GMM \
#                         --model OneRefinedGMM \
#                         --keep_step 200000 \
#                         --decay_step 200000 \
#                         --tensorboard_dir ../tensorboard_results/tensorboard_densepose_One_model_refined_gmm \
#                         --checkpoint_dir ../cp_vton_viton_results/checkpoint_densepose_One_model_refined_gmm \
#                         --dataroot /data/lgq/database/viton_resize \


# CUDA_VISIBLE_DEVICES=3 python viton_origin_refined_test_cloth_points.py --name RefinedGMM \
#                         --datamode test \
#                         --gpu_ids 0 \
#                         --stage GMM \
#                         --model OneRefinedGMM \
#                         --tensorboard_dir ../tensorboard_results/tensorboard_densepose_One_model_test_refined_gmm \
#                         --checkpoint /data/lgq/virtual_try_on/cp_vton_viton_results/checkpoint_densepose_One_model_refined_gmm/RefinedGMM/refined_gmm_final.pth \
#                         --dataroot /data/lgq/database/viton_resize \
#                         --result_dir ../cp_vton_viton_images/densepose_One_model_refined_gmm_final_400000_images

CUDA_VISIBLE_DEVICES=3 python viton_origin_refined_test_cloth_points.py --name RefinedGMM \
                        --datamode test \
                        --gpu_ids 0 \
                        --stage GMM \
                        --model OneRefinedGMM \
                        --tensorboard_dir ../tensorboard_results/tensorboard_densepose_One_model_test_refined_gmm \
                        --checkpoint /data/lgq/virtual_try_on/cp_vton_viton_results/checkpoint_densepose_One_model_refined_gmm/RefinedGMM/step_200000.pth \
                        --dataroot /data/lgq/database/viton_resize \
                        --result_dir ../cp_vton_viton_images/densepose_One_model_refined_gmm_200000_images