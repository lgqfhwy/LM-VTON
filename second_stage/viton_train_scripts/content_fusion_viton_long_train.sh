



CUDA_VISIBLE_DEVICES=1 python content_fusion_mpv_train.py \
                        --phase train \
                        --name content_fusion_pix2pixhd_vitonn \
                        --data_list train_pairs.txt \
                        --gpu_ids 0 \
                        --model content_fusion_mpv \
                        --niter 20 \
                        --niter_decay 20 \
                        --vgg_weight 10 \
                        --no_instance \
                        --checkpoints_dir ../pix2pix_viton_results/checkpoint_long_train_viton_content_fusion_pix2pixhd_viton \
                        --dataroot /data/lgq/database/viton_resize \
                        --batchSize 1 \
                        --dataset_name content_fusion_viton \
                        --gmm_path /data/lgq/new_graduate/viton_refined_results/images_long_train_One_model_add_render_loss_refined_gmm_train_cloth_points/refined_gmm_final.pth \
                        --semantic_path /data/lgq/new_graduate/pix2pix_viton_results/images_retest_semantic_pix2pixhd_viton/semantic_pix2pixhd \


# CUDA_VISIBLE_DEVICES=2 python content_fusion_mpv_test.py \
#                         --phase test \
#                         --name content_fusion_pix2pixhd_viton \
#                         --data_list train_pairs.txt \
#                         --gpu_ids 0 \
#                         --model content_fusion_mpv \
#                         --vgg_weight 10 \
#                         --no_instance \
#                         --checkpoints_dir ../pix2pix_results/checkpoint_retrain_content_fusion_pix2pixhd_mpv \
#                         --results_dir ../pix2pix_results/images_retrain_content_fusion_pix2pixhd_mpv \
#                         --dataroot /data/lgq/database/viton_resize \
#                         --batchSize 1 \
#                         --dataset_name content_fusion_viton \
#                         --gmm_path /data/lgq/new_graduate/viton_refined_results/images_long_train_One_model_add_render_loss_refined_gmm_png_train_cloth_points/refined_gmm_final.pth \
#                         --semantic_path /data/lgq/new_graduate/pix2pix_viton_results/images_semantic_pix2pixhd_viton/semantic_pix2pixhd_mpv \
#                         --how_many 40000 \