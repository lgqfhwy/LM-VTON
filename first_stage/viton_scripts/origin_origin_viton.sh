


# CUDA_VISIBLE_DEVICES=3 python train.py --name gmm_train_new \
#                                        --stage GMM \
#                                        --save_count 5000 \
#                                        --workers 4 \
#                                        --dataroot /data/lgq/database/viton_resize \
#                                        --tensorboard_dir ../tensorboard_results/tensorboard_origin_origin_cp_gmm \
#                                        --checkpoint_dir ../cp_vton_viton_results/checkpoint_origin_origin_cp_gmm \

# CUDA_VISIBLE_DEVICES=3 python test.py --name gmm_train_new \
#                                        --stage GMM \
#                                        --datamode train \
#                                        --data_list train_pairs.txt \
#                                        --workers 4 \
#                                        --dataroot /data/lgq/database/viton_resize \
#                                        --tensorboard_dir ../tensorboard_results/tensorboard_test_origin_origin_cp_gmm \
#                                        --checkpoint ../cp_vton_viton_results/checkpoint_origin_origin_cp_gmm/gmm_train_new/gmm_final.pth \
#                                        --result_dir ../cp_vton_viton_images/origin_origin_cp_vtion_gmm

CUDA_VISIBLE_DEVICES=2 python train.py --name tom_train_new \
                                       --stage TOM \
                                       --save_count 50000 \
                                       --workers 4 \
                                       --dataroot /data/lgq/database/viton_resize \
                                       --tensorboard_dir ../tensorboard_results/tensorboard_origin_origin_cp_tom \
                                       --checkpoint_dir ../cp_vton_viton_results/checkpoint_origin_origin_cp_tom \
                                       --warp_path /data/lgq/virtual_try_on/cp_vton_viton_images/origin_origin_cp_vtion_gmm/gmm_final.pth