#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import os
import time
from mpv_cloth_points_dataset import CPDataset, CPDataLoader
from networks import GMM, UnetGenerator, load_checkpoint
from multi_networks import RefinedGMM, OneRefinedGMM
from tensorboardX import SummaryWriter
from visualization import board_add_image, board_add_images, save_images

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default = "GMM")
    parser.add_argument("--gpu_ids", default = "")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    parser.add_argument('--is_round', type = bool, default = True)
    parser.add_argument("--dataroot", default = "data")
    parser.add_argument("--datamode", default = "train")
    parser.add_argument("--stage", default = "GMM")
    parser.add_argument("--data_list", default = "train_pairs.txt")
    parser.add_argument("--densepose_path", default="/data/lgq/database/mpv/densepose_mpv_shape")
    parser.add_argument("--fine_width", type=int, default = 192)
    parser.add_argument("--fine_height", type=int, default = 256)
    parser.add_argument("--radius", type=int, default = 5)
    parser.add_argument("--grid_size", type=int, default = 5)
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--result_dir', type=str, default='result', help='save result infos')
    parser.add_argument('--checkpoint', type=str, default='', help='model checkpoint for test')
    parser.add_argument("--display_count", type=int, default = 1)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
    parser.add_argument("--gmm_path", type = str, default = "../result")
    parser.add_argument('--loss_weight', type = float, default = 0.01)
    parser.add_argument("--encoder_name", type = str, default="resnet34")
    parser.add_argument("--gmm_feature_A_input_nc", type = int, default=22)
    parser.add_argument("--semantic_path", type = str, default = "/data/lgq/new_graduate/pix2pix_results/images_semantic_pix2pixhd_mpv/semantic_pix2pixhd_mpv")
    parser.add_argument("--add_render_loss", action = 'store_true')
    parser.add_argument("--model", type = str, default = 'RefinedGMM')
    parser.add_argument("--add_origin_cloth", action = 'store_true')
    opt = parser.parse_args()
    return opt

def test_no_background_refined_gmm(opt, test_loader, model, board):
    model.cuda()
    model.eval()

    base_name = os.path.basename(opt.checkpoint)
    save_dir = os.path.join(opt.result_dir, base_name, opt.datamode)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    warp_cloth_dir = os.path.join(save_dir, 'warp-cloth')
    if not os.path.exists(warp_cloth_dir):
        os.makedirs(warp_cloth_dir)
    warp_mask_dir = os.path.join(save_dir, 'warp-mask')
    if not os.path.exists(warp_mask_dir):
        os.makedirs(warp_mask_dir)
    warp_refined_dir = os.path.join(save_dir, 'warp-refined')
    if not os.path.exists(warp_refined_dir):
        os.makedirs(warp_refined_dir)
    no_background_c_dir = os.path.join(save_dir, 'no_background_c')
    if not os.path.exists(no_background_c_dir):
        os.makedirs(no_background_c_dir)


    
    for step, inputs in enumerate(test_loader.data_loader):
        iter_start_time = time.time()
        c_names = inputs['c_name']
        im = inputs['image'].cuda()
        im_pose = inputs['pose_image'].cuda()
        im_h = inputs['head'].cuda()
        shape = inputs['shape'].cuda()
        agnostic = inputs['agnostic'].cuda()
        #c = inputs['cloth'].cuda()
        no_background_c = inputs['no_background_cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()
        im_c =  inputs['parse_cloth'].cuda()
        im_g = inputs['grid_image'].cuda()

        c_point_plane = inputs['cloth_points'].cuda()
        p_point_plane = inputs['person_points'].cuda()
        

        grid, theta, warped_cloth, outputs = model(agnostic, no_background_c)
        #warped_cloth = F.grid_sample(c, grid, padding_mode='border')
        warped_mask = F.grid_sample(cm, grid, padding_mode='zeros')
        warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros')
        #compute_c_point_plane = compute_grid_point(p_point_plane, grid)

        c_rendered, m_composite = torch.split(outputs, 3,1)
        c_rendered = F.tanh(c_rendered)
        m_composite = F.sigmoid(m_composite)
        c_result = warped_cloth * m_composite + c_rendered * (1 - m_composite)



        visuals = [ [im_h, shape, im_pose], 
                   [no_background_c, warped_cloth, im_c], 
                   [warped_grid, (warped_cloth+im)*0.5, im],
                   [m_composite, (c_result+im)*0.5, c_result]]


        save_images(warped_cloth, c_names, warp_cloth_dir) 
        save_images(warped_mask*2-1, c_names, warp_mask_dir)
        #print("warp_refined_dir = ", warp_refined_dir)
        #assert os.path.exists(warp_refined_dir)
        save_images(no_background_c, c_names, no_background_c_dir)
        save_images(c_result, c_names, warp_refined_dir) 
            
        if (step+1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f' % (step+1, t), flush=True)

def test_semantic_parsing_refined_gmm(opt, test_loader, model, board):
    model.cuda()
    model.eval()

    base_name = os.path.basename(opt.checkpoint)
    save_dir = os.path.join(opt.result_dir, base_name, opt.datamode)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    warp_cloth_dir = os.path.join(save_dir, 'warp-cloth')
    if not os.path.exists(warp_cloth_dir):
        os.makedirs(warp_cloth_dir)
    warp_mask_dir = os.path.join(save_dir, 'warp-mask')
    if not os.path.exists(warp_mask_dir):
        os.makedirs(warp_mask_dir)
    warp_refined_dir = os.path.join(save_dir, 'warp-refined')
    if not os.path.exists(warp_refined_dir):
        os.makedirs(warp_refined_dir)


    
    for step, inputs in enumerate(test_loader.data_loader):
        iter_start_time = time.time()
        c_names = inputs['c_name']
        im = inputs['image'].cuda()
        im_pose = inputs['pose_image'].cuda()
        im_h = inputs['head'].cuda()
        shape = inputs['shape'].cuda()
        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()
        im_c =  inputs['parse_cloth'].cuda()
        im_g = inputs['grid_image'].cuda()

        fake_label_cloth_mask = inputs['fake_label_cloth_mask'].cuda()

        c_point_plane = inputs['cloth_points'].cuda()
        p_point_plane = inputs['person_points'].cuda()
        

        grid, theta, warped_cloth, outputs = model(fake_label_cloth_mask, c)
        #warped_cloth = F.grid_sample(c, grid, padding_mode='border')
        warped_mask = F.grid_sample(cm, grid, padding_mode='zeros')
        warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros')
        #compute_c_point_plane = compute_grid_point(p_point_plane, grid)

        c_rendered, m_composite = torch.split(outputs, 3,1)
        c_rendered = F.tanh(c_rendered)
        m_composite = F.sigmoid(m_composite)
        c_result = warped_cloth * m_composite + c_rendered * (1 - m_composite)



        visuals = [ [im_h, shape, im_pose], 
                   [c, warped_cloth, im_c], 
                   [warped_grid, (warped_cloth+im)*0.5, im],
                   [m_composite, (c_result+im)*0.5, c_result]]


        save_images(warped_cloth, c_names, warp_cloth_dir) 
        save_images(warped_mask*2-1, c_names, warp_mask_dir)
        #print("warp_refined_dir = ", warp_refined_dir)
        #assert os.path.exists(warp_refined_dir)
        save_images(c_result, c_names, warp_refined_dir) 
            
        if (step+1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f' % (step+1, t), flush=True)

def test_refined_gmm(opt, test_loader, model, board):
    model.cuda()
    model.eval()

    base_name = os.path.basename(opt.checkpoint)
    save_dir = os.path.join(opt.result_dir, base_name, opt.datamode)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    warp_cloth_dir = os.path.join(save_dir, 'warp-cloth')
    if not os.path.exists(warp_cloth_dir):
        os.makedirs(warp_cloth_dir)
    warp_mask_dir = os.path.join(save_dir, 'warp-mask')
    if not os.path.exists(warp_mask_dir):
        os.makedirs(warp_mask_dir)
    warp_refined_dir = os.path.join(save_dir, 'warp-refined')
    if not os.path.exists(warp_refined_dir):
        os.makedirs(warp_refined_dir)
    warp_rendered_dir = os.path.join(save_dir, 'warp-redered')
    if not os.path.exists(warp_rendered_dir):
        os.makedirs(warp_rendered_dir)

    
    for step, inputs in enumerate(test_loader.data_loader):
        iter_start_time = time.time()
        c_names = inputs['c_name']
        im = inputs['image'].cuda()
        im_pose = inputs['pose_image'].cuda()
        im_h = inputs['head'].cuda()
        shape = inputs['shape'].cuda()
        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()
        im_c =  inputs['parse_cloth'].cuda()
        im_g = inputs['grid_image'].cuda()

        c_point_plane = inputs['cloth_points'].cuda()
        p_point_plane = inputs['person_points'].cuda()
        

        grid, theta, warped_cloth, outputs = model(agnostic, c)
        #warped_cloth = F.grid_sample(c, grid, padding_mode='border')
        warped_mask = F.grid_sample(cm, grid, padding_mode='zeros')
        warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros')
        #compute_c_point_plane = compute_grid_point(p_point_plane, grid)

        c_rendered, m_composite = torch.split(outputs, 3,1)
        c_rendered = F.tanh(c_rendered)
        m_composite = F.sigmoid(m_composite)
        c_result = warped_cloth * m_composite + c_rendered * (1 - m_composite)



        visuals = [ [im_h, shape, im_pose], 
                   [c, warped_cloth, im_c], 
                   [warped_grid, (warped_cloth+im)*0.5, im],
                   [m_composite, (c_result+im)*0.5, c_result]]


        save_images(warped_cloth, c_names, warp_cloth_dir) 
        save_images(warped_mask*2-1, c_names, warp_mask_dir)
        #print("warp_refined_dir = ", warp_refined_dir)
        #assert os.path.exists(warp_refined_dir)
        save_images(c_result, c_names, warp_refined_dir)
        save_images(c_rendered, c_names, warp_rendered_dir)
            
        if (step+1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f' % (step+1, t), flush=True)




def test_gmm(opt, test_loader, model, board):
    model.cuda()
    model.eval()

    base_name = os.path.basename(opt.checkpoint)
    save_dir = os.path.join(opt.result_dir, base_name, opt.datamode)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    warp_cloth_dir = os.path.join(save_dir, 'warp-cloth')
    if not os.path.exists(warp_cloth_dir):
        os.makedirs(warp_cloth_dir)
    warp_mask_dir = os.path.join(save_dir, 'warp-mask')
    if not os.path.exists(warp_mask_dir):
        os.makedirs(warp_mask_dir)

    for step, inputs in enumerate(test_loader.data_loader):
        iter_start_time = time.time()
        
        c_names = inputs['c_name']
        im = inputs['image'].cuda()
        im_pose = inputs['pose_image'].cuda()
        im_h = inputs['head'].cuda()
        shape = inputs['shape'].cuda()
        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()
        im_c =  inputs['parse_cloth'].cuda()
        im_g = inputs['grid_image'].cuda()
            
        grid, theta = model(agnostic, c)
        warped_cloth = F.grid_sample(c, grid, padding_mode='border')
        warped_mask = F.grid_sample(cm, grid, padding_mode='zeros')
        warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros')

        visuals = [ [im_h, shape, im_pose], 
                   [c, warped_cloth, im_c], 
                   [warped_grid, (warped_cloth+im)*0.5, im]]
        
        save_images(warped_cloth, c_names, warp_cloth_dir) 
        save_images(warped_mask*2-1, c_names, warp_mask_dir) 

        if (step+1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f' % (step+1, t), flush=True)
        


def test_tom(opt, test_loader, model, board):
    model.cuda()
    model.eval()
    
    base_name = os.path.basename(opt.checkpoint)
    save_dir = os.path.join(opt.result_dir, base_name, opt.datamode)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    try_on_dir = os.path.join(save_dir, 'try-on')
    if not os.path.exists(try_on_dir):
        os.makedirs(try_on_dir)
    print('Dataset size: %05d!' % (len(test_loader.dataset)), flush=True)
    for step, inputs in enumerate(test_loader.data_loader):
        iter_start_time = time.time()
        
        im_names = inputs['im_name']
        im = inputs['image'].cuda()
        im_pose = inputs['pose_image']
        im_h = inputs['head']
        shape = inputs['shape']

        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()
        
        outputs = model(torch.cat([agnostic, c],1))
        p_rendered, m_composite = torch.split(outputs, 3,1)
        p_rendered = F.tanh(p_rendered)
        m_composite = F.sigmoid(m_composite)
        p_tryon = c * m_composite + p_rendered * (1 - m_composite)

        visuals = [ [im_h, shape, im_pose], 
                   [c, 2*cm-1, m_composite], 
                   [p_rendered, p_tryon, im]]
            
        save_images(p_tryon, im_names, try_on_dir) 
        if (step+1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f' % (step+1, t), flush=True)


def main():
    opt = get_opt()
    print(opt)
    print("Start to test stage: %s, named: %s!" % (opt.stage, opt.name))
   
    # create dataset 
    train_dataset = CPDataset(opt)

    # create dataloader
    train_loader = CPDataLoader(opt, train_dataset)

    # visualization
    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)
    board = SummaryWriter(log_dir = os.path.join(opt.tensorboard_dir, opt.name))
   
    # create model & train
    if opt.stage == 'GMM':
        if opt.model == 'RefinedGMM':
            model = RefinedGMM(opt)
        elif opt.model == 'OneRefinedGMM':
            model = OneRefinedGMM(opt)
        else:
            raise TypeError()
        #model = RefinedGMM(opt)
        load_checkpoint(model, opt.checkpoint)
        with torch.no_grad():
            test_refined_gmm(opt, train_loader, model, board)
    # elif opt.stage == 'semanticGMM':
    #     model = RefinedGMM(opt)
    #     load_checkpoint(model, opt.checkpoint)
    #     with torch.no_grad():
    #         test_semantic_parsing_refined_gmm(opt, train_loader, model, board)
    # elif opt.stage == 'VariGMM':
    #     model = VariGMM(opt)
    #     load_checkpoint(model, opt.checkpoint)
    #     with torch.no_grad():
    #         test_refined_gmm(opt, train_loader, model, board)
    # elif opt.stage == 'no_background_GMM':
    #     model = RefinedGMM(opt)
    #     load_checkpoint(model, opt.checkpoint)
    #     with torch.no_grad():
    #         test_no_background_refined_gmm(opt, train_loader, model, board)
    elif opt.stage == 'TOM':
        model = UnetGenerator(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)
        load_checkpoint(model, opt.checkpoint)
        with torch.no_grad():
            test_tom(opt, train_loader, model, board)
    else:
        raise NotImplementedError('Model [%s] is not implemented' % opt.stage)
  
    print('Finished test %s, named: %s!' % (opt.stage, opt.name))

if __name__ == "__main__":
    main()
