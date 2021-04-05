#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import os
import time
from mpv_cloth_points_dataset import CPDataset, CPDataLoader
from networks import GMM, UnetGenerator, VGGLoss, load_checkpoint, save_checkpoint
from multi_networks import RefinedGMM, OneRefinedGMM, GramLoss
from tensorboardX import SummaryWriter
from visualization import board_add_image, board_add_images

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default = "GMM")
    parser.add_argument("--gpu_ids", default = "")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    parser.add_argument('--is_round', type = bool, default = True)
    parser.add_argument('--loss_weight', type = float, default = 0.01)
    parser.add_argument("--dataroot", default = "data")
    parser.add_argument("--datamode", default = "train")
    parser.add_argument("--stage", default = "GMM")
    parser.add_argument("--data_list", default = "train_pairs.txt")
    parser.add_argument("--densepose_path", default="/data/lgq/database/mpv/densepose_mpv_shape")
    parser.add_argument("--fine_width", type=int, default = 192)
    parser.add_argument("--fine_height", type=int, default = 256)
    parser.add_argument("--radius", type=int, default = 5)
    parser.add_argument("--grid_size", type=int, default = 5)
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint infos')
    parser.add_argument('--checkpoint', type=str, default='', help='model checkpoint for initialization')
    parser.add_argument("--display_count", type=int, default = 20)
    parser.add_argument("--save_count", type=int, default = 100000)
    parser.add_argument("--keep_step", type=int, default = 100000)
    parser.add_argument("--decay_step", type=int, default = 100000)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
    parser.add_argument("--gmm_path", type = str, default = "../result")
    parser.add_argument("--encoder_name", type = str, default="resnet34")
    parser.add_argument("--gmm_feature_A_input_nc", type = int, default=22)
    parser.add_argument("--add_point_loss", action='store_true')
    parser.add_argument("--add_render_loss", action = 'store_true')
    parser.add_argument("--add_origin_cloth", action = 'store_true')
    parser.add_argument("--add_gram_loss", action = 'store_true')
    parser.add_argument("--add_vgg_loss", action = 'store_true')
    parser.add_argument("--add_warped_mask_loss", action= 'store_true')
    parser.add_argument("--model", type = str, default = 'RefinedGMM')
    parser.add_argument("--add_mask_constrain", action = 'store_true')
    parser.add_argument("--mask_constrain_weight", type = float, default = 1)
    parser.add_argument("--semantic_path", type = str, default = "/data/lgq/new_graduate/pix2pix_results/images_semantic_pix2pixhd_mpv/semantic_pix2pixhd_mpv")
    opt = parser.parse_args()
    return opt

def compute_grid_point(p_point_plane, grid, num = 40):
    batch_n, h, w, c = grid.size()
    assert h == 256
    assert w == 192

    c_point_plane = torch.zeros([batch_n, num, 2], dtype = torch.float).cuda()
    for n in range(batch_n):
        for i in range(num):
            if (torch.sum(p_point_plane[n][i]) == 0):
                continue
            y, x = p_point_plane[n][i]
            y = y.int().item()
            x = x.int().item()

            #print("y = ", y, " x = ", x)
            ix = grid[n][x][y][0]
            iy = grid[n][x][y][1]
            c_ix = ((ix + 1) / 2) * (w - 1)
            c_iy = ((iy + 1) / 2) * (h - 1)
            c_point_plane[n][i][0] = c_ix
            c_point_plane[n][i][1] = c_iy
    return c_point_plane



def train_no_background_refined_gmm(opt, train_loader, model, board):
    model.cuda()
    model.train()

    loss_weight = opt.loss_weight
    # if loss_weight > 0.01:
    #     print("Error")
    #     assert False

    # criterion
    warped_criterionL1 = nn.L1Loss()
    result_criterionL1 = nn.L1Loss()
    point_criterionL1 = nn.L1Loss()
    criterionMask = nn.L1Loss()
    criterionVGG = VGGLoss()
    
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda step: 1.0 -
            max(0, step - opt.keep_step) / float(opt.decay_step + 1))
    
    for step in range(opt.keep_step + opt.decay_step):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()
            
        im = inputs['image'].cuda()
        im_pose = inputs['pose_image'].cuda()
        im_h = inputs['head'].cuda()
        shape = inputs['shape'].cuda()
        agnostic = inputs['agnostic'].cuda()
        no_background_c = inputs['no_background_cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()
        no_background_im_c =  inputs['no_background_parse_cloth'].cuda()
        im_g = inputs['grid_image'].cuda()

        c_point_plane = inputs['cloth_points'].cuda()
        p_point_plane = inputs['person_points'].cuda()
        

        grid, theta, warped_cloth, outputs = model(agnostic, no_background_c)
        #warped_cloth = F.grid_sample(c, grid, padding_mode='border')
        warped_mask = F.grid_sample(cm, grid, padding_mode='zeros')
        warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros')
        compute_c_point_plane = compute_grid_point(p_point_plane, grid)

        c_rendered, m_composite = torch.split(outputs, 3,1)
        c_rendered = F.tanh(c_rendered)
        m_composite = F.sigmoid(m_composite)
        c_result = warped_cloth * m_composite + c_rendered * (1 - m_composite)



        visuals = [ [im_h, shape, im_pose], 
                   [no_background_c, warped_cloth, no_background_im_c], 
                   [warped_grid, (warped_cloth+im)*0.5, im],
                   [m_composite, (c_result+im)*0.5, c_result]]
        
        loss_warped_cloth = warped_criterionL1(warped_cloth, no_background_im_c)
        loss_point = point_criterionL1(compute_c_point_plane, c_point_plane)
        loss_c_result = result_criterionL1(c_result, no_background_im_c)
        loss_mask = criterionMask(m_composite, warped_mask)
        loss_vgg = criterionVGG(c_result, no_background_im_c)

        #print("loss cloth = ", loss_cloth)
        #print("loss point = ", loss_point)
        loss = loss_warped_cloth + loss_weight * loss_point + loss_c_result + loss_mask + loss_vgg 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        if (step+1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step+1)
            board.add_scalar('metric', loss.item(), step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %4f' % (step+1, t, loss.item()), flush=True)

        if (step+1) % opt.save_count == 0:
            save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step+1)))


def train_semantic_parsing_refined_gmm(opt, train_loader, model, board):
    model.cuda()
    model.train()

    loss_weight = opt.loss_weight
    # if loss_weight > 0.01:
    #     print("Error")
    #     assert False

    # criterion
    warped_criterionL1 = nn.L1Loss()
    result_criterionL1 = nn.L1Loss()
    point_criterionL1 = nn.L1Loss()
    criterionMask = nn.L1Loss()
    criterionVGG = VGGLoss()
    
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda step: 1.0 -
            max(0, step - opt.keep_step) / float(opt.decay_step + 1))
    
    for step in range(opt.keep_step + opt.decay_step):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()
            
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
        compute_c_point_plane = compute_grid_point(p_point_plane, grid)

        c_rendered, m_composite = torch.split(outputs, 3,1)
        c_rendered = F.tanh(c_rendered)
        m_composite = F.sigmoid(m_composite)
        c_result = warped_cloth * m_composite + c_rendered * (1 - m_composite)



        visuals = [ [im_h, shape, im_pose], 
                   [c, warped_cloth, im_c], 
                   [warped_grid, (warped_cloth+im)*0.5, im],
                   [m_composite, (c_result+im)*0.5, c_result]]
        
        loss_warped_cloth = warped_criterionL1(warped_cloth, im_c)
        loss_point = point_criterionL1(compute_c_point_plane, c_point_plane)
        loss_c_result = result_criterionL1(c_result, im_c)
        loss_mask = criterionMask(m_composite, warped_mask)
        loss_vgg = criterionVGG(c_result, im_c)

        #print("loss cloth = ", loss_cloth)
        #print("loss point = ", loss_point)
        loss = loss_warped_cloth + loss_weight * loss_point + loss_c_result + loss_mask + loss_vgg 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        if (step+1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step+1)
            board.add_scalar('metric', loss.item(), step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %4f' % (step+1, t, loss.item()), flush=True)

        if (step+1) % opt.save_count == 0:
            save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step+1)))






# def train_refined_gmm(opt, train_loader, model, board):
#     model.cuda()
#     model.train()

#     loss_weight = opt.loss_weight
#     # if loss_weight > 0.01:
#     #     print("Error")
#     #     assert False

#     # criterion
#     warped_criterionL1 = nn.L1Loss()
#     result_criterionL1 = nn.L1Loss()
#     point_criterionL1 = nn.L1Loss()
#     criterionMask = nn.L1Loss()
#     criterionVGG = VGGLoss()
#     rendered_criterionL1 = nn.L1Loss()
    
#     # optimizer
#     optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
#     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda step: 1.0 -
#             max(0, step - opt.keep_step) / float(opt.decay_step + 1))
    
#     for step in range(opt.keep_step + opt.decay_step):
#         iter_start_time = time.time()
#         inputs = train_loader.next_batch()
            
#         im = inputs['image'].cuda()
#         im_pose = inputs['pose_image'].cuda()
#         im_h = inputs['head'].cuda()
#         shape = inputs['shape'].cuda()
#         agnostic = inputs['agnostic'].cuda()
#         c = inputs['cloth'].cuda()
#         cm = inputs['cloth_mask'].cuda()
#         im_c =  inputs['parse_cloth'].cuda()
#         im_g = inputs['grid_image'].cuda()

#         c_point_plane = inputs['cloth_points'].cuda()
#         p_point_plane = inputs['person_points'].cuda()
        

#         grid, theta, warped_cloth, outputs = model(agnostic, c)
#         #warped_cloth = F.grid_sample(c, grid, padding_mode='border')
#         warped_mask = F.grid_sample(cm, grid, padding_mode='zeros')
#         warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros')
#         compute_c_point_plane = compute_grid_point(p_point_plane, grid)

#         c_rendered, m_composite = torch.split(outputs, 3,1)
#         c_rendered = F.tanh(c_rendered)
#         m_composite = F.sigmoid(m_composite)
#         c_result = warped_cloth * m_composite + c_rendered * (1 - m_composite)



#         visuals = [ [im_h, shape, im_pose], 
#                    [c, warped_cloth, im_c], 
#                    [warped_grid, (warped_cloth+im)*0.5, im],
#                    [m_composite, (c_result+im)*0.5, c_result]]
        
#         loss_warped_cloth = warped_criterionL1(warped_cloth, im_c)
#         loss_point = point_criterionL1(compute_c_point_plane, c_point_plane)
#         loss_c_result = result_criterionL1(c_result, im_c)
#         loss_mask = criterionMask(m_composite, warped_mask)
#         loss_vgg = criterionVGG(c_result, im_c)
#         if opt.add_render_loss:
#             loss_render = rendered_criterionL1(c_rendered, im_c)

#         # print("loss cloth = ", loss_warped_cloth)
#         # print("loss point = ", loss_point)
#         # print("loss render = ", loss_render)
#         # print("loss_c_result = ", loss_c_result)
#         if opt.add_render_loss:
#             loss = loss_warped_cloth + loss_weight * loss_point + loss_c_result + loss_mask + loss_vgg + loss_render
#         else:
#             loss = loss_warped_cloth + loss_weight * loss_point + loss_c_result + loss_mask + loss_vgg 
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
            
#         if (step+1) % opt.display_count == 0:
#             board_add_images(board, 'combine', visuals, step+1)
#             board.add_scalar('metric', loss.item(), step+1)
#             t = time.time() - iter_start_time
#             print('step: %8d, time: %.3f, loss: %4f' % (step+1, t, loss.item()), flush=True)

#         if (step+1) % opt.save_count == 0:
#             save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step+1)))



def train_refined_gmm(opt, train_loader, model, board):
    model.cuda()
    model.train()

    loss_weight = opt.loss_weight
    # if loss_weight > 0.01:
    #     print("Error")
    #     assert False

    # criterion
    warped_criterionL1 = nn.L1Loss()
    result_criterionL1 = nn.L1Loss()
    point_criterionL1 = nn.L1Loss()
    criterionMask = nn.L1Loss()
    criterionVGG = VGGLoss()
    criterionGram = GramLoss()
    rendered_criterionL1 = nn.L1Loss()

    center_mask_critetionL1 = nn.L1Loss()

    warped_mask_criterionL1 = nn.L1Loss()
    
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda step: 1.0 -
            max(0, step - opt.keep_step) / float(opt.decay_step + 1))
    
    for step in range(opt.keep_step + opt.decay_step):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()
            
        im = inputs['image'].cuda()
        im_pose = inputs['pose_image'].cuda()
        im_h = inputs['head'].cuda()
        shape = inputs['shape'].cuda()
        densepose_shape = inputs['densepose_shape'].cuda()
        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()
        im_c =  inputs['parse_cloth'].cuda()
        im_g = inputs['grid_image'].cuda()
        parse_cloth_mask = inputs['parse_cloth_mask'].cuda()
        target_shape = inputs['target_shape']

        c_point_plane = inputs['cloth_points'].cuda()
        p_point_plane = inputs['person_points'].cuda()
        

        grid, theta, warped_cloth, outputs = model(agnostic, c)
        #warped_cloth = F.grid_sample(c, grid, padding_mode='border')
        warped_mask = F.grid_sample(cm, grid, padding_mode='zeros')
        warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros')
        compute_c_point_plane = compute_grid_point(p_point_plane, grid)

        warped_mask_loss = 0
        if opt.add_warped_mask_loss:
            warped_mask_loss += warped_mask_criterionL1(warped_mask, target_shape)

        c_rendered, m_composite = torch.split(outputs, 3,1)
        c_rendered = F.tanh(c_rendered)
        m_composite = F.sigmoid(m_composite)
        c_result = warped_cloth * m_composite + c_rendered * (1 - m_composite)



        visuals = [ [im_h, shape, im_pose], 
                   [c, warped_cloth, im_c], 
                   [warped_grid, (warped_cloth+im)*0.5, im],
                   [m_composite, (c_result+im)*0.5, c_result]]
        
        loss_warped_cloth = warped_criterionL1(warped_cloth, im_c)
        loss_point = 0
        if opt.add_point_loss:
            loss_point = point_criterionL1(compute_c_point_plane, c_point_plane)
        loss_c_result = result_criterionL1(c_result, im_c)
        loss_mask = criterionMask(m_composite, warped_mask)
        loss_vgg = 0
        if opt.add_vgg_loss:
            loss_vgg = criterionVGG(c_result, im_c)
        loss_gram = 0
        if opt.add_gram_loss:
            loss_gram += criterionGram(c_result, im_c)

        loss_render = 0
        if opt.add_render_loss:
            loss_render += rendered_criterionL1(c_rendered, im_c)
        
        loss_mask_constrain = 0
        if opt.add_mask_constrain:
            center_mask = m_composite * parse_cloth_mask
            ground_mask = torch.ones_like(parse_cloth_mask, dtype = torch.float)
            ground_mask = ground_mask * warped_mask * parse_cloth_mask
            loss_mask_constrain = center_mask_critetionL1(center_mask, ground_mask)
            #print("long_mask_constrain = ", loss_mask_constrain)
            loss_mask_constrain = loss_mask_constrain * opt.mask_constrain_weight
            #print("long_mask_constrain = ", loss_mask_constrain)
        # print("loss cloth = ", loss_warped_cloth)
        # print("loss point = ", loss_point)
        # print("loss render = ", loss_render)
        # print("loss_c_result = ", loss_c_result)

        loss = loss_warped_cloth + loss_weight * loss_point + loss_c_result + loss_mask + loss_vgg + loss_render + loss_mask_constrain + warped_mask_loss + loss_gram


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        if (step+1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step+1)
            board.add_scalar('metric', loss.item(), step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %4f' % (step+1, t, loss.item()), flush=True)

        if (step+1) % opt.save_count == 0:
            save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step+1)))




def train_tom(opt, train_loader, model, board):
    model.cuda()
    model.train()
    
    # criterion
    criterionL1 = nn.L1Loss()
    criterionVGG = VGGLoss()
    criterionMask = nn.L1Loss()
    
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda step: 1.0 -
            max(0, step - opt.keep_step) / float(opt.decay_step + 1))
    
    for step in range(opt.keep_step + opt.decay_step):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()
            
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
        p_tryon = c * m_composite+ p_rendered * (1 - m_composite)

        visuals = [ [im_h, shape, im_pose], 
                   [c, cm*2-1, m_composite*2-1], 
                   [p_rendered, p_tryon, im]]
            
        loss_l1 = criterionL1(p_tryon, im)
        loss_vgg = criterionVGG(p_tryon, im)
        loss_mask = criterionMask(m_composite, cm)
        loss = loss_l1 + loss_vgg + loss_mask
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        if (step+1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step+1)
            board.add_scalar('metric', loss.item(), step+1)
            board.add_scalar('L1', loss_l1.item(), step+1)
            board.add_scalar('VGG', loss_vgg.item(), step+1)
            board.add_scalar('MaskL1', loss_mask.item(), step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %.4f, l1: %.4f, vgg: %.4f, mask: %.4f' 
                    % (step+1, t, loss.item(), loss_l1.item(), 
                    loss_vgg.item(), loss_mask.item()), flush=True)

        if (step+1) % opt.save_count == 0:
            save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step+1)))
def train_deep_tom(opt, train_loader, model, board):
    print("here")
    model.cuda()
    model.train()
    
    # criterion
    criterionL1 = nn.L1Loss()
    criterionVGG = VGGLoss()
    criterionMask = nn.L1Loss()
    
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda step: 1.0 -
            max(0, step - opt.keep_step) / float(opt.decay_step + 1))
    
    for step in range(opt.keep_step + opt.decay_step):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()
            
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
        p_tryon = c * m_composite+ p_rendered * (1 - m_composite)

        visuals = [ [im_h, shape, im_pose], 
                   [c, cm*2-1, m_composite*2-1], 
                   [p_rendered, p_tryon, im]]
            
        loss_l1 = criterionL1(p_tryon, im)
        loss_vgg = criterionVGG(p_tryon, im)
        loss_mask = criterionMask(m_composite, cm)
        loss = loss_l1 + loss_vgg + loss_mask
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        if (step+1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step+1)
            board.add_scalar('metric', loss.item(), step+1)
            board.add_scalar('L1', loss_l1.item(), step+1)
            board.add_scalar('VGG', loss_vgg.item(), step+1)
            board.add_scalar('MaskL1', loss_mask.item(), step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %.4f, l1: %.4f, vgg: %.4f, mask: %.4f' 
                    % (step+1, t, loss.item(), loss_l1.item(), 
                    loss_vgg.item(), loss_mask.item()), flush=True)

        if (step+1) % opt.save_count == 0:
            save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step+1)))


def main():
    opt = get_opt()
    print(opt)
    print("Start to train stage: %s, named: %s!" % (opt.stage, opt.name))
   
    # create dataset 
    train_dataset = CPDataset(opt)

    # create dataloader
    train_loader = CPDataLoader(opt, train_dataset)

    # visualization
    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)
    board = SummaryWriter(log_dir = os.path.join(opt.tensorboard_dir, opt.name))
   
    # create model & train & save the final checkpoint
    if opt.stage == 'GMM':
        if opt.model == 'RefinedGMM':
            model = RefinedGMM(opt)
        elif opt.model == 'OneRefinedGMM':
            model = OneRefinedGMM(opt)
        else:
            raise TypeError()
        if not opt.checkpoint =='' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
        train_refined_gmm(opt, train_loader, model, board)
        save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'refined_gmm_final.pth'))
    elif opt.stage == 'VariGMM':
        model = VariGMM(opt)
        if not opt.checkpoint =='' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
        train_refined_gmm(opt, train_loader, model, board)
        save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'refined_gmm_final.pth'))
    elif opt.stage == 'semanticGMM':
        model = RefinedGMM(opt)
        if not opt.checkpoint =='' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
        train_semantic_parsing_refined_gmm(opt, train_loader, model, board)
        save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'refined_gmm_final.pth'))
    elif opt.stage == 'no_background_GMM':
        model = RefinedGMM(opt)
        if not opt.checkpoint =='' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
        train_no_background_refined_gmm(opt, train_loader, model, board)
        save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'no_background_refined_gmm_final.pth'))
    elif opt.stage == 'TOM':
        model = UnetGenerator(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)
        if not opt.checkpoint =='' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
        train_tom(opt, train_loader, model, board)
        save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'tom_final.pth'))
    elif opt.stage == 'DeepTom':
        norm_layer = 'instance'
        use_dropout = True
        with_tanh = False
        model = Define_G(25, 4, 64, 'treeresnet', 'instance', 
                                True, 'normal', 0.02, opt.gpu_ids, with_tanh=False)
        train_deep_tom(opt, train_loader, model, board)
        save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'tom_final.pth'))
    else:
        raise NotImplementedError('Model [%s] is not implemented' % opt.stage)
        
  
    print('Finished training %s, nameed: %s!' % (opt.stage, opt.name))

if __name__ == "__main__":
    main()
