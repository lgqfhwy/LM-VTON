#coding=utf-8
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
from PIL import ImageDraw

import os.path as osp
import numpy as np
import json
import os
import pickle

class CPDataset(data.Dataset):
    """Dataset for CP-VTON.
    """
    def __init__(self, opt):
        super(CPDataset, self).__init__()
        # base setting
        self.opt = opt
        self.root = opt.dataroot
        self.datamode = opt.datamode # train or test or self-defined
        self.stage = opt.stage # GMM or TOM
        #self.data_list = opt.data_list
        self.data_list = "select_person_clothes.txt"
        self.densepose_path = opt.densepose_path
        self.fine_height = opt.fine_height
        self.fine_width = opt.fine_width
        self.radius = opt.radius
        #self.data_path = osp.join(opt.dataroot, opt.datamode)
        self.data_path = opt.dataroot
        self.gmm_path = opt.gmm_path
        self.warp_path = osp.join(self.gmm_path, self.datamode)
        self.transform = transforms.Compose([  \
                transforms.ToTensor(),   \
                transforms.Normalize((0.5,), (0.5,))])
        self.transform_one = transforms.Compose([  \
                transforms.ToTensor(),   \
                transforms.Normalize((0.5, ), (0.5, ))])
        self.is_round = opt.is_round
        
        # load data list
        im_names = []
        c_names = []
        with open(osp.join(opt.dataroot, self.data_list), 'r') as f:
            for line in f.readlines():
                im_name, c_name, mode_i = line.strip().split()
                if mode_i == self.datamode:
                    im_names.append(im_name)
                    c_names.append(c_name)

        self.im_names = im_names
        self.c_names = c_names

        train_cloth_path = "/data/lgq/database/mpv/cloth_points/train_cloth.pkl"
        assert os.path.exists(train_cloth_path)
        person_cloth_path = "/data/lgq/database/mpv/cloth_points/bbox_train_person_cloth.pkl"
        assert os.path.exists(person_cloth_path)
        self.cloth_points = self.load_point(train_cloth_path)
        self.person_points = self.load_point(person_cloth_path)

        self.semantic_path = os.path.join(opt.semantic_path, self.datamode + "_latest", "images")
        assert os.path.exists(self.semantic_path)

        self.transform_parse = transforms.Compose([
            transforms.ToTensor(),
        ])

    def load_point(self, pickle_path):
        point = pickle.load(open(pickle_path, 'rb'))
        return point    

    def name(self):
        return "CPDataset"
    def transform_point(self, pointxy):
        x, y = pointxy
        x = x * 192 / 72
        y = y * 256 / 96
        if self.is_round:
            x = np.round(x)
            y = np.round(y)
        return x, y
    def transform_point_with_bbox(self, pointxy, bbox):
        x, y = pointxy
        x1, y1, x2, y2 = bbox
        width = y2 - y1
        high = x2 - x1
        x = x * width / 72
        y = y * high / 96
        x = x + y1
        y = y + x1
        if self.is_round:
            #print("x = ", x)
            x = np.round(x)
            y = np.round(y)
        return x, y
        
    def valid_point_parse(self, points, parse_mask, radius = 10):
        y, x = points
        y = int(y)
        x = int(x)
        x1 = max(0, x - radius)
        y1 = max(0, y - radius)

        x2 = min(x + radius, self.fine_height)
        y2 = min(y + radius, self.fine_width)
        sum_points = np.sum(parse_mask[x1:x2, y1:y2])
        if (sum_points) > 0:
            return True
        return False



    def __getitem__(self, index):
        c_name = self.c_names[index]
        im_name = self.im_names[index]

        c_name_i = osp.basename(c_name)
        im_name_i = osp.basename(im_name)

        # cloth image & cloth mask
        if self.stage == 'GMM' or self.stage == 'no_background_GMM' or \
            self.stage == "attenGMM" or self.stage == "semanticGMM" or \
            self.stage == 'crossattenGMM' or self.stage == 'crossattenGMMv2' or self.stage == 'local_refined_gmm':
            c = Image.open(osp.join(self.data_path, 'all', c_name))
            #c = Image.open(osp.join("/data1/lgq/graduate/others/graduate_resize.jpg"))
            cm = Image.open(osp.join(self.data_path, 'all', c_name[:-4] + "_mask.jpg"))

        else:
            assert False
            c_name_i = osp.basename(c_name)
            c = Image.open(osp.join(self.warp_path, 'warp-cloth', c_name_i))
            #c = Image.open(osp.join("/data1/lgq/graduate/others/graduate_resize.jpg"))
            cm = Image.open(osp.join(self.warp_path, 'warp-mask', c_name_i))
        
        arm_label_map = Image.open(osp.join(self.semantic_path, im_name_i[:-4] + "_arm_label_map.png"))
        fake_label_cloth_mask = Image.open(osp.join(self.semantic_path, im_name_i[:-4] + "_fake_label_cloth_mask.png"))

        arm_label_map = self.transform_parse(arm_label_map) * 255.0


        fake_label_cloth_mask_array = np.array(fake_label_cloth_mask)
        fake_label_cloth_mask_array = (fake_label_cloth_mask_array >= 128).astype(np.float32)
        fake_label_cloth_mask = torch.from_numpy(fake_label_cloth_mask_array)

        fake_label_cloth_mask.unsqueeze_(0)
        
        c_point = self.cloth_points[index]
        p_point = self.person_points[index]
        
        # assert c_point['name'] == c_name
        # assert p_point['name'] == im_name

     
        c = self.transform(c)  # [-1,1]
        cm_array = np.array(cm)
        cm_array = (cm_array >= 128).astype(np.float32)
        cm = torch.from_numpy(cm_array) # [0,1]
        cm.unsqueeze_(0)

        no_background_c = c * cm


        # person image 
        im = Image.open(osp.join(self.data_path, 'all', im_name))
        im = self.transform(im) # [-1,1]

        # load parsing image
        parse_name = im_name.replace('.jpg', '.png')
        im_parse = Image.open(osp.join(self.data_path, 'all_parsing', parse_name))
        parse_array = np.array(im_parse)

        parse_name_i = im_name_i.replace(".jpg", ".png")
        densepose_shape = Image.open(osp.join(self.densepose_path, parse_name_i))
        densepose_shape_array = np.array(densepose_shape)
        #print("densepose_array_shape = ", im_name_i, "  shape = ", densepose_shape_array.shape)
        # if densepose_shape_array.shape[0] == 3:
        #     print("im_name_i = ", im_name_i)
        densepose_shape_tensor = self.transform_one(densepose_shape_array)

        target_parse = self.transform(parse_array)
        #print("parse_array shape = ", parse_array.shape)
        parse_shape = (parse_array > 0).astype(np.float32)
        target_shape = parse_shape
        parse_head = (parse_array == 1).astype(np.float32) + \
                (parse_array == 2).astype(np.float32) + \
                (parse_array == 4).astype(np.float32) + \
                (parse_array == 13).astype(np.float32)
        parse_cloth = (parse_array == 5).astype(np.float32) + \
                (parse_array == 6).astype(np.float32) + \
                (parse_array == 7).astype(np.float32)
        
        valid_c_point = np.zeros((40, 2)).astype(np.float32)
        valid_p_point = np.zeros((40, 2)).astype(np.float32)
        for i in range(len(c_point['points'])):
            trans_c_points = self.transform_point(c_point['points'][i])
            trans_p_points = self.transform_point_with_bbox(p_point['points'][i], p_point['bbox'])
            if self.valid_point_parse(trans_p_points, parse_cloth):
                valid_c_point[i] = trans_c_points
                valid_p_point[i] = trans_p_points
        c_point_plane = valid_c_point
        p_point_plane = valid_p_point

        # len_c = len(valid_c_point)
        # if len_c > 40:
        #     len_c = 40
        # c_point_plane = np.zeros((self.fine_height, self.fine_width, 40)).astype(np.float32)
        # p_point_plane = np.zeros((self.fine_height, self.fine_width, 40)).astype(np.float32)
        #print("before c shape = ", c_point_plane.shape)
        # for i in range(len_c):
        #     c_point_plane[valid_c_point[i][0], valid_c_point[i][1], i] = 1
        #     p_point_plane[valid_p_point[i][0], valid_p_point[i][1], i] = 1
        #print("hello c point shape = ", c_point_plane.shape)
        # c_point_plane = self.transform(c_point_plane)
        # p_point_plane = self.transform(p_point_plane)
        #print("c_point_plane size = ", c_point_plane.size())
        
       
        # shape downsample
        parse_shape = Image.fromarray((parse_shape*255).astype(np.uint8))
        parse_shape = parse_shape.resize((self.fine_width//16, self.fine_height//16), Image.BILINEAR)
        parse_shape = parse_shape.resize((self.fine_width, self.fine_height), Image.BILINEAR)
        #print("parse_shape shape = ", parse_shape.size)
        shape = self.transform_one(parse_shape) # [-1,1]
        #print("shape size = ", shape.size())
        phead = torch.from_numpy(parse_head) # [0,1]
        pcm = torch.from_numpy(parse_cloth) # [0,1]

        parse_cloth_mask = pcm
        

        # upper cloth
        im_c = im * pcm + (1 - pcm) # [-1,1], fill 1 for other parts
        im_h = im * phead - (1 - phead) # [-1,1], fill 0 for other parts

        no_background_im_c = im * pcm # fill 0 for other parts.

        parse_cloth_mask.unsqueeze_(0)

        # load pose points
        pose_name = im_name.replace('.jpg', '_keypoints.json')
        with open(osp.join(self.data_path, 'all_person_clothes_keypoints', pose_name), 'r') as f:
            pose_label = json.load(f)
            pose_data = pose_label['people'][0]['pose_keypoints']
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1,3))

        point_num = pose_data.shape[0]
        pose_map = torch.zeros(point_num, self.fine_height, self.fine_width)
        r = self.radius
        im_pose = Image.new('L', (self.fine_width, self.fine_height))
        pose_draw = ImageDraw.Draw(im_pose)
        for i in range(point_num):
            one_map = Image.new('L', (self.fine_width, self.fine_height))
            draw = ImageDraw.Draw(one_map)
            pointx = pose_data[i,0]
            pointy = pose_data[i,1]
            if pointx > 1 and pointy > 1:
                draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
                pose_draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
            one_map = self.transform_one(one_map)
            pose_map[i] = one_map[0]

        # just for visualization
        im_pose = self.transform_one(im_pose)
        
        # cloth-agnostic representation
        #agnostic = torch.cat([shape, im_h, pose_map], 0) 
        agnostic = torch.cat([densepose_shape_tensor, im_h, pose_map], 0) 

        #print("ahnostic size = ", agnostic.size())

        #if self.stage == 'GMM' or self.stage == 'no_background_GMM':
        im_g = Image.open('grid.png')
        im_g = self.transform(im_g)
        # else:
        #     im_g = ''

        c_name = osp.basename(c_name).replace('.jpg', '.png')
        im_name = osp.basename(im_name)
        result = {
            'c_name':   c_name,     # for visualization
            'im_name':  im_name,    # for visualization or ground truth
            'cloth':    c,          # for input
            'cloth_mask':     cm,   # for input
            'image':    im,         # for visualization
            'agnostic': agnostic,   # for input
            'parse_cloth':  im_c,    # for ground truth
            'shape': shape,         # for visualization
            'densepose_shape' : densepose_shape_tensor,
            'head': im_h,           # for visualization
            'pose_image': im_pose,  # for visualization
            'grid_image': im_g,     # for visualization
            'parse_cloth_mask': parse_cloth_mask,
            'cloth_points': c_point_plane,
            'person_points': p_point_plane,
            'target_parse': target_parse,
            'target_shape': target_shape,
            'no_background_cloth': no_background_c,
            'no_background_parse_cloth': no_background_im_c,
            'arm_label_map': arm_label_map,
            'fake_label_cloth_mask': fake_label_cloth_mask,
            }

        return result

    def __len__(self):
        return len(self.im_names)

class CPDataLoader(object):
    def __init__(self, opt, dataset):
        super(CPDataLoader, self).__init__()

        if opt.shuffle :
            train_sampler = torch.utils.data.sampler.RandomSampler(dataset)
        else:
            train_sampler = None

        self.data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
                num_workers=opt.workers, pin_memory=True, sampler=train_sampler)
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()
       
    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch


if __name__ == "__main__":
    print("Check the dataset for geometric matching module!")
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default = "data")
    parser.add_argument("--datamode", default = "train")
    parser.add_argument("--stage", default = "GMM")
    parser.add_argument("--data_list", default = "train_pairs.txt")
    parser.add_argument("--fine_width", type=int, default = 192)
    parser.add_argument("--fine_height", type=int, default = 256)
    parser.add_argument("--radius", type=int, default = 3)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    parser.add_argument('-j', '--workers', type=int, default=1)
    
    opt = parser.parse_args()
    dataset = CPDataset(opt)
    data_loader = CPDataLoader(opt, dataset)

    print('Size of the dataset: %05d, dataloader: %04d' \
            % (len(dataset), len(data_loader.data_loader)))
    first_item = dataset.__getitem__(0)
    first_batch = data_loader.next_batch()

    from IPython import embed; embed()

