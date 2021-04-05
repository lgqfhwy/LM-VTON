import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import os.path as osp
import torchvision.transforms as transforms
import numpy as np
import torch
import json
from PIL import Image
from PIL import ImageDraw
class MpvDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.data_path = opt.dataroot
        self.datamode = opt.phase
        self.data_list = opt.data_list
        self.radius = 5
        self.fine_height = 256
        self.fine_width = 192
        self.warp_path = os.path.join(opt.gmm_path, self.datamode)

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
        
        self.transform = transforms.Compose([  \
            transforms.ToTensor(),   \
            transforms.Normalize((0.5,), (0.5,))])


        ### load precomputed instance-wise encoded features
        if opt.load_features:                              
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' % self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat))

        self.dataset_size = len(self.im_names) 
      
    def __getitem__(self, index):        
        ### input A (label maps)
        im_name = self.im_names[index]
        c_name = self.c_names[index]
        c_name_i = osp.basename(c_name)
        c = Image.open(osp.join(self.warp_path, 'warp-cloth', c_name_i))
        cm = Image.open(osp.join(self.warp_path, 'warp-mask', c_name_i))
        c = self.transform(c)
        cm_array = np.array(cm)
        cm_array = (cm_array >= 128).astype(np.float32)
        cm = torch.from_numpy(cm_array) # [0,1]
        cm.unsqueeze_(0) 

        
        # person image
        im = Image.open(osp.join(self.data_path, "all", im_name))
        im = self.transform(im)

        # load parsing image
        parse_name = im_name.replace('.jpg', '.png')
        im_parse = Image.open(osp.join(self.data_path, 'all_parsing', parse_name))
        parse_array = np.array(im_parse)

        parse_shape = (parse_array > 0).astype(np.float32)
        parse_head = (parse_array == 1).astype(np.float32) + \
                (parse_array == 2).astype(np.float32) + \
                (parse_array == 4).astype(np.float32) + \
                (parse_array == 13).astype(np.float32)
        parse_cloth = (parse_array == 5).astype(np.float32) + \
                (parse_array == 6).astype(np.float32) + \
                (parse_array == 7).astype(np.float32)
        
        # shape downsample
        parse_shape = Image.fromarray((parse_shape*255).astype(np.uint8))
        parse_shape = parse_shape.resize((self.fine_width//16, self.fine_height//16), Image.BILINEAR)
        parse_shape = parse_shape.resize((self.fine_width, self.fine_height), Image.BILINEAR)

        shape = self.transform(parse_shape) # [-1,1]

        phead = torch.from_numpy(parse_head) # [0,1]
        pcm = torch.from_numpy(parse_cloth) # [0,1]


        im_c = im * pcm + (1 - pcm) # [-1,1], fill 1 for other parts

        im_h = im * phead - (1 - phead) # [-1,1], fill 0 for other parts

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
            one_map = self.transform(one_map)
            pose_map[i] = one_map[0]
        
        # just for visualization
        im_pose = self.transform(im_pose)

        agnostic = torch.cat([shape, im_h, pose_map], 0)


        B_tensor = inst_tensor = feat_tensor = 0

        c_name = osp.basename(c_name)
        input_dict = {'agnostic': agnostic, 'cloth': c, 'cloth_mask': cm, 'image': im,
                'feat': feat_tensor, 'path': parse_name, 'im_h':im_h, 'shape':shape, 'im_pose':im_pose}

        return input_dict                         

    def __len__(self):
        return len(self.im_names) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'MpvDataset'