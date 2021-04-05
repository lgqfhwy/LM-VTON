import numpy as np
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import torch.nn as nn
import torch.nn.functional as F
import cv2

def encoder(label_map, size):
    label_nc = 20
    #print("unique = ", torch.unique(label_map))
    oneHot_size = (size[0], label_nc, size[2], size[3])
    input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
    input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)
    return input_label

def morpho(mask,iter,bigger=True):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    new=[]
    for i in range(len(mask)):
        tem=mask[i].cpu().detach().numpy().squeeze().reshape(256,192,1)*255
        tem=tem.astype(np.uint8)
        if bigger:
            tem=cv2.dilate(tem,kernel,iterations=iter)
        else:
            tem=cv2.erode(tem,kernel,iterations=iter)
        tem=tem.astype(np.float64)
        tem=tem.reshape(1,256,192)
        new.append(tem.astype(np.float64)/255.0)
    new=np.stack(new)
    new=torch.FloatTensor(new).cuda()
    return new



def generate_discrete_label(inputs, label_nc, onehot=True, encode=True):
    pred_batch = []
    size = inputs.size()
    #print("size = ", size)
    for input in inputs:
        input = input.view(1, label_nc, size[2], size[3])
        pred = np.squeeze(input.data.max(1)[1].cpu().numpy(), axis=0)
        pred_batch.append(pred)

    pred_batch = np.array(pred_batch)
    pred_batch = torch.from_numpy(pred_batch)
    label_map = []
    for p in pred_batch:
        p = p.view(1, 256, 192)
        label_map.append(p)
    label_map = torch.stack(label_map, 0)
    if not onehot:
        return label_map.float().cuda()
    size = label_map.size()
    oneHot_size = (size[0], label_nc, size[2], size[3])
    input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
    input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)

    return input_label
class Pix2PixHDModel(BaseModel):
    def name(self):
        return 'Pix2PixHDModel'
    
    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss):
        flags = (True, True, True, True)
        def loss_filter(g_gan, g_gan_feat, d_real, d_fake):
            return [l for (l,f) in zip((g_gan,g_gan_feat,d_real,d_fake),flags) if f]
        return loss_filter
    
    def initialize(self, opt):
        self.no_background = opt.no_background
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none' or not opt.isTrain: # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        self.use_features = opt.instance_feat or opt.label_feat
        self.gen_features = self.use_features and not self.opt.load_features
        input_nc = opt.label_nc if opt.label_nc != 0 else opt.input_nc

        ##### define networks        
        # Generator network
        netG_input_nc = input_nc

        self.vgg_weight = opt.vgg_weight
        self.point_loss_weight = 0.01

        self.content_input = 29
        self.add_noise = opt.add_noise

        if self.add_noise:
            self.content_input += 1        
        print("input = ", self.content_input)
        self.gmm_input = opt.gmm_input
        self.netGMM = networks.define_TPS(self.gmm_input, 3, gpu_ids=self.gpu_ids)
        self.netG = networks.define_Refine(self.content_input, 3, gpu_ids=self.gpu_ids)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.warped_criterionL1 = nn.L1Loss()
        self.point_criterionL1 = nn.L1Loss()
        self.BCE = torch.nn.BCEWithLogitsLoss()
        self.criterionL1 = nn.L1Loss()
        self.criterionVGG = networks.VGGLoss(self.gpu_ids)
        # Discriminator network
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = input_nc + opt.output_nc
            if not opt.no_instance:
                netD_input_nc += 1
            self.netD = networks.define_D(self.content_input + 3, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, 
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)


        ### Encoder network
        if self.gen_features:          
            self.netE = networks.define_G(opt.output_nc, opt.feat_num, opt.nef, 'encoder', 
                                          opt.n_downsample_E, norm=opt.norm, gpu_ids=self.gpu_ids)  
        if self.opt.verbose:
                print('---------- Networks initialized -------------')

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)
            

        # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            # define loss functions
            self.loss_filter = self.init_loss_filter(not opt.no_ganFeat_loss, not opt.no_vgg_loss)
            
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)   
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:             
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)
                
        
            # Names so we can breakout loss
            self.loss_names = self.loss_filter('G_GAN','G_GAN_CE','D_real', 'D_fake')

            # initialize optimizers
            # optimizer G
            if opt.niter_fix_global > 0:                
                import sys
                if sys.version_info >= (3,0):
                    finetune_list = set()
                else:
                    from sets import Set
                    finetune_list = Set()

                params_dict = dict(self.netG.named_parameters())
                params = []
                for key, value in params_dict.items():       
                    if key.startswith('model' + str(opt.n_local_enhancers)):                    
                        params += [value]
                        finetune_list.add(key.split('.')[0])  
                print('------------- Only training the local enhancer network (for %d epochs) ------------' % opt.niter_fix_global)
                print('The layers that are finetuned are ', sorted(finetune_list))                         
            else:
                params = list(self.netGMM.parameters())
                params += list(self.netG.parameters())
            # if self.gen_features:              
            #     params += list(self.netE.parameters())         
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))                            

            # optimizer D                        
            params = list(self.netD.parameters())
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

    def ger_average_color(self, mask, arms):
        color = torch.zeros(arms.size()).cuda()
        for i in range(arms.size()[0]):
            count = len(torch.nonzero(mask[i, :, :, :]))
            if count < 10:
                color[i, 0, :, :] = 0
                color[i, 1, :, :] = 0
                color[i, 2, :, :] = 0

            else:
                color[i, 0, :, :] = arms[i, 0, :, :].sum() / count
                color[i, 1, :, :] = arms[i, 1, :, :].sum() / count
                color[i, 2, :, :] = arms[i, 2, :, :].sum() / count
        return color
    def encode_input(self, label_map, clothes_mask, all_clothes_label):

        # print("label map size = ", label_map.size())
        # print("clothes_mask size = ", clothes_mask.size())
        # print("all_clothes_label size = ", all_clothes_label.size())
        # stop

        size = label_map.size()
        oneHot_size = (size[0], 20, size[2], size[3])
        input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
        input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)

        masked_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
        masked_label = masked_label.scatter_(1, (label_map * (1 - clothes_mask)).data.long().cuda(), 1.0)

        c_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
        c_label = c_label.scatter_(1, all_clothes_label.data.long().cuda(), 1.0)

        input_label = Variable(input_label)

        return input_label, masked_label, c_label


    def discriminate_D(self, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:            
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)
    def cross_entropy2d(self, input, target, weight=None, size_average=True):
        n, c, h, w = input.size()
        nt, ht, wt = target.size()

        # Handle inconsistent size between input and target
        if h != ht or w != wt:
            input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

        input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        target = target.view(-1)
        loss = F.cross_entropy(
            input, target, weight=weight, size_average=size_average, ignore_index=250
        )

        return loss
    def gen_noise(self, shape):
        noise = np.zeros(shape, dtype=np.uint8)
        ### noise
        noise = cv2.randn(noise, 0, 255)
        noise = np.asarray(noise / 255, dtype=np.uint8)
        noise = torch.tensor(noise, dtype=torch.float32)
        return noise.cuda()
    def compute_grid_point(self, p_point_plane, grid, num = 40):
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

    def forward(self, label_parse, cloth, cloth_mask, agnostic, im_c, c_point_plane, p_point_plane, arm_label_map, fake_label_cloth_mask, 
            remove_arm_parse, label_cloth_mask, real_image, pose, all_label_person_shape, infer=False):


        label_parse = label_parse.cuda()
        cloth = cloth.cuda()
        cloth_mask = cloth_mask.cuda()
        agnostic = agnostic.cuda()
        im_c = im_c.cuda()
        arm_label_map = arm_label_map.cuda()
        # print("before unique = ", torch.unique(arm_label_map))
        fake_label_cloth_mask = fake_label_cloth_mask.cuda()
        remove_arm_parse = remove_arm_parse.cuda()
        label_cloth_mask = label_cloth_mask.cuda()
        real_image = real_image.cuda()
        pose = pose.cuda()
        all_label_person_shape = all_label_person_shape.cuda()
        c_point_plane = c_point_plane.cuda()
        p_point_plane = p_point_plane.cuda()
        # print("before all_label_person_shape = ", torch.unique(all_label_person_shape))

        arm1_mask = (label_parse == 14).float()
        arm2_mask = (label_parse == 15).float()

        grid, theta = self.netGMM(agnostic, cloth)
        compute_c_point_plane = self.compute_grid_point(p_point_plane, grid)
        warped_cloth = F.grid_sample(cloth, grid, padding_mode='border')
        warped_mask = F.grid_sample(cloth_mask, grid, padding_mode='zeros')

        loss_warped_cloth = self.warped_criterionL1(warped_cloth, im_c)
        loss_point = self.point_criterionL1(compute_c_point_plane, c_point_plane)


        input_label, input_label_no_cloth_mask, input_remove_arm_parse = self.encode_input(label_parse, label_cloth_mask, remove_arm_parse)  
        origin_fake_label_cloth_mask = fake_label_cloth_mask
        # print("fake_label_cloth_mask unique = ", torch.unique(fake_label_cloth_mask))
        fake_cl_dis = morpho(fake_label_cloth_mask, 1, True)
        new_arm1_mask = (arm_label_map == 14).float()
        new_arm2_mask = (arm_label_map == 15).float()
        # print("new_arm1_mask = ", torch.unique(new_arm1_mask))
        # print("new_arm2_mask = ", torch.unique(new_arm2_mask))
        fake_cl_dis = fake_cl_dis * (1 - new_arm1_mask) * (1 - new_arm2_mask)
        #print("all_label_person_shape = ", torch.unique(all_label_person_shape))
        fake_cl_dis *= all_label_person_shape
        #print("3 fake_cl_dis = ", torch.unique(fake_cl_dis))

        arm1_occ = fake_label_cloth_mask * new_arm1_mask
        arm2_occ = fake_label_cloth_mask * new_arm2_mask
        bigger_arm1_occ = morpho(arm1_occ, 10)
        bigger_arm2_occ = morpho(arm2_occ, 10)
        #print("1 unique = ", torch.unique(arm_label_map))
        arm1_full = arm1_occ + (1 - fake_label_cloth_mask) * new_arm1_mask
        arm2_full = arm2_occ + (1 - fake_label_cloth_mask) * new_arm2_mask
        #print("2 unique = ", torch.unique(arm_label_map))
        arm_label_map *= (1 - new_arm1_mask)
        arm_label_map *= (1 - new_arm2_mask)

        arm_label_map = arm_label_map * (1 - arm1_full) + arm1_full * 14
        arm_label_map = arm_label_map * (1 - arm2_full) + arm2_full * 15

        arm_label_map *= (1 - fake_cl_dis)

        dis_label = encoder(arm_label_map, arm_label_map.shape)
        skin_color = self.ger_average_color((arm1_mask + arm2_mask - arm2_mask * arm1_mask),
                                            (arm1_mask + arm2_mask - arm2_mask * arm2_mask) * real_image)

        occlude = (1 - bigger_arm1_occ * (arm2_mask + arm1_mask + fake_label_cloth_mask)) * (1 - bigger_arm2_occ * (arm2_mask + arm1_mask + fake_label_cloth_mask))
        img_hole_hand = real_image * (1 - label_cloth_mask) * occlude * (1 - fake_cl_dis)
        # print("img_hole_hand = ", img_hole_hand)
        # print("dis_label = ", dis_label)
        # print("warped_cloth = ", warped_cloth)
        # print("skin_color = ", skin_color)
        if self.add_noise:
            G_in = torch.cat([img_hole_hand, dis_label, warped_cloth, skin_color, self.gen_noise(all_label_person_shape.size())], 1)
        else:
            G_in = torch.cat([img_hole_hand, dis_label, warped_cloth, skin_color], 1)
        #G_in = torch.cat([img_hole_hand, dis_label, warped_cloth, skin_color], 1)
        fake_image = self.netG.refine(G_in)
        fake_image = self.tanh(fake_image)
        loss_l1 = self.criterionL1(fake_image, real_image)
        loss_vgg = self.criterionVGG(fake_image, real_image)
         # Fake Generation
        # if self.use_features:
        #     if not self.opt.load_features:
        #         feat_map = self.netE.forward(real_image, inst_map)                     
        #     input_concat = torch.cat((input_label, feat_map), dim=1)                        
        # else:
        #     input_concat = input_label
        # fake_image = self.netG.forward(input_concat)

        # Fake Detection and Loss
        # print("G1 in = ", G1_in.size())
        # print("arm label size = ", arm_label.size())
        if self.add_noise:
            G_in_d = torch.cat([img_hole_hand, dis_label, warped_cloth.detach(), skin_color, self.gen_noise(all_label_person_shape.size())], 1)
        else:
            G_in_d = torch.cat([img_hole_hand, dis_label, warped_cloth.detach(), skin_color], 1)
        pred_fake_pool_D = self.discriminate_D(G_in_d, fake_image, use_pool=True)
        loss_D_fake = self.criterionGAN(pred_fake_pool_D, False)        

        # Real Detection and Loss
        # real_parse = (label_parse * (1 - cloth_mask))
        # print("real_parse size = ", real_parse.size())
        # print("G1 in size = ", G1_in.size())
        # print("input_label_no_cloth_mask size = ", input_label_no_cloth_mask.size())      
        pred_real_D = self.discriminate_D(G_in_d, real_image)
        loss_D_real = self.criterionGAN(pred_real_D, True)

        # # GAN loss (Fake Passability Loss)        
        pred_fake_D = self.netD.forward(torch.cat((G_in_d, fake_image), dim=1))        
        loss_G_GAN = self.criterionGAN(pred_fake_D, True)

        # print("loss_l1 = ", loss_l1)
        # print("loss_vgg = ", loss_vgg)
        loss_G_other = loss_l1 + loss_vgg * self.vgg_weight + loss_warped_cloth + self.point_loss_weight * loss_point
        # print("loss_l1 = ", loss_l1)
        # print("loss_vgg = ", loss_vgg * self.vgg_weight)
        # print("warped_cloth = ", loss_warped_cloth)
        # print("loss_point = ", self.point_loss_weight * loss_point)



        
        # Only return the fake_B image if necessary to save BW
        return [ self.loss_filter( loss_G_GAN, loss_G_other, loss_D_real, loss_D_fake ), None if not infer else img_hole_hand,
                                                                                      None if not infer else arm_label_map,
                                                                                      None if not infer else warped_cloth,
                                                                                      None if not infer else skin_color,
                                                                                      None if not infer else fake_image,
                                                                                      None if not infer else real_image,
                                                                                      None if not infer else im_c]
    def inference(self, label_parse, cloth, cloth_mask, agnostic, im_c, c_point_plane, p_point_plane, arm_label_map, fake_label_cloth_mask, 
            remove_arm_parse, label_cloth_mask, real_image, pose, all_label_person_shape):
        # Encode Inputs 
        self.netG.eval()       
        label_parse = label_parse.cuda()
        cloth = cloth.cuda()
        cloth_mask = cloth_mask.cuda()
        agnostic = agnostic.cuda()
        im_c = im_c.cuda()
        arm_label_map = arm_label_map.cuda()
        # print("before unique = ", torch.unique(arm_label_map))
        fake_label_cloth_mask = fake_label_cloth_mask.cuda()
        remove_arm_parse = remove_arm_parse.cuda()
        label_cloth_mask = label_cloth_mask.cuda()
        real_image = real_image.cuda()
        pose = pose.cuda()
        all_label_person_shape = all_label_person_shape.cuda()
        c_point_plane = c_point_plane.cuda()
        p_point_plane = p_point_plane.cuda()
        # print("before all_label_person_shape = ", torch.unique(all_label_person_shape))

        arm1_mask = (label_parse == 14).float()
        arm2_mask = (label_parse == 15).float()

        grid, theta = self.netGMM(agnostic, cloth)
        compute_c_point_plane = self.compute_grid_point(p_point_plane, grid)
        warped_cloth = F.grid_sample(cloth, grid, padding_mode='border')
        warped_mask = F.grid_sample(cloth_mask, grid, padding_mode='zeros')


        input_label, input_label_no_cloth_mask, input_remove_arm_parse = self.encode_input(label_parse, label_cloth_mask, remove_arm_parse)  
        origin_fake_label_cloth_mask = fake_label_cloth_mask
        # print("fake_label_cloth_mask unique = ", torch.unique(fake_label_cloth_mask))
        fake_cl_dis = morpho(fake_label_cloth_mask, 1, True)
        new_arm1_mask = (arm_label_map == 14).float()
        new_arm2_mask = (arm_label_map == 15).float()
        # print("new_arm1_mask = ", torch.unique(new_arm1_mask))
        # print("new_arm2_mask = ", torch.unique(new_arm2_mask))
        fake_cl_dis = fake_cl_dis * (1 - new_arm1_mask) * (1 - new_arm2_mask)
        #print("all_label_person_shape = ", torch.unique(all_label_person_shape))
        fake_cl_dis *= all_label_person_shape
        #print("3 fake_cl_dis = ", torch.unique(fake_cl_dis))

        arm1_occ = fake_label_cloth_mask * new_arm1_mask
        arm2_occ = fake_label_cloth_mask * new_arm2_mask
        bigger_arm1_occ = morpho(arm1_occ, 10)
        bigger_arm2_occ = morpho(arm2_occ, 10)
        #print("1 unique = ", torch.unique(arm_label_map))
        arm1_full = arm1_occ + (1 - fake_label_cloth_mask) * new_arm1_mask
        arm2_full = arm2_occ + (1 - fake_label_cloth_mask) * new_arm2_mask
        #print("2 unique = ", torch.unique(arm_label_map))
        arm_label_map *= (1 - new_arm1_mask)
        arm_label_map *= (1 - new_arm2_mask)

        arm_label_map = arm_label_map * (1 - arm1_full) + arm1_full * 14
        arm_label_map = arm_label_map * (1 - arm2_full) + arm2_full * 15

        arm_label_map *= (1 - fake_cl_dis)

        dis_label = encoder(arm_label_map, arm_label_map.shape)
        skin_color = self.ger_average_color((arm1_mask + arm2_mask - arm2_mask * arm1_mask),
                                            (arm1_mask + arm2_mask - arm2_mask * arm2_mask) * real_image)

        occlude = (1 - bigger_arm1_occ * (arm2_mask + arm1_mask + fake_label_cloth_mask)) * (1 - bigger_arm2_occ * (arm2_mask + arm1_mask + fake_label_cloth_mask))
        img_hole_hand = real_image * (1 - label_cloth_mask) * occlude * (1 - fake_cl_dis)
        # print("img_hole_hand = ", img_hole_hand)
        # print("dis_label = ", dis_label)
        # print("warped_cloth = ", warped_cloth)
        # print("skin_color = ", skin_color)
        if self.add_noise:
            G_in = torch.cat([img_hole_hand, dis_label, warped_cloth, skin_color, self.gen_noise(all_label_person_shape.size())], 1)
        else:
            G_in = torch.cat([img_hole_hand, dis_label, warped_cloth, skin_color], 1)
        #G_in = torch.cat([img_hole_hand, dis_label, warped_cloth, skin_color], 1)
        fake_image = self.netG.refine(G_in)
        fake_image = self.tanh(fake_image)
             
           
        # if torch.__version__.startswith('0.4'):
        #     with torch.no_grad():
        #         fake_image = self.netG.forward(input_concat)
        # else:
        #     fake_image = self.netG.forward(input_concat)
        return img_hole_hand, arm_label_map, warped_cloth, skin_color, fake_image, real_image

    def sample_features(self, inst): 
        # read precomputed feature clusters 
        cluster_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, self.opt.cluster_path)        
        features_clustered = np.load(cluster_path, encoding='latin1').item()

        # randomly sample from the feature clusters
        inst_np = inst.cpu().numpy().astype(int)                                      
        feat_map = self.Tensor(inst.size()[0], self.opt.feat_num, inst.size()[2], inst.size()[3])
        for i in np.unique(inst_np):    
            label = i if i < 1000 else i//1000
            if label in features_clustered:
                feat = features_clustered[label]
                cluster_idx = np.random.randint(0, feat.shape[0]) 
                                            
                idx = (inst == int(i)).nonzero()
                for k in range(self.opt.feat_num):                                    
                    feat_map[idx[:,0], idx[:,1] + k, idx[:,2], idx[:,3]] = feat[cluster_idx, k]
        if self.opt.data_type==16:
            feat_map = feat_map.half()
        return feat_map

    def encode_features(self, image, inst):
        image = Variable(image.cuda(), volatile=True)
        feat_num = self.opt.feat_num
        h, w = inst.size()[2], inst.size()[3]
        block_num = 32
        feat_map = self.netE.forward(image, inst.cuda())
        inst_np = inst.cpu().numpy().astype(int)
        feature = {}
        for i in range(self.opt.label_nc):
            feature[i] = np.zeros((0, feat_num+1))
        for i in np.unique(inst_np):
            label = i if i < 1000 else i//1000
            idx = (inst == int(i)).nonzero()
            num = idx.size()[0]
            idx = idx[num//2,:]
            val = np.zeros((1, feat_num+1))                        
            for k in range(feat_num):
                val[0, k] = feat_map[idx[0], idx[1] + k, idx[2], idx[3]].data[0]            
            val[0, feat_num] = float(num) / (h * w // block_num)
            feature[label] = np.append(feature[label], val, axis=0)
        return feature

    def get_edges(self, t):
        edge = torch.cuda.ByteTensor(t.size()).zero_()
        edge[:,:,:,1:] = edge[:,:,:,1:] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,:,:-1] = edge[:,:,:,:-1] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,1:,:] = edge[:,:,1:,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        edge[:,:,:-1,:] = edge[:,:,:-1,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        if self.opt.data_type==16:
            return edge.half()
        else:
            return edge.float()

    def save(self, which_epoch):
        self.save_network(self.netGMM, 'GMM', which_epoch, self.gpu_ids)
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)


    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())
        if self.gen_features:
            params += list(self.netE.parameters())           
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        if self.opt.verbose:
            print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd        
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

class InferenceModel(Pix2PixHDModel):
    def forward(self, inp):
        label, inst = inp
        return self.inference(label, inst)

        
