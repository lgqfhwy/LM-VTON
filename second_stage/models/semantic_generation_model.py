import numpy as np
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import torch.nn as nn
import torch.nn.functional as F

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
             
        self.netG1 = networks.define_Refine(42, 20, gpu_ids=self.gpu_ids)   
        self.netG2 = networks.define_Refine(42, 1, gpu_ids=self.gpu_ids)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.BCE = torch.nn.BCEWithLogitsLoss()
        # Discriminator network
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = input_nc + opt.output_nc
            if not opt.no_instance:
                netD_input_nc += 1
            self.netD1 = networks.define_D(42 + 20, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, 
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)
            self.netD2 = networks.define_D(42 + 1, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, 
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
            self.load_network(self.netG1, 'G1', opt.which_epoch, pretrained_path)
            self.load_network(self.netG2, 'G2', opt.which_epoch, pretrained_path) 
            if self.isTrain:
                self.load_network(self.netD1, 'D1', opt.which_epoch, pretrained_path)
                self.load_network(self.netD2, 'D2', opt.which_epoch, pretrained_path)  
            

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
                params = list(self.netG1.parameters())
                params += list(self.netG2.parameters())
            # if self.gen_features:              
            #     params += list(self.netE.parameters())         
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))                            

            # optimizer D                        
            params = list(self.netD1.parameters())
            params += list(self.netD2.parameters())  
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))


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


    def discriminate_D1(self, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:            
            fake_query = self.fake_pool.query(input_concat)
            return self.netD1.forward(fake_query)
        else:
            return self.netD1.forward(input_concat)
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
    def discriminate_D2(self, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:            
            fake_query = self.fake_pool.query(input_concat)
            return self.netD2.forward(fake_query)
        else:
            return self.netD2.forward(input_concat)

    def forward(self, label_parse, cloth, cloth_mask, remove_arm_parse, label_cloth_mask, image, pose, infer=False):

        # print("label_parse dtype = ", label_parse.size())
        # print("cloth dtype = ", cloth.size())
        # print("cloth_mask dtype = ", cloth_mask.size())
        # print("remove_arm_parse dtype = ", remove_arm_parse.size())
        # print("label_cloth_mask dtype = ", label_cloth_mask.size())
        # print("image dtype = ", image.size())
        # print("pose dtype = ", pose.size())
        # Encode Inputs
        label_parse = label_parse.cuda()
        cloth = cloth.cuda()
        cloth_mask = cloth_mask.cuda()
        remove_arm_parse = remove_arm_parse.cuda()
        label_cloth_mask = label_cloth_mask.cuda()
        image = image.cuda()
        pose = pose.cuda()
        if self.no_background:
            cloth = cloth * cloth_mask
        #print("unique = ", torch.unique(label_cloth_mask))
        #print("unique = ", torch.unique(remove_arm_parse))
        input_label, input_label_no_cloth_mask, input_remove_arm_parse = self.encode_input(label_parse, label_cloth_mask, remove_arm_parse)  
        # label_parse = label_parse.cuda()
        # cloth = cloth.cuda()
        # cloth_mask = cloth_mask.cuda()
        # remove_arm_parse = remove_arm_parse.cuda()
        # label_cloth_mask = label_cloth_mask.cuda()
        # image = image.cuda()
        # pose = pose.cuda()
        G1_in = torch.cat([cloth_mask, cloth, input_remove_arm_parse, pose], 1)
        arm_label = self.netG1.refine(G1_in)
        arm_label = self.sigmoid(arm_label)
        real_g1_label = label_parse * (1 - label_cloth_mask)
        CE_loss_G1 = self.cross_entropy2d(arm_label, (label_parse * (1 - label_cloth_mask))[0].long())
        #CE_loss_G1 = 0
        arm_label_map = generate_discrete_label(arm_label.detach(), 20, False)
        #print("arm_label_map = ", arm_label_map.size())
        dis_label = generate_discrete_label(arm_label.detach(), 20)
        # print("arm_label_map size = ", arm_label_map.size())
        # print("dis_label size = ", dis_label.size())
        # stop

        G2_in = torch.cat([cloth_mask, cloth, dis_label, pose], 1)
        fake_cl = self.netG2.refine(G2_in)
        fake_cl = self.sigmoid(fake_cl)
        CE_loss_G2 = self.BCE(fake_cl, label_cloth_mask)
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
        pred_fake_pool_D1 = self.discriminate_D1(G1_in, arm_label, use_pool=True)
        loss_D1_fake = self.criterionGAN(pred_fake_pool_D1, False)        

        # Real Detection and Loss
        # real_parse = (label_parse * (1 - cloth_mask))
        # print("real_parse size = ", real_parse.size())
        # print("G1 in size = ", G1_in.size())
        # print("input_label_no_cloth_mask size = ", input_label_no_cloth_mask.size())      
        pred_real_D1 = self.discriminate_D1(G1_in, input_label_no_cloth_mask)
        loss_D1_real = self.criterionGAN(pred_real_D1, True)

        # # GAN loss (Fake Passability Loss)        
        pred_fake_D1 = self.netD1.forward(torch.cat((G1_in, arm_label), dim=1))        
        loss_G1_GAN = self.criterionGAN(pred_fake_D1, True)

        # Fake Detection and Loss
        pred_fake_pool_D2 = self.discriminate_D2(G2_in, fake_cl, use_pool=True)
        loss_D2_fake = self.criterionGAN(pred_fake_pool_D2, False)        

        # # Real Detection and Loss        
        pred_real_D2 = self.discriminate_D2(G2_in, label_cloth_mask)
        loss_D2_real = self.criterionGAN(pred_real_D2, True)

        # # GAN loss (Fake Passability Loss)        
        pred_fake_D2 = self.netD2.forward(torch.cat((G2_in, fake_cl), dim=1))        
        loss_G2_GAN = self.criterionGAN(pred_fake_D2, True)

        loss_G_GAN = loss_G1_GAN + loss_G2_GAN
        loss_D_real = loss_D1_real + loss_D2_real
        loss_D_fake = loss_D1_fake + loss_D2_fake
        loss_G_CE = CE_loss_G1 + CE_loss_G2
        # loss_G_GAN = loss_G1_GAN = 0
        # loss_D_real = loss_D1_real = 0
        # loss_D_fake = loss_D1_fake = 0
        # loss_G_CE = CE_loss_G1 = 0

        # print("loss_G_CE = ", loss_G_CE.dtype)


        #stop

        # GAN feature matching loss
        # loss_G_GAN_Feat = 0
        # if not self.opt.no_ganFeat_loss:
        #     feat_weights = 4.0 / (self.opt.n_layers_D + 1)
        #     D_weights = 1.0 / self.opt.num_D
        #     for i in range(self.opt.num_D):
        #         for j in range(len(pred_fake[i])-1):
        #             loss_G_GAN_Feat += D_weights * feat_weights * \
        #                 self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat
                   
        # VGG feature matching loss

        # if not self.opt.no_vgg_loss:
        #     loss_G_VGG = self.criterionVGG(fake_image, real_image) * self.opt.lambda_feat
        
        # Only return the fake_B image if necessary to save BW
        return [ self.loss_filter( loss_G_GAN, loss_G_CE, loss_D_real, loss_D_fake ), None if not infer else arm_label_map,
                                                                                      None if not infer else real_g1_label,
                                                                                      None if not infer else fake_cl,
                                                                                      None if not infer else label_cloth_mask,
                                                                                      None if not infer else cloth]
    def inference(self, label_parse, cloth, cloth_mask, remove_arm_parse, label_cloth_mask, image, pose):
        # Encode Inputs
        self.netG1.eval()
        self.netG2.eval()       
        label_parse = label_parse.cuda()
        cloth = cloth.cuda()
        cloth_mask = cloth_mask.cuda()
        remove_arm_parse = remove_arm_parse.cuda()
        label_cloth_mask = label_cloth_mask.cuda()
        image = image.cuda()
        pose = pose.cuda()

        if self.no_background:
            cloth = cloth * cloth_mask             
        input_label, input_label_no_cloth_mask, input_remove_arm_parse = self.encode_input(label_parse, label_cloth_mask, remove_arm_parse) 
        G1_in = torch.cat([cloth_mask, cloth, input_remove_arm_parse, pose], 1)
        arm_label = self.netG1.refine(G1_in)
        arm_label = self.sigmoid(arm_label)
        real_g1_label = label_parse * (1 - label_cloth_mask)
        arm_label_map = generate_discrete_label(arm_label.detach(), 20, False)
        unique_num = torch.unique(arm_label_map)
        num_flag = torch.any(unique_num > 19)
        if num_flag:
            print("unique_num = ", unique_num)
            assert false
        dis_label = generate_discrete_label(arm_label.detach(), 20)
        G2_in = torch.cat([cloth_mask, cloth, dis_label, pose], 1)
        fake_cl = self.netG2.refine(G2_in)
        fake_cl = self.sigmoid(fake_cl)

       
        return arm_label_map, real_g1_label, fake_cl, label_cloth_mask, cloth, remove_arm_parse

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
        self.save_network(self.netG1, 'G1', which_epoch, self.gpu_ids)
        self.save_network(self.netD1, 'D1', which_epoch, self.gpu_ids)
        self.save_network(self.netG2, 'G2', which_epoch, self.gpu_ids)
        self.save_network(self.netD2, 'D2', which_epoch, self.gpu_ids)
        if self.gen_features:
            self.save_network(self.netE, 'E', which_epoch, self.gpu_ids)

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

        
