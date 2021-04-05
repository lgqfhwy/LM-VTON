import os
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch

opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

# test
if not opt.engine and not opt.onnx:
    model = create_model(opt)
    if opt.data_type == 16:
        model.half()
    elif opt.data_type == 8:
        model.type(torch.uint8)
            
    if opt.verbose:
        print(model)
else:
    from run_engine import run_trt_engine, run_onnx
    
for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break


    arm_label_map, real_g1_label, fake_label_cloth_mask, real_label_cloth_mask, cloth, remove_arm_parse = model.inference(Variable(data['label_parse']), Variable(data['cloth']), Variable(data['cloth_mask']),
            Variable(data['remove_arm_parse']), Variable(data['label_cloth_mask']), 
            Variable(data['image']), Variable(data['pose_map']))
        
    visuals = OrderedDict([('visualize_arm_label_map_vis', util.tensor2label(arm_label_map.data[0], 20)),
                            ('visualize_arm_parse_vis', util.tensor2label(remove_arm_parse.data[0], 20)),
                            ('input_remove_parse', util.tensor2label(data['remove_arm_parse'][0], 20)),
                            ('real_g1_label', util.tensor2label(real_g1_label.data[0], 20)),
                            ('arm_label_map', util.tensor2im(arm_label_map.data[0], retain_origin = True)),
                            ('fake_label_cloth_mask', util.tensor2im(fake_label_cloth_mask.data[0])),
                            ('real_label_cm', util.tensor2im(real_label_cloth_mask.data[0])),
                           ('cloth_image', util.tensor2im(cloth.data[0]))])
    img_path = data['path']
    print('process image... %s' % img_path)
    visualizer.save_images(webpage, visuals, img_path)

webpage.save()
