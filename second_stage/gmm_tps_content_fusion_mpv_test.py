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


    cloth, warped_cloth, refine_cloth, im_c, real_image = model.inference(Variable(data['label_parse']), Variable(data['cloth']), Variable(data['cloth_mask']), 
            Variable(data['agnostic']), Variable(data['parse_cloth']), Variable(data['cloth_points']), Variable(data['person_points']), Variable(data['arm_label_map']),
            Variable(data['fake_label_cloth_mask']), Variable(data['remove_arm_parse']), Variable(data['label_cloth_mask']), 
            Variable(data['image']), Variable(data['pose_map']), Variable(data['all_label_person_shape']))
        
    visuals = OrderedDict([('cloth', util.tensor2im(cloth.data[0])),
                            ('warped_cloth', util.tensor2im(warped_cloth.data[0])),
                            ('refine_cloth', util.tensor2im(refine_cloth.data[0])),
                            ('im_c', util.tensor2im(im_c.data[0])),
                            ('real_image', util.tensor2im(real_image.data[0]))])
    img_path = data['path']
    print('process image... %s' % img_path)
    visualizer.save_images(webpage, visuals, img_path)

webpage.save()
