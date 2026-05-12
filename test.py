import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from itertools import islice
from util import html

# options
opt = TestOptions().parse()
opt.num_threads = 0  # test code only supports num_threads=1
opt.batch_size = 1  # test code only supports batch_size=1
opt.serial_batches = True  # no shuffle

# create IRImages
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
model.setup(opt)
model.eval()
print('Loading model %s' % opt.model)

# create website
web_dir = os.path.join(opt.results_dir, opt.phase + '_sync' if opt.sync else opt.phase)
webpage = html.HTML(web_dir, 'Training = %s, Phase = %s, Class =%s' % (opt.name, opt.phase, opt.name))

# test stage
for i, data in enumerate(dataset):
    model.set_input(data)
    print('process input image %3.3d/%3.3d' % (i, len(dataset)))
    if not opt.sync:
        t_samples = [-100, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                     17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    for nn in t_samples:
        T = None if nn == -100 else nn
        real_A, fake_B, real_B = model.test(T)
        if nn == -100:
            images = [real_A, real_B, fake_B]
            names = ['input', 'ground truth', 'encoded']
        else:
            images.append(fake_B)
            names.append('sample_%2.2d' % nn)

    img_path = 'input_%3.3d' % i
    save_images(webpage, images, names, img_path, aspect_ratio=opt.aspect_ratio, width=opt.crop_size)

webpage.save()
