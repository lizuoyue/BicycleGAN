import os
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model

# options
opt = TrainOptions().parse()
opt.num_threads = 1   # test code only supports num_threads=1
opt.batch_size = 1   # test code only supports batch_size=1
opt.serial_batches = True  # no shuffle

# create dataset
dataset = create_dataset(opt)
model = create_model(opt)
model.setup(opt)
model.eval()
print('Loading model %s' % opt.model)

# test stage
for i, data in enumerate(dataset):
    print(data['A_paths'])
    model.set_input(data)
    print('process input image %3.3d/%3.3d' % (i, len(dataset)))
    continue
    if not opt.sync:
        z_samples = model.get_z_random(opt.n_samples + 1, opt.nz)
    for nn in range(opt.n_samples + 1):
        encode = nn == 0 and not opt.no_encode
        real_A, fake_B, real_B = model.test(z_samples[[nn]], encode=encode)
        if nn == 0:
            images = [real_A, real_B, fake_B]
            names = ['input', 'ground truth', 'encoded']
        else:
            images.append(fake_B)
            names.append('random_sample%2.2d' % nn)
