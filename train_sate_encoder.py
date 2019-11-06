import torch
import torchvision
import random
import numpy as np
from models import networks

class option(object):
	def __init__(self):
		self.output_nc = 3
		self.nz = 32
		self.nef = 96
		self.netE = 'resnet_256'
		self.norm = 'instance'
		self.nl = 'relu'
		self.init_type = 'xavier'
		self.init_gain = 0.02

		self.gpu_ids = []
		self.use_vae = True

		self.lr = 0.0002
		self.beta1 = 0.5
		self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
		self.batch_size = 4
		self.epoch = 400
		return

if __name__=='__main__':
	opt = option()
	netE = networks.define_E(opt.output_nc, opt.nz, opt.nef, netE=opt.netE, norm=opt.norm, nl=opt.nl,
							init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=opt.gpu_ids, vaeLike=opt.use_vae)
	criterionL1 = torch.nn.L1Loss()
	optimizer_E = torch.optim.Adam(netE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

	transforms = torchvision.transforms.Compose([
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
	])

	d = np.load('encoded_z.npy')
	names = sorted(list(d.keys()))
	vectors = [d[name] for name in names]
	paths = ['/home/zoli/xiaohu_new_data/train2/%s_sate_rgb.png' % name for name in names]
	idx = [i for i in range(len(names))]

	random.seed(7)
	li = []
	for _ in range(opt.epoch):
		random.shuffle(idx)
		li += idx
	rounds = int(len(li) / opt.epoch)

	for i in range(rounds):
		beg = i * opt.batch_size
		end = beg + opt.batch_size
		choose = li[beg: end]

		optimizer.zero_grad()
		z_target = np.array([vectors[choose[j]] for j in range(opt.batch_size)])
		z_target = torch.from_numpy(z_target).to(opt.device)
		sate_rgb = [transforms(Image.open(paths[choose[j]])) for j in range(opt.batch_size)]
		sate_rgb = torch.stack(sate_rgb).to(opt.device)
		z_pred = netE(sate_rgb)

		print(z_target.shape)
		print(z_pred.shape)
		continue

		loss = criterionL1(z_pred, z_target)
		loss.backward()
		optimizer_E.step()


