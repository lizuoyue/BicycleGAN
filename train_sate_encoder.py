import torch
from models import networks

class option(object):
	def __init__(self):
		self.output_nc = 3
		self.nz = 32
		self.nef = 96
		self.netE = ''
		self.norm = ''
		self.nl = ''
		self.init_type = 0
		self.init_gain = 0.02

		self.gpu_ids = []
		self.use_vae = True

		self.lr = 
		self.beta1 = 
		self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
		self.netE_path =
		self.batch_size = 
		self.epoch = 
		self.
		return

if __name__=='__main__':
	opt = option()
	d = opt.
	netE = networks.define_E(opt.output_nc, opt.nz, opt.nef, netE=opt.netE, norm=opt.norm, nl=opt.nl,
							init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=opt.gpu_ids, vaeLike=opt.use_vae)
	criterionL1 = torch.nn.L1Loss()
	optimizer_E = torch.optim.Adam(self.netE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

	for item in []:
		rgb = None
		sate = None
		optimizer.zero_grad()
		z_target = netE(rgb)
		z_pred = netE(sate)

		loss = criterionL1(z_pred, z_target)
		loss.backward()
		optimizer_E.step()