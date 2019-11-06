import torch
from models import networks
from options.train_options import TrainOptions

if __name__=='__main__':
	opt = TrainOptions().parse()
	dataset = create_dataset(opt)

	netE = networks.define_E(opt.output_nc, opt.nz, opt.nef, netE=opt.netE, norm=opt.norm, nl=opt.nl,
							init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=opt.gpu_ids, vaeLike=opt.use_vae)
	criterionL1 = torch.nn.L1Loss()
	optimizer_E = torch.optim.Adam(netE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

	for item in []:
		rgb = None
		sate = None
		optimizer.zero_grad()
		z_target = netE(rgb)
		z_pred = netE(sate)

		loss = criterionL1(z_pred, z_target)
		loss.backward()
		optimizer_E.step()
