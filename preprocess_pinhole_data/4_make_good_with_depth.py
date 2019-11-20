from PIL import Image
import PIL
import glob
import numpy as np
import os
import tqdm

two_dim = np.array([
	[255, 255],
	[  0, 127],
	[127,   0],
	[  0, 255],
	[255,   0],
])

target = '../datasets/L2R_pinhole/'
os.makedirs(target, exist_ok = True)
for mode in ['val']:
	os.makedirs(target + mode, exist_ok = True)

	sem_path = '/media/zhaopeng/data/Zuoyue/test_pinhole/*_pred_sem_label.png'
	rgb_path = '/media/zhaopeng/data/Zuoyue/test_pinhole/*_street_rgb.png'
	dep_path = '/media/zhaopeng/data/Zuoyue/test_pinhole/*_proj_dis.png'
	sem_files = sorted(glob.glob(sem_path))
	rgb_files = sorted(glob.glob(rgb_path))
	dep_files = sorted(glob.glob(dep_path))
	assert(len(sem_files) == len(rgb_files))
	assert(len(rgb_files) == len(dep_files))

	for sem_file, rgb_file, dep_file in tqdm.tqdm(list(zip(sem_files, rgb_files, dep_files))):
		sem = np.array(Image.open(sem_file).resize((256, 256), PIL.Image.BILINEAR))
		dep = np.array(Image.open(dep_file).convert('L').resize((256, 256), PIL.Image.BILINEAR))
		rgb = np.array(Image.open(rgb_file).resize((256, 256), PIL.Image.BILINEAR))
		info = rgb.copy()
		info[..., 2] = dep
		for i in range(5):
			info[sem == i, :2] = two_dim[i]
		bi = np.concatenate([info, rgb], 1)
		basename = '/%s.png' % os.path.basename(sem_file).replace('_pred_sem_label.png', '') 
		Image.fromarray(bi).save(target + mode + basename)
