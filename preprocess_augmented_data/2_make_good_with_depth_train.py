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

sem_path = '/home/zoli/xiaohu_new_data/train_augment/train_*/*_street_label*.png'
rgb_path = '/home/zoli/xiaohu_new_data/train_augment/train_*/*_street_rgb.png'
dep_path = '/home/zoli/xiaohu_new_data/train_augment/train_*/*_proj_dis*.png'

sem_files = sorted(glob.glob(sem_path))
rgb_files = sorted(glob.glob(rgb_path))
dep_files = sorted(glob.glob(dep_path))

target = '../datasets/L2R_aug_good_with_depth/'
os.makedirs(target, exist_ok = True)
for mode in ['train']:
	os.makedirs(target + mode, exist_ok = True)
	for sem_file, rgb_file, dep_file in tqdm.tqdm(zip(sem_files, rgb_files, dep_files)):
		sem = np.array(Image.open(sem_file).resize((512, 256), PIL.Image.BILINEAR))
		dep = np.array(Image.open(dep_file).convert('L').resize((512, 256), PIL.Image.BILINEAR))
		rgb = np.array(Image.open(rgb_file).resize((512, 256), PIL.Image.BILINEAR))
		info = rgb.copy()
		info[..., 2] = dep
		for i in range(5):
			info[sem == i, :2] = two_dim[i]
		bi = np.concatenate([info, rgb], 1)
		basename = '/%s.png' % line 
		Image.fromarray(bi).save(target + mode + basename)