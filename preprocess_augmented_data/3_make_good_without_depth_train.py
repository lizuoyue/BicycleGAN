from PIL import Image
import PIL
import glob
import numpy as np
import os
import tqdm

two_dim = np.array([
	[255, 255, 0],
	[  0, 127, 0],
	[127,   0, 0],
	[  0, 255, 0],
	[255,   0, 0],
])

sem_path = '/home/zoli/xiaohu_new_data/train_augment/train_*/*_pred_sem_label_*.png'
rgb_path = '/home/zoli/xiaohu_new_data/train_augment/train_*/*_street_rgb*.png'

sem_files = sorted(glob.glob(sem_path))
rgb_files = sorted(glob.glob(rgb_path))
assert(len(sem_files) == len(rgb_files))

target = '../datasets/L2R_aug_good_without_depth/'
os.makedirs(target, exist_ok = True)
for mode in ['train']:
	os.makedirs(target + mode, exist_ok = True)
	for sem_file, rgb_file in tqdm.tqdm(zip(sem_files, rgb_files), total=len(rgb_files)):
		sem = np.array(Image.open(sem_file).resize((512, 256), PIL.Image.BILINEAR))
		rgb = np.array(Image.open(rgb_file).resize((512, 256), PIL.Image.BILINEAR))
		info = rgb.copy()
		for i in range(5):
			info[sem == i] = two_dim[i]
		bi = np.concatenate([info, rgb], 1)
		basename = '/' + os.path.basename(rgb_file).replace('img_street_rgb', '')
		Image.fromarray(bi).save(target + mode + basename)
