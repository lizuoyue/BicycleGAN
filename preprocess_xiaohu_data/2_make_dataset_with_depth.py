from PIL import Image
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

target = '../datasets/L2R_with_depth'
os.makedirs(target, exist_ok = True)
for mode in ['/train', '/val']:
	os.makedirs(target + mode, exist_ok = True)
	source_mode = mode.replace('/', '').replace('val', 'test')
	sem_label_path = '/home/zoli/xiaohu_new_data/predict_of_%s/*_pred_sem_label.png' % source_mode
	ss = 'predict_of_%s' % source_mode
	tt = source_mode + '2'
	files = glob.glob(sem_label_path)
	for file in tqdm.tqdm(files):
		sem = np.array(Image.open(file).resize((512, 256), PIL.Image.BILINEAR))
		dep = np.array(Image.open(file.replace(ss, tt).replace('_pred_sem_label', '_proj_dis')).convert('L').resize((512, 256), PIL.Image.BILINEAR))
		rgb = np.array(Image.open(file.replace(ss, tt).replace('_pred_sem_label', '_street_rgb')).resize((512, 256), PIL.Image.BILINEAR))
		info = rgb.copy()
		info[..., 2] = dep
		for i in range(5):
			info[sem == i, :2] = two_dim[i]
		bi = np.concatenate([info, rgb], 1)
		basename = '/' + os.path.basename(file)
		Image.fromarray(bi).save(target + mode + basename.replace('_pred_sem_label', ''))
