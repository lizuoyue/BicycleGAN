import os
import glob
import random

dataset_dir = '/home/zoli/xiaohu_new_data/train2' % mode
files = glob.glob(dataset_dir + '/*_sate_lable.png')
names = [os.path.basename(item).replace('.png', '') for item in files]
random.seed(7)
random.shuffle(names)
with open('test.txt', 'w') as f:
	for name in names[:64]:
		f.write('%s\n' % name)
with open('train.txt', 'w') as f:
	for name in names[64:]:
		f.write('%s\n' % name)
