import tensorflow as tf
import numpy as np
import glob
from PIL import Image

def readImageList(path):
	li = sorted(glob.glob(path))
	return [np.array(Image.open(item).resize((512, 256))) for item in li]

im1 = tf.placeholder(tf.uint8, [None, None, None, None])
im2 = tf.placeholder(tf.uint8, [None, None, None, None])
psnr = tf.image.psnr(im1, im2, max_val=255)
ssim = tf.image.ssim(im1, im2, max_val=255)
sess = tf.Session()

xh1  = readImageList('/home/zoli/xiaohu_new_data/comparison/*_pred_rgb_init.png')
xh2  = readImageList('/home/zoli/xiaohu_new_data/comparison/*_pred_rgb_dll.png')
xh3  = readImageList('/home/zoli/xiaohu_new_data/comparison/*_pred_rgb_finetune.png')
cvpr = readImageList('/home/zoli/xiaohu_new_data/comparison/*_pred_rgb_2018.png')
p2p  = readImageList('/home/zoli/xiaohu_new_data/comparison/*_pred_rgb_p2p.png')
gt   = readImageList('/home/zoli/xiaohu_new_data/comparison/*_street_pano.png')

for item in [xh1, xh2, xh3, cvpr, p2p]:
	psnr_val, ssim_val = sess.run([psnr, ssim], feed_dict={im1: gt, im2: item})
	print('PSNR: ', psnr_val.mean())
	print('SSIM: ', ssim_val.mean())
	print()
