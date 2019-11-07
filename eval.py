import tensorflow as tf
import numpy as np
import glob
from PIL import Image

im1 = tf.placeholder(tf.uint8, [None, None, None, None])
im2 = tf.placeholder(tf.uint8, [None, None, None, None])
psnr = tf.image.psnr(im1, im2, max_val=255)
ssim = tf.image.ssim(im1, im2, max_val=255)
sess = tf.Session()

dataset = 'L2R_good_with_depth'
for file in glob.glob('results/%s/val_sync/images/*_ground_truth.png' % dataset):
	gt = str(file)
	en = gt.replace('_ground_truth', '_encoded')
	sa = gt.replace('_ground_truth', '_encoded_satellite')
	# Read images from file.
	gts.append(np.array(Image.open(gt)))
	sas.append(np.array(Image.open(sa)))

print(sess.run([psnr, ssim], feed_dict={im1: gts, im2: sas}))
