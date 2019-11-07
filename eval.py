import tensorflow as tf
import glob

dataset = 'L2R_good_with_depth'
for file in glob.glob('./results/%s/val_sync/images/*_ground_truth.png' % dataset):
	gt = file
	en = file.replace('_ground_truth', '_encoded')
	sa = file.replace('_ground_truth', '_encoded_satellite')
	# Read images from file.
	im1 = tf.io.decode_png(gt)
	im2 = tf.io.decode_png(sa)
	# Compute SSIM over tf.uint8 Tensors.
	psnr = tf.image.psnr(im1, im2, max_val=255)
	ssim = tf.image.ssim(im1, im2, max_val=255)
	print(psnr, ssim)
