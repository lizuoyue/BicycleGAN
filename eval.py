import tensorflow as tf
import numpy as np
import glob
from PIL import Image

im1 = tf.placeholder(tf.uint8, [None, None, None, None])
im2 = tf.placeholder(tf.uint8, [None, None, None, None])
psnr = tf.image.psnr(im1, im2, max_val=255)
ssim = tf.image.ssim(im1, im2, max_val=255)
sess = tf.Session()

# li=['51.49461961891024,-0.12526567147165224,338.7314758300781',
# '51.49705679788585,-0.13829752367507808,249.71269226074222',
# '51.49882231691279,-0.13130842639066032,239.08230590820307',
# '51.50002178361034,-0.08153489310416262,344.699951171875',
# '51.50096975802277,-0.11590477427841961,166.5968322753906',
# '51.50104273116423,-0.11592224428875397,15.560152053833011',
# '51.50128293722674,-0.08203508503117973,339.9909057617188',
# '51.50199971315091,-0.09245542280564223,204.7879638671875',
# '51.50220085642118,-0.08241487109614809,345.8883056640625',
# '51.50405142475574,-0.11426075542362923,275.4901428222656',
# '51.50562249414822,-0.12453453944840476,194.8160858154297',
# '51.50581077818933,-0.12600205582839408,67.17388153076173',
# '51.50614032157811,-0.09530486579558328,3.208474159240722',
# '51.50757318168793,-0.13135930999499124,238.85107421874994',
# '51.50803216362053,-0.10686987471876819,236.47779846191406',
# '51.50957238727601,-0.08716631144341136,11.81525421142578',
# '51.50992684740172,-0.08128581223331821,120.11615753173828',
# '51.50994107754272,-0.13005288796944114,170.2737579345703',
# '51.51012272705364,-0.06819902708207337,161.99368286132812',
# '51.51123922634032,-0.07521698323068904,348.16534423828125',
# '51.5112668301138,-0.09293147755226983,187.8216552734375',
# '51.51158379552164,-0.0723433525818109,351.5007629394531',
# '51.51200747588516,-0.06633529723183074,179.65267944335943',
# '51.51254568942095,-0.07250087544070993,342.9503784179688',
# '51.51281126709726,-0.0756455743795641,350.8323059082031',
# '51.5131238335865,-0.10797997651502556,176.67709350585938',
# '51.51376218047177,-0.13417872175386947,336.6068420410156',
# '51.51386067462892,-0.10195641907534991,238.2407836914062',
# '51.51399931538243,-0.12293529433088679,223.25317382812497',
# '51.51400864721708,-0.10738178915221397,358.47189331054693',
# '51.51410113041543,-0.1394290971564942,4.986813068389893',
# '51.51417160243486,-0.06169401022407328,171.9039001464844',
# '51.51440870469823,-0.14158025756353254,229.0980224609375',
# '51.51457594201497,-0.07852851998734423,308.0751647949219',
# '51.51528741326723,-0.0651884209689797,354.36859130859386',
# '51.5157003550564,-0.10930250520368645,348.21749877929693',
# '51.5160009871691,-0.1046384696385303,101.2675476074219',
# '51.51661691532991,-0.08973237468785555,-3.581604361534118',
# '51.5167509515753,-0.08972454714569267,2.118565559387207',
# '51.5170675020608,-0.14268941203977192,163.5618133544922',
# '51.51712806886996,-0.09861586642455222,357.4749755859375',
# '51.51759456615886,-0.07373468371974923,153.3416137695312',
# '51.51783237946995,-0.07103441969729829,333.61645507812494',
# '51.51816314536884,-0.07989877470367901,24.03661918640137',
# '51.51902653365302,-0.11057258265543624,2.122227191925049',
# '51.51912823381299,-0.12138584237879968,323.8551940917969',
# '51.51920513600567,-0.11056264702983754,1.7700691223144527',
# '51.51941348153292,-0.09022924575924662,20.0032901763916',
# '51.52010330672504,-0.14129234564438775,337.21646118164057',
# '51.52036507836065,-0.0925133986468154,193.2580871582031',
# '51.5217468237011,-0.08982273291883303,1.8282723426818848',
# '51.52175486004469,-0.07330963173251348,14.048789978027342',
# '51.52195621787808,-0.10682751913759603,334.1705322265625',
# '51.52234021143187,-0.10948730775919557,198.06825256347656',
# '51.52235064144221,-0.11725185533350668,158.7006225585938',
# '51.52254221597414,-0.14235845085704568,162.65011596679693',
# '51.52422523647335,-0.12746597408806792,320.8424377441407',
# '51.52451633207743,-0.1360857242966631,145.9485473632812',
# '51.52523852482189,-0.13253181404695624,147.4909057617188',
# '51.52539007479157,-0.08236395777885264,312.958251953125',
# '51.52597366629382,-0.13081762883257397,325.02746582031256',
# '51.52630736412968,-0.1367135981267893,145.7978363037109',
# '51.52706868073723,-0.09348542480449851,241.43681335449222',
# '51.52713153741079,-0.08362327644636025,183.3238067626953']

dataset = 'L2R_good_with_depth'
gts, ens, sas, xhs = [], [], [], []
for file, key in zip(sorted(glob.glob('results/%s/val_sync/images/*_ground_truth.png' % dataset)), li):
	gt = str(file)
	en = gt.replace('_ground_truth', '_encoded')
	sa = gt.replace('_ground_truth', '_encoded_satellite')
	xh = '../xiaohu_new_data/predict_of_train/%s_pred_rgb.png' % key
	# Read images from file.
	gts.append(np.array(Image.open(gt)))
	ens.append(np.array(Image.open(en)))
	sas.append(np.array(Image.open(sa)))
	xhs.append(np.array(Image.open(xh)))

psnr_val, ssim_val = sess.run([psnr, ssim], feed_dict={im1: gts, im2: xhs})
print('PSNR: ', psnr_val.mean())
print('SSIM: ', ssim_val.mean())
print()
psnr_val, ssim_val = sess.run([psnr, ssim], feed_dict={im1: gts, im2: ens})
print('PSNR: ', psnr_val.mean())
print('SSIM: ', ssim_val.mean())
print()
psnr_val, ssim_val = sess.run([psnr, ssim], feed_dict={im1: gts, im2: sas})
print('PSNR: ', psnr_val.mean())
print('SSIM: ', ssim_val.mean())
