from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cv2

def alg_keypoints(imgs:Path,sift_th=None,surf_th=None,orb_th=None):
	sift_data, surf_data, orb_data = [], [], []
	for item in list(imgs.iterdir()):
		img=cv2.imread(item.as_posix(), cv2.IMREAD_GRAYSCALE)
		if sift_th!=None and surf_th!=None and orb_th!=None:
			sift = cv2.xfeatures2d.SIFT_create(edgeThreshold=sift_th)
			surf = cv2.xfeatures2d.SURF_create(hessianThreshold=surf_th)
			orb = cv2.ORB_create(edgeThreshold=orb_th)
		else:
			sift = cv2.xfeatures2d.SIFT_create()
			surf = cv2.xfeatures2d.SURF_create()
			orb = cv2.ORB_create()

		kp_sift, dsc_sift = sift.detectAndCompute(img, None)
		kp_surf, dsc_surf = surf.detectAndCompute(img, None)
		kp_orb, dsc_orb = orb.detectAndCompute(img, None)
		sift_data.append(len(kp_sift))
		surf_data.append(len(kp_surf))
		orb_data.append(len(kp_orb))

	return sift_data, surf_data, orb_data

def chart(sift_data, surf_data, orb_data, label:str, th):
	labels_dark = ['exit-0','exit-32','exit-64','exit-96']
	labels_oryginal = ['exit-128',]
	labels_light = ['exit-160','exit-192','exit-224','exit-255']
	rects=[]

	if label=='light':
		labels = labels_light
	elif label == 'dark':
		labels= labels_dark
	elif label == 'oryginal':
		labels = labels_oryginal

	x = np.arange(len(labels))

	fig, ax = plt.subplots()
	rects1 = ax.bar(x + 0.00, sift_data, width = 0.25, label='SIFT')
	rects.append(rects1)
	rects2 = ax.bar(x + 0.25, surf_data, width = 0.25, label='SURF')
	rects.append(rects2)
	rects3 = ax.bar(x + 0.50, orb_data, width = 0.25, label='ORB')
	rects.append(rects3)


	ax.set_ylabel('Number of key points')
	if th == 0:
		ax.set_xlabel('Threshold = default')
	else:
		ax.set_xlabel(f'Threshold = {th}%')
	ax.set_xticks(x)
	ax.set_xticklabels(labels)
	ax.legend()
	ax.legend()

	for rect in rects:
		for item in rect:
			height = item.get_height()
			ax.annotate('{}'.format(height),xy=(item.get_x() + item.get_width() / 2, height),xytext=(0, 3),  textcoords="offset points",ha='center', va='bottom')

	fig.tight_layout()

	plt.show()


if __name__ == '__main__':

	imgs = Path(r'C:\Users\tomsk\Desktop\alghoritms\images')

	while True:
		res=input('Type number:\n'
		          '0 - Threshold default\n'
		          '25 - Threshold = 25%\n'
		          '50 - Threshold = 50%\n'
		          '75 - Threshold = 75%\n\n'
		          'Exit: (Y/y)\n')


		if res == 'Y' or res == 'y':
			exit()

		elif res == '0':
			sift_data,surf_data,orb_data = alg_keypoints(imgs)

		elif res == '25':
			sift_data,surf_data,orb_data = alg_keypoints(imgs,50,2250,73)

		elif res == '50':
			sift_data,surf_data,orb_data = alg_keypoints(imgs,100,4500,145)

		elif res == '75':
			sift_data,surf_data,orb_data = alg_keypoints(imgs,150,6750,218)

		else:
			print('Allowed numbers: 0, 25, 50, 75')


		if res in ['0','25','50','75']:
			data={'dark':[sift_data[:4], surf_data[:4], orb_data[:4]],
			      'oryginal':[sift_data[4], surf_data[4], orb_data[4]],
			      'light':[sift_data[5:], surf_data[5:], orb_data[5:]]}
			for k,v in data.items():
				chart(*v,k,int(res))