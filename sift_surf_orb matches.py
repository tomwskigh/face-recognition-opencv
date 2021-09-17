from pathlib import Path
import matplotlib.pyplot as plt
import cv2

def alg_matches(imgs,label,th=None):
	files = list(Path(imgs).iterdir())
	org_img = Path(r'C:\Users\tomsk\Desktop\alghoritms\images\5.jpg')
	img1 = cv2.imread(org_img.as_posix(),cv2.IMREAD_GRAYSCALE)
	others_img = [item.as_posix() for item in files if item!=org_img]

	for img in others_img:
		img2=cv2.imread(img, cv2.IMREAD_GRAYSCALE)
		print(img)
		if label == 'SIFT':
			if th!=None:
				sift1 = cv2.xfeatures2d.SIFT_create(edgeThreshold=th)
				sift2 = cv2.xfeatures2d.SIFT_create(edgeThreshold=th)
			else:
				sift1 = cv2.xfeatures2d.SIFT_create()
				sift2 = cv2.xfeatures2d.SIFT_create()
			keypoints_1, descriptors_1 = sift1.detectAndCompute(img1,None)
			keypoints_2, descriptors_2 = sift2.detectAndCompute(img2,None)
			bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

		if label == 'SURF':
			if th!=None:
				surf1 = cv2.xfeatures2d.SURF_create(hessianThreshold=th)
				surf2 = cv2.xfeatures2d.SURF_create(hessianThreshold=th)
			else:
				surf1 = cv2.xfeatures2d.SURF_create()
				surf2 = cv2.xfeatures2d.SURF_create()
			keypoints_1, descriptors_1 = surf1.detectAndCompute(img1,None)
			keypoints_2, descriptors_2 = surf2.detectAndCompute(img2,None)
			bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

		if label == 'ORB':
			if th!=None:
				orb1 = cv2.ORB_create(edgeThreshold=th)
				orb2 = cv2.ORB_create(edgeThreshold=th)
			else:
				orb1 = cv2.ORB_create()
				orb2 = cv2.ORB_create()
			keypoints_1, descriptors_1 = orb1.detectAndCompute(img1,None)
			keypoints_2, descriptors_2 = orb2.detectAndCompute(img2,None)
			bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
		matches = bf.match(descriptors_1,descriptors_2)
		matches = sorted(matches, key = lambda x:x.distance)
		print(len(matches))
		img = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
		plt.imshow(img),plt.show()


if __name__ == '__main__':

	imgs =  Path(r'C:\Users\tomsk\Desktop\alghoritms\images')

	while True:
		res=input('Type number:\n'
				  '1 - SIFT => edgeThreshold = default\n'
				  '2 - SIFT => edgeThreshold = 25%\n'
				  '3 - SIFT => edgeThreshold = 50%\n'
				  '4 - SIFT => edgeThreshold = 75%\n'
				  '5 - SURF => edgeThreshold = default\n'
				  '6 - SURF => edgeThreshold = 25%\n'
				  '7 - SURF => edgeThreshold = 50%\n'
				  '8 - SURF => edgeThreshold = 75%\n'
				  '9 - ORB => edgeThreshold = default\n'
				  '10 - ORB => edgeThreshold = 25%\n'
				  '11 - ORB => edgeThreshold = 50%\n'
				  '12 - ORB => edgeThreshold = 75%\n'
				  'Exit: (Y/y)\n')

		data = {'1':[imgs,'SIFT'],'2':[imgs,'SIFT', 50],'3':[imgs,'SIFT', 100],'4':[imgs,'SIFT', 150],
		        '5':[imgs,'SURF'],'6':[imgs,'SURF', 2250],'7':[imgs,'SURF', 4500],'8':[imgs,'SURF', 6750],
		        '9':[imgs,'ORB'],'10':[imgs,'ORB', 73],'11':[imgs,'ORB', 145],'12':[imgs,'ORB', 218]}

		for k,v in data.items():
			if k == res:
				alg_matches(*v)

		if res == 'Y' or res == 'y':
			exit()

		else:
			print('Allowed numbers: 1,2,3,4,5,6,7,8,9,10,11,12')