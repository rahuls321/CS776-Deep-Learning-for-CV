# -*- coding: utf-8 -*-
"""Assignment 1-DLCV.ipynb
   Name - Rahul Kumar
   Roll - 21111069
   Course - CS776-Deep Learning for Computer Vision
"""

# from cgi import test
import pickle
import numpy as np
import os
import cv2
import math
import random
import matplotlib.pyplot as plt

from feature_extractor import BBResNet18
from mlp import Model, get_one_hot_vector

def unpickle(datapaths):
	labels_mapping={}
	train_data={'data':np.array([]), 'labels':[]}
	test_data={'data':np.array([]), 'labels':[]}
	for file in os.listdir(datapaths):
		if(file=="data_batch_1" or file=="data_batch_2" or file=="data_batch_3" or file=="data_batch_4" or file=="data_batch_5"):
			print(file)
			with open(os.path.join(datapaths, file), 'rb') as fo:
				dict = pickle.load(fo, encoding='bytes')
			if(train_data['data'].shape[0]==0):
				train_data['data']=dict[b'data']
			else: train_data['data'] = np.vstack([train_data['data'], dict[b'data']])
			# train_data['filenames'] = train_data['filenames']+dict[b'filenames']
			train_data['labels'] = train_data['labels']+dict[b'labels']
		elif(file=='batches.meta'):
			print(file)
			with open(os.path.join(datapaths, file), 'rb') as fo:
				labels_mapping = pickle.load(fo, encoding='bytes')
		elif(file=='test_batch'):
			print(file)
			with open(os.path.join(datapaths, file), 'rb') as fo:
				dict = pickle.load(fo, encoding='bytes')
			if(test_data['data'].shape[0]==0):
				test_data['data']=dict[b'data']
			else: test_data['data'] = np.vstack([test_data['data'], dict[b'data']])
			# test_data['filenames'] = test_data['filenames']+dict[b'filenames']
			test_data['labels'] = test_data['labels']+dict[b'labels']
	return train_data, test_data, labels_mapping


def preprocessing(data):
	images=[]
	for img in data['data']:
		img_new = img.reshape((3, 32, 32))
		image = np.transpose(img_new, [1, 2, 0])
		images.append(image)
	return np.array(images)


"""## Question 2"""

##This function rotates the image around its center by random degree between [-180, 180].
def random_rotation(image):
	#Choose Random degree
	degree = random.randint(-180, 180)
	# print("Random degree chosen: ", degree)
	# First we will convert the degrees into radians
	rads = math.radians(degree)
	cosine = math.cos(rads)
	sine = math.sin(rads)
	
	# Find the height and width of the rotated image using cosine and sine transformations
	height_rot_img = round(abs(image.shape[0]*cosine)) + round(abs(image.shape[1]*sine))
	width_rot_img = round(abs(image.shape[1]*cosine)) + round(abs(image.shape[0]*sine))

	#Initialising the rotated image by zeros
	rot_img = np.uint8(np.zeros((height_rot_img,width_rot_img,image.shape[2])))
	
	# Finding the center point of the original image
	orgx, orgy = (image.shape[1]//2, image.shape[0]//2)

	# Finding the center point of rotated image.
	rotx,roty = (width_rot_img//2, height_rot_img//2)
	 
	for i in range(rot_img.shape[0]):
		for j in range(rot_img.shape[1]):
			# Find the all new coordinates for orginal image wrt the new center point
			x= (i-rotx)*cosine+(j-roty)*sine
			y= -(i-rotx)*sine+(j-roty)*cosine

			x=round(x)+orgy
			y=round(y)+orgx

			#Restricting the index in between original height and width of image.
			if (x>=0 and y>=0 and x<image.shape[0] and  y<image.shape[1]):
				rot_img[i,j,:] = image[x,y,:]
	return rot_img, degree

def random_cutout(image):
	#Choose Random degree
	random_pxl = random.randint(0, 16)
	# print("Random height and width chosen: ", random_pxl)

	#Choose random position in an image and do cutout and replace with single that random pixel value.
	img_height=image.shape[0]
	img_width=image.shape[1]
	img=np.uint8(np.zeros((img_height,img_width,image.shape[2])))

	cx = random.randint(0, img_width-random_pxl+1)
	cy = random.randint(0, img_height-random_pxl+1)

	for i in range(img_height):
		for j in range(img_width):
			if i>= cy and i<cy+random_pxl and j>=cx and j<cx+random_pxl:
				img[i, j, :] = random_pxl
			else: img[i, j, :] = image[i, j, :]
	return img, random_pxl

def random_crop(image):
	img_height=image.shape[0]
	img_width=image.shape[1]
	#Adding two pixels padding across the image
	padded_img=np.uint8(np.zeros((img_height+4,img_width+4,image.shape[2])))
	img=np.uint8(np.zeros((img_height,img_width,image.shape[2])))

	#Choose random position for cropping
	cx =random.randint(0, 4)
	cy =random.randint(0, 4)

	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			padded_img[i+2, j+2, :] = image[i, j, :]
	for i in range(cy, padded_img.shape[0]):
		for j in range(cx, padded_img.shape[1]):
			if i-cy<img.shape[0] and j-cx<img.shape[1]:
				img[i-cy, j-cx, :] = padded_img[i, j, :]
	return img, (cx, cy)

def contrast_and_flip(image):
	img_height=image.shape[0]
	img_width=image.shape[1]
	#Adding two pixels padding across the image
	img=np.uint8(np.zeros((img_height,img_width,image.shape[2])))

	alpha = random.uniform(0.5, 2.0)
	flip_prob = random.randint(0, 1)
	# print("Alpha value: ", alpha)

	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			pxl=[]
			for c in range(image.shape[2]):
				cont_pxl = int(alpha*(image[i, j, c] - 128) + 128)
				if cont_pxl>255: cont_pxl=255
				elif cont_pxl<0: cont_pxl=0
				pxl.append(cont_pxl)
			img[i, j, :] = np.array(pxl)
	if(flip_prob):
		# print("Including Horizontal Flipping")
		img = img[:, ::-1, :] #Horizontal Flipping
	return img, round(alpha, 3)

"""## Question 3"""

##Generating Augmented Images
def get_augmented_images(data, labels):
	augmented_img=[]
	augmented_labels=[]
	preprocess_func = {0: random_rotation, 1: random_cutout, 2: random_crop, 3: contrast_and_flip}
	i=0
	for img in data:
		rndm_idx = random.randint(0, 3)
		if(i%1000==0): 
			print("Image no.: ", i, end=' ')
			print("Random function: ", preprocess_func[rndm_idx])
		#This is because rotation changes the actual size of image so use it to convert back to 32x32
		if(preprocess_func[rndm_idx] == random_rotation):
			img1, _ = preprocess_func[rndm_idx](img)
			img2, _ = preprocess_func[rndm_idx](img)
			img1 = cv2.resize(img1, (32, 32))
			img2 = cv2.resize(img2, (32, 32))
		else:
			img1, _ = preprocess_func[rndm_idx](img)
			img2, _ = preprocess_func[rndm_idx](img)
		augmented_img.append(img1)
		augmented_img.append(img2)
		augmented_labels.append(labels[i])
		augmented_labels.append(labels[i])
		i+=1
	return np.array(augmented_img), augmented_labels

def get_feat_vec(images, obj):
	feat_vec=[]
	for img in images:
		img = cv2.resize(img, (224, 224))
		img = np.transpose(img, (2, 1, 0))
		#Performing Normalization before sending into ResNet model
		img = img/255
		img=np.array(img, dtype=np.float32)
		feat_vec.append(obj.feature_extraction(np.array([img]))[0])
	return np.array(feat_vec)
