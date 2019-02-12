# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import cv2,gzip
import numpy as np
import pickle as p

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def resize_img_dir(pictures_dir,output_dir,n_img,height,width):
	'''对路径下所有png图片进行resize'''
	num = 0
	for (root, dirs, files) in os.walk(pictures_dir):
		for file in files:
			image = os.path.join(root,file)
			image_name = os.path.basename(image)
			image_raw_data = tf.gfile.GFile(image, 'rb').read()
			with tf.Session() as sess:
				img_data = tf.image.decode_png(image_raw_data)
				resized = tf.image.resize_images(img_data, [height,width], method=0)
				retype = tf.cast(resized, tf.uint8)
				encoder_image = tf.image.encode_png(retype)
				image_name_new = os.path.join(output_dir, image_name)
				with tf.gfile.GFile(image_name_new, 'wb') as f:
					f.write(encoder_image.eval())
			num +=1
			if num == n_img:
				break

#####################################################################################
"""read image data"""
def imread(im_path,color="RGB", mode=cv2.IMREAD_UNCHANGED):
	im = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
	if color == "RGB":
		im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
	return im

def count_num_im(dir):
	num = 0
	all_files = os.listdir(dir)
	for each_file in all_files:
		if os.path.isdir(each_file):
			continue
		else:
			if each_file.endswith(".png"):
			 num +=1
	return num

def get_image_name(dir):
	name_list = []
	for (root, dirs, files) in os.walk(dir):
		for file in files:
			name_list.append(file)
	return name_list

def merge_f(x,y):
	'''交叉合并两个列表'''
	lst = []
	for i in list(zip(x, y)):
		lst.append(list(i))
	m = []
	for i in lst:
		for j in i:
			m.append(j)
	return m

def read_data(data_path,good_image_list,bad_image_list,height,width,channel=3,color='RGB'):
	'''读取图片数据，将features和1,0的labels分别存入数组，再存为data字典，注意0，1间隔存入，方便后面取数据'''
	DATA_LEN = height*width*channel
	CHANNEL_LEN = height*width
	image_1_path = os.path.join(data_path,"1")
	image_0_path = os.path.join(data_path,"0")
	num_1 = len(good_image_list)
	num_0 = len(bad_image_list)
	total_num = num_1 + num_0

	data = np.zeros((total_num, DATA_LEN), dtype=np.uint8)
	names_list = merge_f(good_image_list,bad_image_list)
	labels_list = merge_f([1]*num_1,[0]*num_0)

	idx = 0
	c = CHANNEL_LEN
	for image in names_list:
		if os.path.exists(os.path.join(image_1_path,image)):
			path=image_1_path
			lable = 1
		else:
			path=image_0_path
			lable = 0
		im = imread(os.path.join(path,image),color='RGB')
		data[idx,:c] =  np.reshape(im[:,:,0], c)
		data[idx, c:2*c] = np.reshape(im[:,:,1], c)
		data[idx, 2*c:] = np.reshape(im[:,:,2], c)
		idx = idx + 1
	return data, labels_list, names_list

def gdump(savepath, bach_id,data, label, fnames, mode="train"):
	"""pickle the data"""
	dict = {'data': data,'labels': label,'filenames': fnames}
	num_1 = str(label.count(1))
	num_0 = str(label.count(0))
	if mode == "train":
		dict['batch_label'] = "training batch_{0} with {1} 1 class and {2} 0 class".format(bach_id, num_1,num_0)
	else:
		dict['batch_label'] = "testing batch_{0} with {1} 1 class and {2} 0 class".format(bach_id, num_1,num_0)
	filename = os.path.join(savepath, 'data_batch_'+str(bach_id))
	file = gzip.GzipFile(filename, 'wb')
	p.dump(dict, file,protocol=4)
	file.close