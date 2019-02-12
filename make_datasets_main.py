# -*- coding: UTF-8 -*-
import os,argparse
from make_datasets_fun import *

work_path = " "
datasets_dir = " "

height = 900
width = 100
channel = 3
num_img = 5000 #合格/不合格图片各5000张
per_batch_count =1000   #每个pickled文件1000张图片，合格/不合格各500张

#####################################################################################################

'''resize images'''
good_path = os.path.join(datasets_dir,"1")
bad_path = os.path.join(datasets_dir,"0")
resized_dir = os.path.join(work_path,"resized")
good_resized_dir = os.path.join(resized_dir,"1")
bad_resized_dir = os.path.join(resized_dir,"0")
os.makedirs(good_resized_dir)
os.makedirs(bad_resized_dir)

resize_img_dir(good_path, good_resized_dir,num_img,height,width)
resize_img_dir(bad_path, bad_resized_dir,num_img,height,width)
print("Finish in resizing!")

'''pickled images'''
pickled_dir = os.path.join(work_path,"pickled")
os.makedir(pickled_dir)
good_image_list = get_image_name(good_resized_dir)
bad_image_list = get_image_name(bad_resized_dir)

batch_num = min(len(good_image_list),len(bad_image_list))//per_batch_count

for i in range(0,batch_num):
	interval = i * per_batch_count
	start = int(0+interval)
	end = int(per_batch_count+interval)
	data, labels, names = read_data(resized_dir,good_image_list[start:end],bad_image_list[start:end],height,width)
	gdump(pickled_dir,i,data, labels, names)
	print("batch {0} has been pickled!".format(str(i)))
