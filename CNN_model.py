# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")
from img_function import *


#基本参数设置
height = 224
width = 224
channel = 3
n_class = 2

## train集参数设置
train_batch_size = 40
train_ratio = 0.8
keep_prob = 0.6
epochs = 4

## 加载数据
data_path = " "
train_dataset = os.path.join(data_path,"train")
test_dataset = os.path.join(data_path,"test")

save_model_path = " "
results1 = open("train_output.txt","w")
## 训练模型
inputs_, targets_,conv1, conv2,fc1,fc2,logits_,cost, optimizer,correct_pred,accuracy = create_network(n_class,keep_prob,height,width,channel)
count = 0
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for epoch in range(epochs):
		for batch in get_batch_file(train_dataset):
			x_train_, x_val, y_train_, y_val = get_train_data(batch,n_class,train_ratio, height, width, channel)
			img_shape = x_train_.shape[0] + x_val.shape[0]
			num_sub_batch = img_shape//train_batch_size - 1
			print(num_sub_batch)
			for i in range(0,num_sub_batch):
				feature_batch = x_train_[i * train_batch_size: (i + 1) * train_batch_size]
				label_batch = y_train_[i * train_batch_size: (i + 1) * train_batch_size]
				#label_batch = label_batch.reshape(int(label_batch.shape[0])//2,2)
				train_loss, _ = sess.run([cost, optimizer], feed_dict={inputs_: feature_batch,targets_: label_batch})
				val_acc = sess.run(accuracy,feed_dict={inputs_: x_val,targets_: y_val})
				if (count % 10 == 0):
					print('Epoch {:>2}, Train Loss {:.4f}, Validation Accuracy {:4f} '.format(epoch + 1, train_loss, val_acc))
					results1.write("Epoch {:>2}, Train Loss {:.4f}, Validation Accuracy {:4f} \n ".format(epoch + 1, train_loss, val_acc))
				count += 1
	# 存储参数
	saver = tf.train.Saver()
	save_path = saver.save(sess, save_model_path)
results1.close()

#测试结果
import random
results2 = open("test_output.txt","w")
loaded_graph = tf.Graph()
test_batch_size = 20


with tf.Session(graph=loaded_graph) as sess:
	# 加载模型
	loader = tf.train.import_meta_graph(save_model_path + '.meta')
	loader.restore(sess, save_model_path)
	# 加载tensor
	loaded_x = loaded_graph.get_tensor_by_name('inputs_:0')
	loaded_y = loaded_graph.get_tensor_by_name('targets_:0')
	loaded_logits = loaded_graph.get_tensor_by_name('logits_:0')
	loaded_acc = loaded_graph.get_tensor_by_name('accuracy:0')
	# 计算test的准确率
	test_batch_acc_total = 0
	test_batch_count = 0
	results2.write("Begin test...\n")
	for batch in get_batch_file(test_dataset):
		x_test, y_test = get_test_data(batch, n_class,height, width, channel)
		num_sub_batch = x_test.shape[0] // test_batch_size - 1
		for j in range(0,num_sub_batch):
			test_feature_batch = x_test[j * test_batch_size: (j + 1) * test_batch_size]
			test_label_batch = y_test[j * test_batch_size: (j + 1) * test_batch_size]
			test_batch_acc_total += sess.run(loaded_acc,feed_dict={loaded_x: test_feature_batch, loaded_y: test_label_batch})
			test_batch_count += 1
		results2.write("Test Accuracy: {}\n".format(test_batch_acc_total / test_batch_count))
results2.close()



