import os,gzip
import pickle
import numpy as np
import tensorflow as tf

'''数据读取'''
def get_batch_file(dir):
	file_list = []
	for (root, dirs, files) in os.walk(dir):
		for file in files:
			file_path = os.path.join(root,file)
			file_list.append(file_path)
	return file_list

def load_data_batch(pickled_file,height,width,channel):
	file = gzip.GzipFile(pickled_file, 'rb')
	batch = pickle.load(file)
	# features and labels
	features = batch['data'].reshape((len(batch['data']),channel, height,width)).transpose(0, 2, 3, 1)
	labels = batch['labels']
	file.close()
	return features, labels

'''数据预处理'''
def normalization_features(x,height,width,channel):
	'''对输入特征归一化'''
	from sklearn.preprocessing import MinMaxScaler
	minmax = MinMaxScaler()
	# 重塑
	x_rows = x.reshape(x.shape[0], height * width * channel)
	# 归一化
	x = minmax.fit_transform(x_rows)
	# 重新变为height * width * channel
	x = x.reshape(x.shape[0], height, width, channel)
	return x

def one_hot_code(n_class,y):
	'''对目标变量进行one-hot编码'''
	from sklearn.preprocessing import LabelBinarizer
	lb = LabelBinarizer().fit(np.array(range(n_class)))
	y = lb.transform(y)
	return y

def one_hot_code2(y,n_class=2):
	#创建一个适当大小的矩阵来接收
	length = np.array(y).shape[0]
	array=np.arange(length*n_class).reshape(length,n_class)
	for i in range(0,length):
		if y[i] == 0:
			array[i] = [0,1] #这里采用one-hot编码，即[0,1]值来表示图片不合格，[1,0]值来表示图片合格
		else:
			array[i] = [1, 0]
	return array


def split_train_data(train_ratio,x_train,y_train):
	'''划分train与val数据集'''
	from sklearn.model_selection import train_test_split
	x_train_, x_val, y_train_, y_val = train_test_split(x_train,y_train,train_size=train_ratio,random_state=123)
	return x_train_, x_val, y_train_, y_val


def get_train_data(batch,n_class,train_ratio, height, width, channel):
	features, labels = load_data_batch(batch, height, width, channel)
	x_train = normalization_features(features, height, width, channel)
	y_train = one_hot_code2(labels,n_class)
	x_train_, x_val, y_train_, y_val = split_train_data(train_ratio, x_train, y_train)
	return x_train_, x_val, y_train_, y_val

def get_test_data(batch, n_class,height, width, channel):
	features, labels = load_data_batch(batch, height, width, channel)
	x_test = normalization_features(features, height, width, channel)
	y_test = one_hot_code2(labels,n_class)
	return x_test,y_test

'''构建CNN网络'''
def create_network(n_class,keep_prob,height,width,channel):
	'''输入与标签'''
	inputs_ = tf.placeholder(tf.float32, [None, height,width,channel], name='inputs_')
	targets_ = tf.placeholder(tf.float32, [None, n_class], name='targets_')
	'''卷积与池化'''
	conv1 = tf.layers.conv2d(inputs_, 64, (2, 2), padding='same', activation=tf.nn.relu,
							 kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
	conv1 = tf.layers.max_pooling2d(conv1, (2, 2), (2, 2), padding='same')
	conv2 = tf.layers.conv2d(conv1, 128, (4, 4), padding='same', activation=tf.nn.relu,
							 kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
	conv2 = tf.layers.max_pooling2d(conv2, (2, 2), (2, 2), padding='same')
	# 重塑输出
	shape = np.prod(conv2.get_shape().as_list()[1:])
	conv2 = tf.reshape(conv2, [-1, shape])
	# 第一层全连接层
	fc1 = tf.contrib.layers.fully_connected(conv2, 1024, activation_fn=tf.nn.relu)
	fc1 = tf.nn.dropout(fc1, keep_prob)
	# 第二层全连接层
	fc2 = tf.contrib.layers.fully_connected(fc1, 512, activation_fn=tf.nn.relu)
	# logits层
	logits_ = tf.contrib.layers.fully_connected(fc2, 2, activation_fn=None)
	logits_ = tf.identity(logits_, name='logits_')
	# cost & optimizer
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_, labels=targets_))
	optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
	# accuracy
	correct_pred = tf.equal(tf.argmax(logits_, 1), tf.argmax(targets_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
	return inputs_, targets_,conv1, conv2,fc1,fc2,logits_,cost, optimizer,correct_pred,accuracy
