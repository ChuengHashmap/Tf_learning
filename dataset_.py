"""
tf.data模块-包括了一些列灵活的数据集构建API，可帮助快速构建数据输入的流水线
核心类-tf.data.DataSet：由可迭代的访问元素Element组成，
每个元素包含一个/多个张量(图片数据-(长,宽,高)三个张量)
"""

# 建立DataSet的方法: tf.data.DataSet.from_tensor_slices()
import numpy as np
import tensorflow as tf

X = tf.constant([1, 2, 3, 4, 5, 6])
Y = tf.constant([10, 20, 30, 40, 50, 60])

dataset = tf.data.Dataset.from_tensor_slices((X, Y))
for d in dataset:
    print(d)
print("-------------------------------")
it = iter(dataset)
next(it)

# 载入MNIST数据
import matplotlib.pyplot as plt

(train_data, train_label), (_, _) = tf.keras.datasets.mnist.load_data()
mnist_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_label))
# for image, label in mnist_dataset:
#     plt.title(label.numpy())
#     plt.imshow(image.numpy()[:, :])
#     plt.show()

"""
tf.data.Dataset 类为我们提供了多种数据集预处理方法。最常用的如：
Dataset.map(f) ：对数据集中的每个元素应用函数 f ，得到一个新的数据集（这部分往往结合 tf.io 进行读写和解码文件， tf.image 进行图像处理）；
Dataset.shuffle(buffer_size) ：将数据集打乱（设定一个固定大小的缓冲区（Buffer），取出前 buffer_size 个元素放入，并从缓冲区中随机采样，采样后的数据用后续数据替换）；
Dataset.batch(batch_size) ：将数据集分成批次；
Dataset.repeat():重复数据集的元素,epoch
"""
# dataset中的元素+1
dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4])
dataset = dataset.map(lambda x: x + 1)
for d in dataset:
    print(d)

# .batch()

# shuffle(): 打散，维持一个固定大小的buffer,
# 并从该buffer中随机均匀地选择下一个元素,参数buffer_size建议设为样本数量，
# 过大会浪费内存空间，过小会导致打乱不充分。
dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])
dataset = dataset.shuffle(3)
for d in dataset:
    print("------")
    print(d)

# repeat(): 重复序列多次，主要用于epoch,repeat(5)-5个epoch
dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])
dataset = dataset.repeat(2)
dataset = dataset.shuffle(4)
for d in dataset:
    print(d)

