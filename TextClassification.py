import tensorflow as tf
from tensorflow import keras
import numpy as np

"""
此练习将评论文本分为积极(positive)和消极(nagetive)，数据来源IMDB数据集，包含50000条影评文本
"""

# 下载IMDB数据集
imdb = keras.datasets.imdb
(train_x, train_y), (test_x, test_y) = imdb.load_data(num_words=10000)

print("Trainning entries: {}, labels: {}".format(len(train_x), len(train_y)))
print(train_x[0])

# 不同电影的句子长度不同，但神经元输入需要是一致的
print(len(train_x[0]), len(train_x[1]))

# 映射单词到整数索引的词典
word_index = imdb.get_word_index()

# 保留第一个索引
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reversed_dict = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
    return ' '.join(reversed_dict.get(i, '?') for i in text)


print(decode_review(train_x[0]))

# pad_sequences来标准化，使得句子长度一致
train_x = keras.preprocessing.sequence.pad_sequences(train_x,
                                                     value=word_index['<PAD>'],  # 填充的元素
                                                     padding='post',
                                                     maxlen=256)  # 填充的目标长度

test_x = keras.preprocessing.sequence.pad_sequences(test_x,
                                                    value=word_index['<PAD>'],
                                                    padding='post',
                                                    maxlen=256)
# 标准化后的长度均为: 256
print("标准化后的数据长度")
print(len(train_x[0]), len(train_x[1]))
# 填充后的数据
print("标准化后的数据")
print(train_x[0])

"""
构建模型:
-模型有多少层
-每个层中的神经元
"""
vocal_size = 10000  # 总词汇数量

model = keras.Sequential()
model.add(keras.layers.Embedding(vocal_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

print("模型架构:")
print(model.summary())

"""
损失函数:
使用binary_crossentropy来定义——分类
使用mean_squared_error来定义——回归
"""

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
val_x = train_x[:10000]
partial_train_x = train_x[10000:]

val_y = train_y[:10000]
partial_train_y = train_y[10000:]

from tensorflow.keras.callbacks import TensorBoard

# 展示回调函数
TBcallback = TensorBoard(log_dir='./TensorBoard/TextClassification', histogram_freq=1)
# EarlyStop回调函数,在第26步的时候自动停下
EScallback = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0,
                                           mode='auto', baseline=None, restore_best_weights=False)

history = model.fit(partial_train_x,
                    partial_train_y,
                    epochs=40,
                    batch_size=512,
                    validation_data=(val_x, val_y),
                    verbose=2,
                    callbacks=[TBcallback, EScallback])

result = model.evaluate(test_x, test_y, verbose=2)

print(result)
