import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

tb_CallBack = TensorBoard(log_dir='./TensorBoard', histogram_freq=1)

predictions = model(x_train[:1]).numpy()
print("predicts:", predictions)
# 可以在最后一层加入softmax层，但不建议，因为softmax输出时可能不会提供较为精确的损失计算
print("softmax_predicts:", tf.nn.softmax(predictions).numpy())

# 返回标量损失
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss = loss_fn(y_train[:1], predictions).numpy()
# 十分类，
print("loss(~2.3): ", loss)

# 开始之前，使用keras的model.compile配置和编译模型，其中metrics为模型评估的指标
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# model.fit方法来调整模型参数、最小化损失
model.fit(x_train, y_train, epochs=5, batch_size=30, shuffle=True, validation_split=0.2, callbacks=[tb_CallBack])

# verbose=0时:不输出日志信息；verbose=1:输出带进度条j的信息; verbose=3:为每个epoch输出记录(没有进度条)
model.evaluate(x_test, y_test, verbose=2)

# 可以结合softmax函数，封装为新的模型
probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])

probability_model(x_test[:5])
