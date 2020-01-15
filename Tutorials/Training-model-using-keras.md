# 使用Keras训练模型
<!-- TOC -->

- [使用Keras训练模型](#使用keras训练模型)
    - [1. 一般模型的构造,训练,测试流程](#1-一般模型的构造训练测试流程)
    - [2. 自定义损失和指标](#2-自定义损失和指标)
        - [2.1 自定义metric-1](#21-自定义metric-1)
        - [2.2 自定义metric-2](#22-自定义metric-2)
        - [2.3 自定义metric-3](#23-自定义metric-3)
        - [2.4 自定义loss](#24-自定义loss)
    - [3. 使用tf.data构造数据](#3-使用tfdata构造数据)
    - [4. 样本权重和类权重](#4-样本权重和类权重)
    - [5. 多输入多输出模型](#5-多输入多输出模型)
    - [6. 使用回调](#6-使用回调)
        - [6.1 使用回调](#61-使用回调)
        - [6.2 创建自己的回调方法](#62-创建自己的回调方法)
    - [7. 自己构造训练和验证循环](#7-自己构造训练和验证循环)
        - [7.1 构建网络并训练:](#71-构建网络并训练)
        - [7.2 构建网络,训练并验证](#72-构建网络训练并验证)
        - [7.3 构造自定义loss,并训练](#73-构造自定义loss并训练)

<!-- /TOC -->
## 1. 一般模型的构造,训练,测试流程
```python
# 载入数据
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') /255
x_test = x_test.reshape(10000, 784).astype('float32') /255

x_val = x_train[-10000:]
y_val = y_train[-10000:]

x_train = x_train[:-10000]
y_train = y_train[:-10000]

# 模型构造
inputs = keras.Input(shape=(784,), name='mnist_input')
h1 = keras.layers.Dense(64, activation='relu')(inputs)
h2 = keras.layers.Dense(64, activation='relu')(h1)
outputs = keras.layers.Dense(10, activation='softmax')(h2)
model = keras.Model(inputs, outputs)
# keras.utils.plot_model(model, 'net001.png', show_shapes=True)

model.compile(optimizer=keras.optimizers.RMSprop(),
             loss=keras.losses.SparseCategoricalCrossentropy(),
             metrics=[keras.metrics.SparseCategoricalAccuracy()])

# 训练模型
history = model.fit(x_train, y_train, batch_size=64, epochs=3,
         validation_data=(x_val, y_val))
print('history:')
print(history.history)

result = model.evaluate(x_test, y_test, batch_size=128, verbose=0)
print('evaluate:')
print(result)
pred = model.predict(x_test[:2])
print('predict:')
print(pred)
```
## 2. 自定义损失和指标
自定义指标只需继承Metric类， 并重写一下函数:  
- `_init_(self)`，初始化。
- `update_state(self，y_true，y_pred，sample_weight = None)`，它使用目标y_true和模型预测y_pred来更新状态变量。
- `result(self)`，它使用状态变量来计算最终结果。
- `reset_states(self)`，重新初始化度量的状态。  
一下列出了4中与自定义Metric相关的方法和一种自定义loss的方法：
### 2.1 自定义metric-1
继承自[`keras.metrics.Metric`](https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Metric)：
```python
# 模型构造
inputs = keras.Input(shape=(784,), name='mnist_input')
h1 = keras.layers.Dense(64, activation='relu')(inputs)
h2 = keras.layers.Dense(64, activation='relu')(h1)
outputs = keras.layers.Dense(10, activation='softmax')(h2)
model = keras.Model(inputs, outputs)
# keras.utils.plot_model(model, 'net001.png', show_shapes=True)

model.compile(optimizer=keras.optimizers.RMSprop(),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=[keras.metrics.SparseCategoricalAccuracy()])

# 载入数据
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') /255
x_test = x_test.reshape(10000, 784).astype('float32') /255

x_val = x_train[-10000:]
y_val = y_train[-10000:]

x_train = x_train[:-10000]
y_train = y_train[:-10000]

# 这是一个简单的示例，显示如何实现CatgoricalTruePositives指标，该指标计算正确分类为属于给定类的样本数量
class CatgoricalTruePostives(keras.metrics.Metric):
    def __init__(self, name='binary_true_postives', **kwargs):
        super(CatgoricalTruePostives, self).__init__(name=name, **kwargs)
        self.true_postives = self.add_weight(name='tp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred)
        y_true = tf.equal(tf.cast(y_pred, tf.int32), tf.cast(y_true, tf.int32))

        y_true = tf.cast(y_true, tf.float32)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            y_true = tf.multiply(sample_weight, y_true)

        return self.true_postives.assign_add(tf.reduce_sum(y_true))

    def result(self):
        return tf.identity(self.true_postives)

    def reset_states(self):
        self.true_postives.assign(0.)

# 在compile中的使用方法
model.compile(optimizer=keras.optimizers.RMSprop(1e-3),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=[CatgoricalTruePostives()])

model.fit(x_train, y_train,
        batch_size=64, epochs=3)
```
  
### 2.2 自定义metric-2
继承自`keras.layers.Layer`,并以层的方式使用:
```python
# 载入数据
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') /255
x_test = x_test.reshape(10000, 784).astype('float32') /255

x_val = x_train[-10000:]
y_val = y_train[-10000:]

x_train = x_train[:-10000]
y_train = y_train[:-10000]

class MetricLoggingLayer(keras.layers.Layer):
    def call(self, inputs):
        self.add_metric(keras.backend.std(inputs),
                       name='std_of_activation',
                       aggregation='mean')

        return inputs

inputs = keras.Input(shape=(784,), name='mnist_input')
h1 = keras.layers.Dense(64, activation='relu')(inputs)
h1 = MetricLoggingLayer()(h1)  # 用的地方比较灵活
h1 = keras.layers.Dense(64, activation='relu')(h1)
outputs = keras.layers.Dense(10, activation='softmax')(h1)
model = keras.Model(inputs, outputs)
# keras.utils.plot_model(model, 'net001.png', show_shapes=True)

model.compile(optimizer=keras.optimizers.RMSprop(),
             loss=keras.losses.SparseCategoricalCrossentropy(),
             metrics=[keras.metrics.SparseCategoricalAccuracy()])
model.fit(x_train, y_train, batch_size=32, epochs=1)
```
  
### 2.3 自定义metric-3
继承自`keras.layers.Layer`,直接作用与model:
```python
# 载入数据
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') /255
x_test = x_test.reshape(10000, 784).astype('float32') /255

x_val = x_train[-10000:]
y_val = y_train[-10000:]

x_train = x_train[:-10000]
y_train = y_train[:-10000]

# 也可以直接在model上面加
# 也可以以定义网络层的方式添加要统计的metric
class MetricLoggingLayer(keras.layers.Layer):
    def call(self, inputs):
        self.add_metric(keras.backend.std(inputs),
                       name='std_of_activation',
                       aggregation='mean')

        return inputs

inputs = keras.Input(shape=(784,), name='mnist_input')
h1 = keras.layers.Dense(64, activation='relu')(inputs)
h2 = keras.layers.Dense(64, activation='relu')(h1)
outputs = keras.layers.Dense(10, activation='softmax')(h2)
model = keras.Model(inputs, outputs)

# 直接添加到模型上
model.add_metric(keras.backend.std(inputs),
                       name='std_of_activation',
                       aggregation='mean')
model.add_loss(tf.reduce_sum(h1)*0.1)

# keras.utils.plot_model(model, 'net001.png', show_shapes=True)

model.compile(optimizer=keras.optimizers.RMSprop(),
             loss=keras.losses.SparseCategoricalCrossentropy(),
             metrics=[keras.metrics.SparseCategoricalAccuracy()])
model.fit(x_train, y_train, batch_size=32, epochs=1)
```
  
### 2.4 自定义loss
以自定义层的方式,继承自keras.layers.Layer, 在`call`方法中重写[`add_loss`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer#add_loss):
```python
# 载入数据
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') /255
x_test = x_test.reshape(10000, 784).astype('float32') /255

x_val = x_train[-10000:]
y_val = y_train[-10000:]

x_train = x_train[:-10000]
y_train = y_train[:-10000]

# 以定义网络层的方式添加网络loss
class ActivityRegularizationLayer(keras.layers.Layer):
    def call(self, inputs):
        self.add_loss(tf.reduce_sum(inputs) * 0.1)
        return inputs

inputs = keras.Input(shape=(784,), name='mnist_input')
h1 = keras.layers.Dense(64, activation='relu')(inputs)
h1 = ActivityRegularizationLayer()(h1) # loss层
h1 = keras.layers.Dense(64, activation='relu')(h1)
outputs = keras.layers.Dense(10, activation='softmax')(h1)
model = keras.Model(inputs, outputs)
# keras.utils.plot_model(model, 'net001.png', show_shapes=True)

model.compile(optimizer=keras.optimizers.RMSprop(),
             loss=keras.losses.SparseCategoricalCrossentropy(),
             metrics=[keras.metrics.SparseCategoricalAccuracy()])
model.fit(x_train, y_train, batch_size=32, epochs=1)
```
## 3. 使用tf.data构造数据
```python
def get_compiled_model():
    # 定义模型结构
    inputs = keras.Input(shape=(784,), name='mnist_input')
    h1 = keras.layers.Dense(64, activation='relu')(inputs)
    h2 = keras.layers.Dense(64, activation='relu')(h1)
    outputs = keras.layers.Dense(10, activation='softmax')(h2)
    model = keras.Model(inputs, outputs) # 生成模型实例
    # 编译模型,此后模型可用于训练了
    model.compile(optimizer=keras.optimizers.RMSprop(),
                 loss=keras.losses.SparseCategoricalCrossentropy(),
                 metrics=[keras.metrics.SparseCategoricalAccuracy()])
    return model

model = get_compiled_model()
# 利用`(x_train, y_train)`生成数据集,`from_tensor_slices`的用法见https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_tensor_slices
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)) 
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(64)

# model.fit(train_dataset, epochs=3)
# steps_per_epoch 每个epoch只训练几步
# validation_steps 每次验证，验证几步
model.fit(train_dataset, epochs=3, steps_per_epoch=100,
         validation_data=val_dataset, validation_steps=3)
```
## 4. 样本权重和类权重
“样本权重”数组是一个数字数组，用于指定批处理中每个样本在计算总损失时应具有多少权重。 它通常用于不平衡的分类问题（这个想法是为了给予很少见的类更多的权重）。 当使用的权重是1和0时，该数组可以用作损失函数的掩码（完全丢弃某些样本对总损失的贡献）。    
“类权重”dict是同一概念的更具体的实例：它将类索引映射到应该用于属于该类的样本的样本权重。 例如，如果类“0”比数据中的类“1”少两倍，则可以使用class_weight = {0：1.，1：0.5}。  
```python
# 载入数据
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') /255
x_test = x_test.reshape(10000, 784).astype('float32') /255

x_val = x_train[-10000:]
y_val = y_train[-10000:]

x_train = x_train[:-10000]
y_train = y_train[:-10000]


def get_compiled_model():
    # 定义模型结构
    inputs = keras.Input(shape=(784,), name='mnist_input')
    h1 = keras.layers.Dense(64, activation='relu')(inputs)
    h2 = keras.layers.Dense(64, activation='relu')(h1)
    outputs = keras.layers.Dense(10, activation='softmax')(h2)
    model = keras.Model(inputs, outputs) # 生成模型实例
    # 编译模型,此后模型可用于训练了
    model.compile(optimizer=keras.optimizers.RMSprop(),
                 loss=keras.losses.SparseCategoricalCrossentropy(),
                 metrics=[keras.metrics.SparseCategoricalAccuracy()])
    return model

# 增加第5类的权重
# 类权重
model = get_compiled_model()
class_weight = {i:1.0 for i in range(10)}
class_weight[5] = 2.0
print(class_weight)
model.fit(x_train, y_train,
         class_weight=class_weight, # 此处指定
         batch_size=64,
         epochs=4)


# 样本权重
model = get_compiled_model()
sample_weight = np.ones(shape=(len(y_train),))
sample_weight[y_train == 5] = 2.0
model.fit(x_train, y_train,
         sample_weight=sample_weight, # 此处指定
         batch_size=64,
         epochs=4)


# tf.data数据
model = get_compiled_model()

sample_weight = np.ones(shape=(len(y_train),))
sample_weight[y_train == 5] = 2.0

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train,
                                                    sample_weight))  # 构造数据训练数据时指定
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(64)

model.fit(train_dataset, epochs=3, )
```
## 5. 多输入多输出模型
```python
image_input = keras.Input(shape=(32, 32, 3), name='img_input') # 图像输入
timeseries_input = keras.Input(shape=(None, 10), name='ts_input') # 时间输入

x1 = keras.layers.Conv2D(3, 3)(image_input)
x1 = keras.layers.GlobalMaxPooling2D()(x1)

x2 = keras.layers.Conv1D(3, 3)(timeseries_input)
x2 = keras.layers.GlobalMaxPooling1D()(x2)

x = keras.layers.concatenate([x1, x2])  # 两个特征组合

score_output = keras.layers.Dense(1, name='score_output')(x) # 多输入输出时，输入和输出tensor最好有name参数
class_output = keras.layers.Dense(5, activation='softmax', name='class_output')(x) 

model = keras.Model(inputs=[image_input, timeseries_input],  # 多输入，列表
                    outputs=[score_output, class_output])    # 多输出，列表
# keras.utils.plot_model(model, 'multi_input_output_model.png', show_shapes=True)

# 模型定义完毕，准备训练

# case 1
# 可以为模型指定不同的loss和metrics
model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss=[keras.losses.MeanSquaredError(),  #  图片loss
          keras.losses.CategoricalCrossentropy()]) # 时间loss


# case 2
# 还可以指定loss的权重
model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss={'score_output': keras.losses.MeanSquaredError(), # 因为上面使用了name参数，这里可以使用字典
          'class_output': keras.losses.CategoricalCrossentropy()},
    metrics={'score_output': [keras.metrics.MeanAbsolutePercentageError(),
                              keras.metrics.MeanAbsoluteError()],
             'class_output': [keras.metrics.CategoricalAccuracy()]},
    loss_weight={'score_output': 2., 'class_output': 1.})

# case 3
# 可以把不需要传播的loss置0
model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss=[None, keras.losses.CategoricalCrossentropy()])

# case 4
# dict loss version
model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss={'class_output': keras.losses.CategoricalCrossentropy()})
```
几个APIs:
-  [`keras.layers.concatenate`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/concatenate): Functional interface to the Concatenate layer.

## 6. 使用回调
- Keras中的回调是在训练期间（在epoch开始时，batch结束时，epoch结束时等）在不同点调用的对象，可用于实现以下行为：  
    - 在训练期间的不同时间点进行验证（超出内置的每个时期验证）
    - 定期检查模型或超过某个精度阈值
    - 在训练似乎平稳时改变模型的学习率
    - 在训练似乎平稳时对顶层进行微调
    - 在培训结束或超出某个性能阈值时发送电子邮件或即时消息通知等等。
- 可使用的内置回调有:
    - [`ModelCheckpoint`](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint)：定期保存模型。
    - [`EarlyStopping`](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping)：当训练不再改进验证指标时停止培训。
    - [`TensorBoard`](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/TensorBoard)：定期编写可在TensorBoard中显示的模型日志（更多细节见“可视化”）。
    - [`CSVLogger`](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/CSVLogger)：将丢失和指标数据流式传输到CSV文件。
    - [等等](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks)
### 6.1 使用回调
```python
# 载入数据
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') /255
x_test = x_test.reshape(10000, 784).astype('float32') /255

x_val = x_train[-10000:]
y_val = y_train[-10000:]

x_train = x_train[:-10000]
y_train = y_train[:-10000]

def get_compiled_model():
    # 定义模型结构
    inputs = keras.Input(shape=(784,), name='mnist_input')
    h1 = keras.layers.Dense(64, activation='relu')(inputs)
    h2 = keras.layers.Dense(64, activation='relu')(h1)
    outputs = keras.layers.Dense(10, activation='softmax')(h2)
    model = keras.Model(inputs, outputs) # 生成模型实例
    # 编译模型,此后模型可用于训练了
    model.compile(optimizer=keras.optimizers.RMSprop(),
                 loss=keras.losses.SparseCategoricalCrossentropy(),
                 metrics=[keras.metrics.SparseCategoricalAccuracy()])
    return model

model = get_compiled_model()

# EarlyStopping
early_stop_callbacks = [  # 写在列表中!!!
    keras.callbacks.EarlyStopping(
        # 是否有提升关注的指标
        monitor='val_loss',
        # 不再提升的阈值
        min_delta=1e-2,
        # 2个epoch没有提升就停止
        patience=2,
        verbose=1)
    ]

model.fit(x_train, y_train,
          epochs=2,
          batch_size=64,
          callbacks=early_stop_callbacks, # 此处使用
          validation_split=0.2)

# checkpoint模型回调
model = get_compiled_model()
check_callback = keras.callbacks.ModelCheckpoint(
    filepath='mymodel_{epoch}.h5',
    save_best_only=True,
    monitor='val_loss',
    verbose=1
)

model.fit(x_train, y_train,
         epochs=2,
         batch_size=64,
         callbacks=[check_callback], # 此处使用
         validation_split=0.2)


# 动态调整学习率
initial_learning_rate = 0.1
lr_schedule = keras.optimizers.schedules.ExponentialDecay(  # 定义lr的schedule
    initial_learning_rate,
    decay_steps=10000,
    decay_rate=0.96,
    staircase=True
    )
optimizer = keras.optimizers.RMSprop(learning_rate=lr_schedule) # 将自定义的schedule用于optimizer


# # 使用tensorboard
# tensorboard_cbk = keras.callbacks.TensorBoard(log_dir='./full_path_to_your_logs')
# model.fit(x_train, y_train,
#          epochs=5,
#          batch_size=64,
#          callbacks=[tensorboard_cbk],
#          validation_split=0.2)
```
### 6.2 创建自己的回调方法
需要继承自keras.callbacks.Callback, 并重写相关方法
```python
# 载入数据
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') /255
x_test = x_test.reshape(10000, 784).astype('float32') /255

x_val = x_train[-10000:]
y_val = y_train[-10000:]

x_train = x_train[:-10000]
y_train = y_train[:-10000]

def get_compiled_model():
    # 定义模型结构
    inputs = keras.Input(shape=(784,), name='mnist_input')
    h1 = keras.layers.Dense(64, activation='relu')(inputs)
    h2 = keras.layers.Dense(64, activation='relu')(h1)
    outputs = keras.layers.Dense(10, activation='softmax')(h2)
    model = keras.Model(inputs, outputs) # 生成模型实例
    # 编译模型,此后模型可用于训练了
    model.compile(optimizer=keras.optimizers.RMSprop(),
                 loss=keras.losses.SparseCategoricalCrossentropy(),
                 metrics=[keras.metrics.SparseCategoricalAccuracy()])
    return model

class LossHistory(keras.callbacks.Callback): # 继承自keras.callbacks.Callback
    def on_train_begin(self, logs): # 重写函数
        self.losses = []
    def on_epoch_end(self, batch, logs): # 重写函数
        self.losses.append(logs.get('loss'))
        print('\nloss:',self.losses[-1])

model = get_compiled_model()

callbacks = [
    LossHistory()
]
model.fit(x_train, y_train,
          epochs=3,
          batch_size=64,
          callbacks=callbacks,
          validation_split=0.2)
```
## 7. 自己构造训练和验证循环
不使用`fit`方法!!!  
### 7.1 构建网络并训练:  
```python
# 载入数据
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') /255
x_test = x_test.reshape(10000, 784).astype('float32') /255

x_val = x_train[-10000:]
y_val = y_train[-10000:]

x_train = x_train[:-10000]
y_train = y_train[:-10000]

# 构建一个全连接网络.
inputs = keras.Input(shape=(784,), name='digits')
x = keras.layers.Dense(64, activation='relu', name='dense_1')(inputs)
x = keras.layers.Dense(64, activation='relu', name='dense_2')(x)
outputs = keras.layers.Dense(10, activation='softmax', name='predictions')(x)  # outputs.shape: TensorShape([None, 10])
model = keras.Model(inputs=inputs, outputs=outputs)

# 优化器.
optimizer = keras.optimizers.SGD(learning_rate=1e-3)
# 损失函数.
loss_fn = keras.losses.SparseCategoricalCrossentropy()

# 准备数据.
batch_size = 64
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

# 训练循环
for epoch in range(3):
    print('epoch: ', epoch)
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        # 开一个gradient tape, 计算梯度
        with tf.GradientTape() as tape:
            logits = model(x_batch_train)

            loss_value = loss_fn(y_batch_train, logits)  # 1. 计算loss
            grads = tape.gradient(loss_value, model.trainable_variables) # 计算梯度
            optimizer.apply_gradients(zip(grads, model.trainable_variables)) # 反向传播

        if step % 200 == 0:
            print('Training loss (for one batch) at step %s: %s' % (step, float(loss_value)))
            print('Seen so far: %s samples' % ((step + 1) * 64))
```
- 利用[`tf.GradientTape()`](https://www.tensorflow.org/api_docs/python/tf/GradientTape)训练的固定步骤:
    - 计算loss:
    - 计算梯度: [`tape.gradient`](https://www.tensorflow.org/api_docs/python/tf/GradientTape#gradient)
    - 反向传播(Apply gradients to variables): [`optimizer.gradient`](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam#apply_gradients)
### 7.2 构建网络,训练并验证
```python
# 载入数据
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') /255
x_test = x_test.reshape(10000, 784).astype('float32') /255

x_val = x_train[-10000:]
y_val = y_train[-10000:]

x_train = x_train[:-10000]
y_train = y_train[:-10000]

# 训练并验证
# 获取模型
inputs = keras.Input(shape=(784,), name='digits')
x = keras.layers.Dense(64, activation='relu', name='dense_1')(inputs)
x = keras.layers.Dense(64, activation='relu', name='dense_2')(x)
outputs = keras.layers.Dense(10, activation='softmax', name='predictions')(x)
model = keras.Model(inputs=inputs, outputs=outputs)

# sgd优化器
optimizer = keras.optimizers.SGD(learning_rate=1e-3)
# 分类损失函数
loss_fn = keras.losses.SparseCategoricalCrossentropy()

# 设定统计参数
train_acc_metric = keras.metrics.SparseCategoricalAccuracy() 
val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

# 准备训练数据
batch_size = 64
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

# 准备验证数据
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(64)


# 迭代训练
for epoch in range(2):
    print('Start of epoch %d' % (epoch,))
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            logits = model(x_batch_train)
            loss_value = loss_fn(y_batch_train, logits)
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # 更新统计传输
    train_acc_metric(y_batch_train, logits)

    # 输出
    if step % 200 == 0:
        print('Training loss (for one batch) at step %s: %s' % (step, float(loss_value)))
        print('Seen so far: %s samples' % ((step + 1) * 64))

  # 输出统计参数的值
train_acc = train_acc_metric.result()
print('Training acc over epoch: %s' % (float(train_acc),))
# 重置统计参数
train_acc_metric.reset_states()

# 用模型进行验证
for x_batch_val, y_batch_val in val_dataset:
    val_logits = model(x_batch_val)
    # 根据验证的统计参数
    val_acc_metric(y_batch_val, val_logits)
val_acc = val_acc_metric.result()
val_acc_metric.reset_states()
print('Validation acc: %s' % (float(val_acc),))
```
### 7.3 构造自定义loss,并训练
一下代码段中涉及的API：
- [`model.losses`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer#losses): Losses which are associated with this Layer. Variable regularization tensors are created when this property is accessed, so it is eager safe: accessing losses under a `tf.GradientTape` will propagate gradients back to the corresponding variables. Reture a list of tensors.
```python
# 载入数据
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') /255
x_test = x_test.reshape(10000, 784).astype('float32') /255

x_val = x_train[-10000:]
y_val = y_train[-10000:]

x_train = x_train[:-10000]
y_train = y_train[:-10000]

# 准备数据.
batch_size = 64
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

##　添加自己构造的loss, 每次只能看到最新一次训练增加的loss
class ActivityRegularizationLayer(keras.layers.Layer):
    def call(self, inputs):
        self.add_loss(1e-2 * tf.reduce_sum(inputs))
        return inputs

inputs = keras.Input(shape=(784,), name='digits')
x = keras.layers.Dense(64, activation='relu', name='dense_1')(inputs)
# Insert activity regularization as a layer
x = ActivityRegularizationLayer()(x)
x = keras.layers.Dense(64, activation='relu', name='dense_2')(x)
outputs = keras.layers.Dense(10, activation='softmax', name='predictions')(x)

model = keras.Model(inputs=inputs, outputs=outputs)
logits = model(x_train[:64])
print(model.losses)  
# logits = model(x_train[:64])
logits = model(x_train[64: 128])
logits = model(x_train[128: 192])
# print(model.losses)

# 损失函数.
loss_fn = keras.losses.SparseCategoricalCrossentropy()

# 将loss添加进求导中
optimizer = keras.optimizers.SGD(learning_rate=1e-3)

for epoch in range(3):
    print('Start of epoch %d' % (epoch,))

    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            logits = model(x_batch_train)
            loss_value = loss_fn(y_batch_train, logits)

            # 添加额外的loss
            loss_value += sum(model.losses)

            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # 每200个batch输出一次学习.
        if step % 200 == 0:
            print('Training loss (for one batch) at step %s: %s' % (step, float(loss_value)))
            print('Seen so far: %s samples' % ((step + 1) * 64))
```