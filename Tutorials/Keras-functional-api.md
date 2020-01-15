# Keras 函数式 API
<!-- TOC -->

- [Keras 函数式 API](#keras-函数式-api)
    - [1. 利用函数式API构建简单网络](#1-利用函数式api构建简单网络)
    - [1.1 创建网络](#11-创建网络)
    - [1.2 训练,验证及测试](#12-训练验证及测试)
    - [2. 使用共享网络创建多个模型](#2-使用共享网络创建多个模型)
    - [3. 复杂网络结构构建](#3-复杂网络结构构建)
        - [3.1 多输入与多输出网络](#31-多输入与多输出网络)
        - [3.2 小型残差网络](#32-小型残差网络)
    - [4. 共享网络层](#4-共享网络层)
    - [5. 模型复用](#5-模型复用)
    - [6. 自定义网络层](#6-自定义网络层)

<!-- /TOC -->
## 1. 利用函数式API构建简单网络
## 1.1 创建网络
```python
inputs = tf.keras.Input(shape=(784,), name='img')
h1 = layers.Dense(32, activation='relu')(inputs)
h2 = layers.Dense(32, activation='relu')(h1)
outputs = layers.Dense(10, activation='softmax')(h2)
model = tf.keras.Model(inputs=inputs, outputs=outputs, name='mnist_model') # name中不要有空格, 否则会报错: ValueError: 'mnist model' is not a valid scope name

model.summary()
keras.utils.plot_model(model, 'mnist_model.png') # Converts a Keras model to dot format and save to a file.
keras.utils.plot_model(model, 'model_info.png', show_shapes=True)
```
[`keras.utils.plot_model`](https://www.tensorflow.org/api_docs/python/tf/keras/utils/plot_model): Converts a Keras model to dot format and save to a file. Reture a Jupyter notebook Image object if Jupyter is installed. This enables in-line display of the model plots in notebooks. 可能会报错,[见这里](https://blog.csdn.net/xovee/article/details/91952352).
```python
tf.keras.utils.plot_model(
    model, # A Keras model instance
    to_file='model.png',
    show_shapes=False, # whether to display shape information.
    show_layer_names=True, # whether to display layer names.
    rankdir='TB', # rankdir argument passed to PyDot, a string specifying the format of the plot: 'TB' creates a vertical plot; 'LR' creates a horizontal plot.
    expand_nested=False, # Whether to expand nested models into clusters.
    dpi=96 # Dots per inch.
)
```
## 1.2 训练,验证及测试
```python
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') /255
x_test = x_test.reshape(10000, 784).astype('float32') /255
model.compile(optimizer=keras.optimizers.RMSprop(),
            loss='sparse_categorical_crossentropy', 
            metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=64, epochs=5, validation_split=0.2)
test_scores = model.evaluate(x_test, y_test, verbose=0)
print('test loss:', test_scores[0])
print('test acc:', test_scores[1])
```
- 模型保存和恢复
```python
model.save('model_save.h5')
del model
model = keras.models.load_model('model_save.h5')
```
## 2. 使用共享网络创建多个模型  
在函数API中，通过在**图层图**中指定其输入和输出来创建模型。 这意味着可以使用单个图层图来生成多个模型。  
下面两个例子中，第一个例子创建了一个编码器和一个解码器；第二个例子不仅创建了一个编码器和一个解码器，还将编码器的输出作为解码器的输入。
```python
# 编码器网络和自编码器网络
# 编码
encode_input = keras.Input(shape=(28,28,1), name='img')
h1 = keras.layers.Conv2D(16, 3, activation='relu')(encode_input)
h1 = keras.layers.Conv2D(32, 3, activation='relu')(h1)
h1 = keras.layers.MaxPool2D(3)(h1)
h1 = keras.layers.Conv2D(32, 3, activation='relu')(h1)
h1 = keras.layers.Conv2D(16, 3, activation='relu')(h1)
encode_output = keras.layers.GlobalMaxPool2D()(h1)

encode_model = keras.Model(inputs=encode_input, outputs=encode_output, name='encoder')
encode_model.summary()
# 解码
h2 = keras.layers.Reshape((4, 4, 1))(encode_output)
h2 = keras.layers.Conv2DTranspose(16, 3, activation='relu')(h2)
h2 = keras.layers.Conv2DTranspose(32, 3, activation='relu')(h2)
h2 = keras.layers.UpSampling2D(3)(h2)
h2 = keras.layers.Conv2DTranspose(16, 3, activation='relu')(h2)
decode_output = keras.layers.Conv2DTranspose(1, 3, activation='relu')(h2)

autoencoder = keras.Model(inputs=encode_input, outputs=decode_output, name='autoencoder')
autoencoder.summary()
```
可以把整个模型当作一层网络使用:
```python
encode_input = keras.Input(shape=(28,28,1), name='src_img')
h1 = keras.layers.Conv2D(16, 3, activation='relu')(encode_input)
h1 = keras.layers.Conv2D(32, 3, activation='relu')(h1)
h1 = keras.layers.MaxPool2D(3)(h1)
h1 = keras.layers.Conv2D(32, 3, activation='relu')(h1)
h1 = keras.layers.Conv2D(16, 3, activation='relu')(h1)
encode_output = keras.layers.GlobalMaxPool2D()(h1)

encode_model = keras.Model(inputs=encode_input, outputs=encode_output, name='encoder')
encode_model.summary()

decode_input = keras.Input(shape=(16,), name='encoded_img')
h2 = keras.layers.Reshape((4, 4, 1))(decode_input)
h2 = keras.layers.Conv2DTranspose(16, 3, activation='relu')(h2)
h2 = keras.layers.Conv2DTranspose(32, 3, activation='relu')(h2)
h2 = keras.layers.UpSampling2D(3)(h2)
h2 = keras.layers.Conv2DTranspose(16, 3, activation='relu')(h2)
decode_output = keras.layers.Conv2DTranspose(1, 3, activation='relu')(h2)
decode_model = keras.Model(inputs=decode_input, outputs=decode_output, name='decoder')
decode_model.summary()

# 重点！！！
autoencoder_input = keras.Input(shape=(28,28,1), name='img')
h3 = encode_model(autoencoder_input)
autoencoder_output = decode_model(h3)
autoencoder = keras.Model(inputs=autoencoder_input, outputs=autoencoder_output,
                        name='autoencoder')
autoencoder.summary()
```
## 3. 复杂网络结构构建
### 3.1 多输入与多输出网络
```python
# 构建一个根据文档内容、标签和标题，预测文档优先级和执行部门的网络
# 超参
num_words = 2000
num_tags = 12
num_departments = 4

# 输入
body_input = keras.Input(shape=(None,), name='body')
title_input = keras.Input(shape=(None,), name='title')
tag_input = keras.Input(shape=(num_tags,), name='tag')

# 嵌入层
body_feat = keras.layers.Embedding(num_words, 64)(body_input)
title_feat = keras.layers.Embedding(num_words, 64)(title_input)

# 特征提取层
body_feat = keras.layers.LSTM(32)(body_feat)
title_feat = keras.layers.LSTM(128)(title_feat)
features = keras.layers.concatenate([title_feat,body_feat, tag_input])  # 将body和title的特征组合在一起

# 分类层
priority_pred = keras.layers.Dense(1, activation='sigmoid', name='priority')(features)
department_pred = keras.layers.Dense(num_departments, activation='softmax', name='department')(features)


# 构建模型
model = keras.Model(inputs=[body_input, title_input, tag_input], # inputs使用了一个列表,多输入
                    outputs=[priority_pred, department_pred]) # 多输出
model.summary()
# keras.utils.plot_model(model, 'multi_model.png', show_shapes=True)
model.compile(optimizer=keras.optimizers.RMSprop(1e-3),
            loss={'priority': 'binary_crossentropy',
                'department': 'categorical_crossentropy'},
            loss_weights=[1., 0.2])

# 载入输入数据
title_data = np.random.randint(num_words, size=(1280, 10))
body_data = np.random.randint(num_words, size=(1280, 100))
tag_data = np.random.randint(2, size=(1280, num_tags)).astype('float32')
# 标签
priority_label = np.random.random(size=(1280, 1))
department_label = np.random.randint(2, size=(1280, num_departments))
# 训练
history = model.fit(
    {'title': title_data, 'body':body_data, 'tag':tag_data}, # 多输入
    {'priority':priority_label, 'department':department_label}, # 多输出
    batch_size=32,
    epochs=5
)
```
### 3.2 小型残差网络
```python
inputs = keras.Input(shape=(32,32,3), name='img')
h1 = keras.layers.Conv2D(32, 3, activation='relu')(inputs)
h1 = keras.layers.Conv2D(64, 3, activation='relu')(h1)
block1_out = keras.layers.MaxPooling2D(3)(h1)

h2 = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(block1_out)
h2 = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(h2)
block2_out = keras.layers.add([h2, block1_out])

h3 = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(block2_out)
h3 = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(h3)
block3_out = keras.layers.add([h3, block2_out])

h4 = keras.layers.Conv2D(64, 3, activation='relu')(block3_out)
h4 = keras.layers.GlobalMaxPool2D()(h4)
h4 = keras.layers.Dense(256, activation='relu')(h4)
h4 = keras.layers.Dropout(0.5)(h4)
outputs = keras.layers.Dense(10, activation='softmax')(h4)

model = keras.Model(inputs, outputs, name='small_resnet')
model.summary()
# keras.utils.plot_model(model, 'small_resnet_model.png', show_shapes=True)
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32') / 255
x_test = y_train.astype('float32') / 255
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

model.compile(optimizer=keras.optimizers.RMSprop(1e-3),
            loss='categorical_crossentropy',
            metrics=['acc'])
model.fit(x_train, y_train,
        batch_size=64,
        epochs=1,
        validation_split=0.2)

#model.predict(x_test, batch_size=32)       
```
- [`keras.layers.add`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/add): Functional interface to the Add layer. Return a tensor, the sum of the inputs.
## 4. 共享网络层
```python
share_embedding = keras.layers.Embedding(1000, 64)

input1 = keras.Input(shape=(None,), dtype='int32')
input2 = keras.Input(shape=(None,), dtype='int32')

feat1 = share_embedding(input1) # 共享网络层,即使用同一个layer处理数据...
feat2 = share_embedding(input2)
```
## 5. 模型复用  
```python  
from tensorflow.keras.applications import VGG16
vgg16=VGG16()

feature_list = [layer.output for layer in vgg16.layers]
feat_ext_model = keras.Model(inputs=vgg16.input, outputs=feature_list)

img = np.random.random((1, 224, 224, 3).astype('float32'))
ext_features = feat_ext_model(img)       
```
- Resources:  
    - [Tensorflow 2.0 Official Transfer Learning Tutorial](https://www.tensorflow.org/tutorials/images/transfer_learning)
## 6. 自定义网络层
以下代码中用到的几个API
- [`get_config`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer#get_config): Returns the config of the layer. A layer config is a Python dictionary (serializable) containing the configuration of a layer. The same layer can be reinstantiated later (without its trained weights) from this configuration. The config of a layer does not include connectivity information, nor the layer class name. These are handled by Network (one layer of abstraction above).  
- [`from_config or model_from_config`](https://www.tensorflow.org/api_docs/python/tf/keras/models/model_from_config)
```python
class MyDense(keras.layers.Layer):
    def __init__(self, units=32):
        super(MyDense, self).__init__()
        self.units = units
    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                initializer='random_normal',
                                trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                initializer='random_normal',
                                trainable=True)
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):  # 重写该方法
        return {'units': self.units}

inputs = keras.Input((4,))
outputs = MyDense(10)(inputs)
model = keras.Model(inputs, outputs) # 生成了一个keras.Model实例，该模型的输入为inputs，输出为MyDense的输出，即，该模型只有MyDense一层

config = model.get_config() # 利用get_config()方法获取配置
new_model = keras.Model.from_config( # 利用from_config()方法生成一个新的模型
            config, custom_objects={'MyDense':MyDense}
            )
```
下面的代码段先创建了一个MyRnn层，然后展示了，可以将MyRnn接在其他层的后面构成一个model：
```python
# 在自定义网络层调用其他网络层
# 超参
time_step = 10
batch_size = 32
hidden_dim = 32
inputs_dim = 5

# 网络 (没有调用MyDense)
class MyRnn(keras.layers.Layer):
    def __init__(self):
        super(MyRnn, self).__init__()
        self.hidden_dim = hidden_dim
        self.projection1 = keras.layers.Dense(units=hidden_dim, activation='relu')
        self.projection2 = keras.layers.Dense(units=hidden_dim, activation='relu')
        self.classifier = keras.layers.Dense(1, activation='sigmoid')
    def call(self, inputs):
        outs = []
        states = tf.zeros(shape=[inputs.shape[0], self.hidden_dim])
        for t in range(inputs.shape[1]):
            x = inputs[:,t,:]
            h = self.projection1(x)
            y = h + self.projection2(states)
            states = y
            outs.append(y)
        # print(outs)
        features = tf.stack(outs, axis=1)
        print(features.shape)
        return self.classifier(features)

# 构建网络
inputs = keras.Input(batch_shape=(batch_size, time_step, inputs_dim))
x = keras.layers.Conv1D(32, 3)(inputs)
print(x.shape)
outputs = MyRnn()(x)  # 将MyRnn层接在了一个Conv1D层后面
model = keras.Model(inputs, outputs) # 该模型实际包含了一个Conv1D和MyRnn
_ = model(tf.zeros((1, 10, 5)))

rnn_model = MyRnn()  # 单独一个MyRnn
_ = rnn_model(tf.zeros((1, 10, 5)))
```