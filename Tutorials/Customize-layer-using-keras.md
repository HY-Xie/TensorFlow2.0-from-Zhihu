# 用keras构建自己的网络层
<!-- TOC -->

- [用keras构建自己的网络层](#用keras构建自己的网络层)
    - [1. 构建一个简单的网络层](#1-构建一个简单的网络层)
        - [1.1 已知网络输入大小的情况-1](#11-已知网络输入大小的情况-1)
        - [1.2 已知网络输入大小的情况-2](#12-已知网络输入大小的情况-2)
        - [1.2 未知网络输入大小的情况](#12-未知网络输入大小的情况)
    - [2. 使用子层递归构建网络层](#2-使用子层递归构建网络层)
        - [2.1 一般的递归构建](#21-一般的递归构建)
        - [2.2 通过构建网络收集loss](#22-通过构建网络收集loss)
    - [3. 其他网络层配置](#3-其他网络层配置)
        - [3.1 序列化自定义层](#31-序列化自定义层)
    - [4. 构建自己的模型](#4-构建自己的模型)
        - [4.1 Model类与Layer的区别](#41-model类与layer的区别)
        - [4.2 实例](#42-实例)

<!-- /TOC -->
## 1. 构建一个简单的网络层
### 1.1 已知网络输入大小的情况-1
```python
import tensorflow as tf
tf.keras.backend.clear_session()  # Destroys the current TF graph and creates a new one.
import tensorflow.keras as keras
# 定义网络层就是：设置网络权重和输出到输入的计算过程
class MyLayer(keras.layers.Layer):
    def __init__(self, input_dim=32, unit=32): # 此处假设已知input大小
        super(MyLayer, self).__init__()
        
        w_init = tf.random_normal_initializer()
        self.weight = tf.Variable(initial_value=w_init(
            shape=(input_dim, unit), dtype=tf.float32), trainable=True)
        
        b_init = tf.zeros_initializer()
        self.bias = tf.Variable(initial_value=b_init(
            shape=(unit,), dtype=tf.float32), trainable=True)
    
    def call(self, inputs):
        return tf.matmul(inputs, self.weight) + self.bias
        
x = tf.ones((3,5))
my_layer = MyLayer(5, 4)
out = my_layer(x)
print(out)
"""
Output:
tf.Tensor(
[[ 0.02612966 -0.03196752 -0.00892349 -0.02248625]
 [ 0.02612966 -0.03196752 -0.00892349 -0.02248625]
 [ 0.02612966 -0.03196752 -0.00892349 -0.02248625]], shape=(3, 4), dtype=float32)
"""
```
### 1.2 已知网络输入大小的情况-2
按上面构建网络层，图层会自动跟踪权重w和b，当然我们也可以直接用add_weight的方法构建权重：
```python
import tensorflow as tf
tf.keras.backend.clear_session()  # Destroys the current TF graph and creates a new one.
import tensorflow.keras as keras

class MyLayer(keras.layers.Layer):
    def __init__(self, input_dim=32, unit=32):  # input_dim 和 unit 代表的东西不一样
        super(MyLayer, self).__init__()
        self.weight = self.add_weight(shape=(input_dim, unit),
                                     initializer=keras.initializers.RandomNormal(),
                                     trainable=True)
        self.bias = self.add_weight(shape=(unit,),
                                   initializer=keras.initializers.Zeros(),
                                   trainable=True)
    
    def call(self, inputs):
        # return tf.matmul(inputs, self.weight) + self.bias
        return inputs @ self.weight + self.bias
        
x = tf.ones((3,5))
my_layer = MyLayer(5, 4)
out = my_layer(x)
print(out)
"""
Output:
tf.Tensor(
[[ 0.01390542 -0.08676334 -0.08993102 -0.05397398]
 [ 0.01390542 -0.08676334 -0.08993102 -0.05397398]
 [ 0.01390542 -0.08676334 -0.08993102 -0.05397398]], shape=(3, 4), dtype=float32)
 """
```
也可以设置**不可训练**的权重:
```python
import tensorflow as tf
tf.keras.backend.clear_session()  # Destroys the current TF graph and creates a new one.
import tensorflow.keras as keras

class AddLayer(keras.layers.Layer):
    def __init__(self, input_dim=32):
        super(AddLayer, self).__init__()
        self.sum = self.add_weight(shape=(input_dim,),
                                     initializer=keras.initializers.Zeros(),
                                     trainable=False)  # 不可训练！！！
       
    
    def call(self, inputs):
        self.sum.assign_add(tf.reduce_sum(inputs, axis=0))  # assign_add 可以自更新
        return self.sum
        
x = tf.ones((1,3))
print(x.shape) # (1, 3) 1个样本，三个特征，即网络输入是3
my_layer = AddLayer(3)
out = my_layer(x)
print(out.numpy())
out = my_layer(x)
print(out.numpy())
print('weight:', my_layer.weights)
print('non-trainable weight:', my_layer.non_trainable_weights)
print('trainable weight:', my_layer.trainable_weights) # trainable weight: []
```
### 1.2 未知网络输入大小的情况
当定义网络时不知道网络的维度时，**可以重写[`build(input_shape)`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer#build)函数**，用获得的shape构建网络:
```python
import tensorflow as tf
tf.keras.backend.clear_session()  # Destroys the current TF graph and creates a new one.
import tensorflow.keras as keras

class MyLayer(keras.layers.Layer):
    def __init__(self, unit=32): 
        super(MyLayer, self).__init__()
        self.unit = unit
        
    def build(self, input_shape):  # IMPORTANT!!!
        self.weight = self.add_weight(shape=(input_shape[-1], self.unit), 
                                     initializer=keras.initializers.RandomNormal(),
                                     trainable=True)
        self.bias = self.add_weight(shape=(self.unit,),
                                   initializer=keras.initializers.Zeros(),
                                   trainable=True)
    
    def call(self, inputs):
        return tf.matmul(inputs, self.weight) + self.bias
        


my_layer = MyLayer(3)
x = tf.ones((3,5)) # 输入维度是5
out = my_layer(x)
print(out)
```
## 2. 使用子层递归构建网络层
### 2.1 一般的递归构建 
包含重写和不重写`build`两种方式的使用：
```python
class MyLayer(keras.layers.Layer):
    def __init__(self, unit=32): 
        super(MyLayer, self).__init__()
        self.unit = unit
        
    def build(self, input_shape):  # IMPORTANT!!!
        self.weight = self.add_weight(shape=(input_shape[-1], self.unit), 
                                     initializer=keras.initializers.RandomNormal(),
                                     trainable=True)
        self.bias = self.add_weight(shape=(self.unit,),
                                   initializer=keras.initializers.Zeros(),
                                   trainable=True)
    
    def call(self, inputs):
        return tf.matmul(inputs, self.weight) + self.bias

class MyBlock(keras.layers.Layer):
    def __init__(self):
        super(MyBlock, self).__init__()
        self.layer1 = MyLayer(32)
        self.layer2 = MyLayer(16)
        self.layer3 = MyLayer(2)
    def call(self, inputs):
        h1 = self.layer1(inputs)
        h1 = tf.nn.relu(h1)
        h2 = self.layer2(h1)
        h2 = tf.nn.relu(h2)
        return self.layer3(h2)
    
my_block = MyBlock()
print('trainable weights:', len(my_block.trainable_weights)) # trainable weights: 0 网络还没没有真正构建

my_block.build(input_shape=tf.ones(shape=(3,64)).shape[-1])
print('trainable weights:', len(my_block.trainable_weights)) # trainable weights: 0 网络还没没有真正构建

y = my_block(tf.ones(shape=(3, 64)))
# 构建网络在build()里面，所以必须执行了才有网络
print('trainable weights:', len(my_block.trainable_weights)) # trainable weights: 6 必须执行了才有网络
```
下面是一个没有重新`build(input_shape)`的例子:
```python
import tensorflow as tf
tf.keras.backend.clear_session()  # Destroys the current TF graph and creates a new one.
import tensorflow.keras as keras

class MyLayer(keras.layers.Layer):
    def __init__(self, unit=32): 
        super(MyLayer, self).__init__()
        self.unit = unit
        
    
        self.weight = self.add_weight(shape=(64, self.unit), 
                                     initializer=keras.initializers.RandomNormal(),
                                     trainable=True)
        self.bias = self.add_weight(shape=(self.unit,),
                                   initializer=keras.initializers.Zeros(),
                                   trainable=True)
    
    def call(self, inputs):
        return tf.matmul(inputs, self.weight) + self.bias

class MyBlock(keras.layers.Layer):
    def __init__(self):
        super(MyBlock, self).__init__()
        self.layer1 = MyLayer(32)
        self.layer2 = MyLayer(16)
        self.layer3 = MyLayer(2)
    def call(self, inputs):
        h1 = self.layer1(inputs)
        h1 = tf.nn.relu(h1)
        h2 = self.layer2(h1)
        h2 = tf.nn.relu(h2)
        return self.layer3(h2)
    
my_block = MyBlock()
# print('trainable weights:', len(my_block.trainable_weights)) # trainable weights: 6 没有重写build函数，生成对象之后，网络便存在

my_block.build(input_shape=tf.ones(shape=(3,64)).shape[-1])
print('trainable weights:', len(my_block.trainable_weights)) # trainable weights: 6 

# y = my_block(tf.ones(shape=(3, 64)))
# # 构建网络在build()里面，所以必须执行了才有网络
# print('trainable weights:', len(my_block.trainable_weights)) # trainable weights: 6 
```
### 2.2 通过构建网络收集loss
```python
import tensorflow as tf
tf.keras.backend.clear_session()  # Destroys the current TF graph and creates a new one.
import tensorflow.keras as keras

class LossLayer(keras.layers.Layer):
  
    def __init__(self, rate=1e-2):
        super(LossLayer, self).__init__()
        self.rate = rate
  
    def call(self, inputs):
        self.add_loss(self.rate * tf.reduce_sum(inputs))
        return inputs

class OutLayer(keras.layers.Layer):
    def __init__(self):
        super(OutLayer, self).__init__()
        self.loss_fun=LossLayer(1e-2)  # 这里调用了自定义的LossLayer
    def call(self, inputs):
        return self.loss_fun(inputs)
    
my_layer = OutLayer()
print(len(my_layer.losses)) # 还未call
y = my_layer(tf.zeros(1,1))
print(len(my_layer.losses)) # 执行call之后
y = my_layer(tf.zeros(1,1))
print(len(my_layer.losses)) # call之前会重新置0
print(my_layer.losses) # [<tf.Tensor: id=17, shape=(), dtype=float32, numpy=0.0>]
```
如果中间调用了keras网络层，里面的正则化loss也会被加入进来:
```python
import tensorflow as tf
tf.keras.backend.clear_session()  # Destroys the current TF graph and creates a new one.
import tensorflow.keras as keras

class OuterLayer(keras.layers.Layer):
    def __init__(self):
        super(OuterLayer, self).__init__()
        self.dense = keras.layers.Dense(32, kernel_regularizer=tf.keras.regularizers.l2(1e-3))
    
    def call(self, inputs):
        return self.dense(inputs)

my_layer = OuterLayer()
y = my_layer(tf.zeros((1,1)))
print(my_layer.losses) 
print(my_layer.weights) 
```
## 3. 其他网络层配置
### 3.1 序列化自定义层
```python
import tensorflow as tf
tf.keras.backend.clear_session()  # Destroys the current TF graph and creates a new one.
import tensorflow.keras as keras

class Linear(keras.layers.Layer):
    def __init__(self, units=32, **kwargs):
        super(Linear, self).__init__(**kwargs)
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

    def get_config(self):
        config = super(Linear, self).get_config()
        config.update({'units':self.units})
        return config

layer = Linear(125)
config = layer.get_config()
print(config) 
# Output: {'name': 'linear', 'trainable': True, 'dtype': 'float32', 'units': 125}
```
配置只有训练时可以执行的网络层:
```python
class MyDropout(keras.layers.Layer):
    def __init__(self, rate, **kwargs):
        super(MyDropout, self).__init__(**kwargs)
        self.rate = rate
    def call(self, inputs, training=None):  # 多了training参数
        return tf.cond(training, 
                       lambda: tf.nn.dropout(inputs, rate=self.rate),
                      lambda: inputs)
```
- [`tf.cond`](https://www.tensorflow.org/api_docs/python/tf/cond): Return `true_fn()` if the predicate pred is true else `false_fn()`.
    ```python
    tf.cond(
    pred,
    true_fn=None,
    false_fn=None,
    name=None
    )
    ```
## 4. 构建自己的模型
通常，我们使用Layer类来定义内部计算块，并使用Model类来定义外部模型 - 即要训练的对象。
### 4.1 Model类与Layer的区别
- `Model`它公开了内置的训练，评估和预测循环`model.fit(),model.evaluate(),model.predict()`.
- `Model`通过`model.layers`属性公开其内层列表。
- `Model`公开了保存和序列化API.
### 4.2 实例
- 下面通过构建一个变分自编码器（VAE），来介绍如何构建自己的网络。
    ```python
    import tensorflow as tf
    tf.keras.backend.clear_session()  # Destroys the current TF graph and creates a new one.
    import tensorflow.keras as keras

    # 采样网络
    class Sampling(keras.layers.Layer):
        def call(self, inputs):
            z_mean, z_log_var = inputs
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5*z_log_var) * epsilon
    # 编码器
    class Encoder(keras.layers.Layer):  # 利用了keras的API Dense
        def __init__(self, latent_dim=32, 
                    intermediate_dim=64, name='encoder', **kwargs):
            super(Encoder, self).__init__(name=name, **kwargs)
            self.dense_proj = keras.layers.Dense(intermediate_dim, activation='relu')
            self.dense_mean = keras.layers.Dense(latent_dim)
            self.dense_log_var = keras.layers.Dense(latent_dim)
            self.sampling = Sampling() # 嵌套了未使用keras API的自定义Sampling
            
        def call(self, inputs):
            h1 = self.dense_proj(inputs)
            z_mean = self.dense_mean(h1)
            z_log_var = self.dense_log_var(h1)
            z = self.sampling((z_mean, z_log_var))
            return z_mean, z_log_var, z
            
    # 解码器
    class Decoder(keras.layers.Layer):
        def __init__(self, original_dim, 
                    intermediate_dim=64, name='decoder', **kwargs):
            super(Decoder, self).__init__(name=name, **kwargs)
            self.dense_proj = keras.layers.Dense(intermediate_dim, activation='relu')
            self.dense_output = keras.layers.Dense(original_dim, activation='sigmoid')
        def call(self, inputs):
            h1 = self.dense_proj(inputs)
            return self.dense_output(h1)
        
    # 变分自编码器
    class VAE(keras.Model):
        def __init__(self, original_dim, latent_dim=32, 
                    intermediate_dim=64, name='encoder', **kwargs):
            super(VAE, self).__init__(name=name, **kwargs)
        
            self.original_dim = original_dim
            self.encoder = Encoder(latent_dim=latent_dim,
                                intermediate_dim=intermediate_dim)
            self.decoder = Decoder(original_dim=original_dim,
                                intermediate_dim=intermediate_dim)
        def call(self, inputs):
            z_mean, z_log_var, z = self.encoder(inputs)
            reconstructed = self.decoder(z)
            
            kl_loss = -0.5*tf.reduce_sum(
                z_log_var-tf.square(z_mean)-tf.exp(z_log_var)+1)
            self.add_loss(kl_loss)
            return reconstructed

    (x_train, _), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(60000, 784).astype('float32') / 255
    vae = VAE(784,32,64)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    vae.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())
    vae.fit(x_train, x_train, epochs=3, batch_size=64)
    ```
- 以下使用自定义训练循环
```python
import tensorflow as tf
tf.keras.backend.clear_session()  # Destroys the current TF graph and creates a new one.
import tensorflow.keras as keras

# 采样网络
class Sampling(keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5*z_log_var) * epsilon
# 编码器
class Encoder(keras.layers.Layer):  # 利用了keras的API Dense
    def __init__(self, latent_dim=32, 
                intermediate_dim=64, name='encoder', **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.dense_proj = keras.layers.Dense(intermediate_dim, activation='relu')
        self.dense_mean = keras.layers.Dense(latent_dim)
        self.dense_log_var = keras.layers.Dense(latent_dim)
        self.sampling = Sampling() # 嵌套了未使用keras API的自定义Sampling
        
    def call(self, inputs):
        h1 = self.dense_proj(inputs)
        z_mean = self.dense_mean(h1)
        z_log_var = self.dense_log_var(h1)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z
        
# 解码器
class Decoder(keras.layers.Layer):
    def __init__(self, original_dim, 
                 intermediate_dim=64, name='decoder', **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.dense_proj = keras.layers.Dense(intermediate_dim, activation='relu')
        self.dense_output = keras.layers.Dense(original_dim, activation='sigmoid')
    def call(self, inputs):
        h1 = self.dense_proj(inputs)
        return self.dense_output(h1)
    
# 变分自编码器
class VAE(keras.Model):
    def __init__(self, original_dim, latent_dim=32, 
                intermediate_dim=64, name='encoder', **kwargs):
        super(VAE, self).__init__(name=name, **kwargs)
    
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim,
                              intermediate_dim=intermediate_dim)
        self.decoder = Decoder(original_dim=original_dim,
                              intermediate_dim=intermediate_dim)
    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        
        kl_loss = -0.5*tf.reduce_sum(
            z_log_var-tf.square(z_mean)-tf.exp(z_log_var)+1)
        self.add_loss(kl_loss)
        return reconstructed

(x_train, _), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255
train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

original_dim = 784
vae = VAE(original_dim, 64, 32)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
mse_loss_fn = tf.keras.losses.MeanSquaredError()

loss_metric = tf.keras.metrics.Mean()

# 每个epoch迭代.
for epoch in range(3):
    print('Start of epoch %d' % (epoch,))

    # 取出每个batch的数据并训练.
    for step, x_batch_train in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            reconstructed = vae(x_batch_train)
            # 计算 reconstruction loss
            loss = mse_loss_fn(x_batch_train, reconstructed)
            loss += sum(vae.losses)  # 添加 KLD regularization loss

            grads = tape.gradient(loss, vae.trainable_variables)
            optimizer.apply_gradients(zip(grads, vae.trainable_variables))
        loss_metric(loss)
        if step % 100 == 0:
            print('step %s: mean loss = %s' % (step, loss_metric.result()))
```
**可以看到，二者定义的模型完全相同，可以使用keras提供的compile&fit方法，也可以自定义训练过程。**
