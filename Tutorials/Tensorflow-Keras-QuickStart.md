# Tensorflow 2.0 基础 - Keras快速入门
Keras 是一个用于构建和训练深度学习模型的高阶 API。它可用于快速设计原型、高级研究和生产。  keras的3个优点： 方便用户使用、模块化和可组合、易于扩展.
<!-- TOC -->

- [Tensorflow 2.0 基础 - Keras快速入门](#tensorflow-20-基础---keras快速入门)
    - [1. 导入tf.keras.](#1-导入tfkeras)
    - [2. 构建简单模型](#2-构建简单模型)
        - [2.1 模型堆叠](#21-模型堆叠)
        - [2.2 网络配置](#22-网络配置)
    - [3. 训练和评估](#3-训练和评估)
        - [3.1 训练流程](#31-训练流程)
        - [3.2 输入Numpy数据](#32-输入numpy数据)
        - [3.3 tf.data输入数据](#33-tfdata输入数据)
        - [3.4 评估与预测](#34-评估与预测)
    - [4. 构建高级模型](#4-构建高级模型)
        - [4.1 函数式API](#41-函数式api)
        - [4.2 模型子类化](#42-模型子类化)
        - [4.3 自定义层](#43-自定义层)
        - [4.4 回调](#44-回调)
    - [5. 保持和恢复](#5-保持和恢复)
        - [5.1 保存权重](#51-保存权重)
        - [5.2 保存网络结构](#52-保存网络结构)
        - [5.3 保存整个模型](#53-保存整个模型)
    - [6. 将keras用于Estimator](#6-将keras用于estimator)

<!-- /TOC -->
## 1. 导入tf.keras. 
Tensorflow 2.0 推荐使用keras构建网络，常见的神经网络都包含在keras.layers中(最新的tf.keras的版本可能和keras不同)  

```python
import tensorflow as tf 
from tensorflow import keras
print(tf.__version__)  # '2.0.0'
print(keras.__version__)  # '2.0.0'
```

## 2. 构建简单模型  
### 2.1 模型堆叠  
常见的模型类型是层的堆叠：[`tf.keras.Sequential`](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential) 模型: (注意最后一条注释)
```python
model = tf.keras.Sequential()
model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dense(32, activation=tf.nn.relu))  # 这里可以使用字符串参数'relu'或者 直接使用函数
model.add(keras.layers.Dense(10, activation=tf.nn.softmax))
# model.add_layer(keras.layers.Dense(10, activation='softmax'))  # ERROR. AttributeError: 'Sequential' object has no attribute 'add_layer'
```
        

### 2.2 网络配置 
        
以下为 [keras.layers.Dense 类](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense) (Just regular densely-connected NN layer.) 的构造函数:
```python
__init__(
    units, # Positive integer, dimensionality of the ***output space***.
    activation=None, # Activation function to use. If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x). [设置层的激活函数。此参数由内置函数的名称指定，或指定为可调用对象。默认情况下，系统不会应用任何激活函数。]
    use_bias=True, # Boolean, whether the layer uses a bias vector.
    kernel_initializer='glorot_uniform', # Initializer for the kernel weights matrix. [创建层权重（核）的初始化方案。此参数是一个名称或可调用对象，默认为 "Glorot uniform" 初始化器。]
    bias_initializer='zeros', # Initializer for the bias vector. 
    kernel_regularizer=None, # Regularizer function applied to the kernel weights matrix. [应用层权重（核和偏差）的正则化方案，例如 L1 或 L2 正则化。默认情况下，系统不会应用正则化函数。]
    bias_regularizer=None, # Regularizer function applied to the bias vector.
    activity_regularizer=None, # Regularizer function applied to the output of the layer (its "activation")..
    kernel_constraint=None, # Constraint function applied to the kernel weights matrix.
    bias_constraint=None, # Constraint function applied to the bias vector.
    **kwargs
)
```
关于L1,L2正则化, 可见[知乎文章](https://zhuanlan.zhihu.com/p/35356992);  
额外示例代码:
```python
model.add(keras.layers.Dense(10, activation='sigmoid'))
model.add(keras.layers.Dense(10, activation=tf.nn.sigmoid))

model.add(keras.layers.Dense(32, kernel_initializer='orthogonal'))
model.add(keras.layers.Dense(32, kernel_initializer=keras.initializers.glorot_normal))

model.add(keras.layers.Dense(32, kernel_regularizer=keras.regularizers.l2(0.01)))
model.add(keras.layers.Dense(32, kernel_regularizer=keras.regularizers.l1(0.01)))
```
## 3. 训练和评估
### 3.1 训练流程  
创建好模型后, 通过调用 [`compile`](https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile) (Configures the model for training) 方法配置该模型的学习流程 (`compile`通常与`fit`结合使用，如果使用自定义的训练循环，不适用`compile`,[见这里的第七部分](./Training-model-using-keras.md))：
```python
model.compile(optimizer=keras.optimizers.Adam(0.001),
            loss=tf.losses.categorical_crossentropy,
            metrics=[keras.metrics.categorical_accuracy])
```
其中`loss`也可以是使用keras的: `loss=keras.losses.categorical_crossentropy`  
[compile](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential#compile)(Configures the model for training)的API: (该方法在Sequential模型的文档中) 
```python
compile( # 前三个参数很重要
    optimizer='rmsprop', # String (name of optimizer) or optimizer instance. 
    loss=None, #  String (name of objective function), objective function or tf.losses.Loss instance. If the model has multiple outputs, you can use a different loss on each output by passing a dictionary or a list of losses. The loss value that will be minimized by the model will then be the sum of all individual losses.(解释了如何处理具有多个输出的情况)
    metrics=None, # List of metrics to be evaluated by the model during training and testing. Typically you will use metrics=['accuracy']. To specify different metrics for different outputs of a multi-output model, you could also pass a dictionary, such as metrics={'output_a': 'accuracy', 'output_b': ['accuracy', 'mse']}. You can also pass a list (len = len(outputs)) of lists of metrics such as metrics=[['accuracy'], ['accuracy', 'mse']] or metrics=['accuracy', ['accuracy', 'mse']].
    loss_weights=None, # Optional list or dictionary specifying scalar coefficients (Python floats) to weight the loss contributions of different model outputs. The loss value that will be minimized by the model will then be the weighted sum of all individual losses, weighted by the loss_weights coefficients. If a list, it is expected to have a 1:1 mapping to the model's outputs. If a tensor, it is expected to map output names (strings) to scalar coefficients.
    sample_weight_mode=None, # If you need to do timestep-wise sample weighting (2D weights), set this to "temporal". None defaults to sample-wise weights (1D). If the model has multiple outputs, you can use a different sample_weight_mode on each output by passing a dictionary or a list of modes.
    weighted_metrics=None,  # List of metrics to be evaluated and weighted by sample_weight or class_weight during training and testing.
    target_tensors=None, # By default, Keras will create placeholders for the model's target, which will be fed with the target data during training. If instead you would like to use your own target tensors (in turn, Keras will not expect external Numpy data for these targets at training time), you can specify them via the target_tensors argument. It can be a single tensor (for a single-output model), a list of tensors, or a dict mapping output names to target tensors.
    distribute=None, # TF 2.0 中不再支持, please create and compile the model under distribution strategy scope instead of passing it to compile.
    **kwargs
)
```
### 3.2 输入Numpy数据
```python
import numpy as np 
train_x = np.random.random((1000, 72))
train_y = np.random.random((1000, 10))

val_x = np.random.random((200, 72))
val_y = np.random.random((200, 10))

model.fit(train_x, train_y, epochs=10, batch_size=100, validation_data=(val_x,val_y))
```
[`fit`方法](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential#fit)(Trains the model for a fixed number of epochs (iterations on a dataset))的API:  (该方法在Sequential模型的文档中) 
```python
fit(
    x=None, # Input data. It could be: 1) A Numpy array (or array-like), or a list of arrays (in case the model has multiple inputs); 2) A TensorFlow tensor, or a list of tensors (in case the model has multiple inputs); 3) A dict mapping input names to the corresponding array/tensors, if the model has named inputs; 4) A tf.data dataset. Should return a tuple of either (inputs, targets) or (inputs, targets, sample_weights); 5) A generator or keras.utils.Sequence returning (inputs, targets) or (inputs, targets, sample weights)
    y=None,　# arget data. Like the input data x, it could be either Numpy array(s) or TensorFlow tensor(s). It should be consistent with x (you cannot have Numpy inputs and tensor targets, or inversely). If x is a dataset, generator, or keras.utils.Sequence instance, y should not be specified (since targets will be obtained from x).　
    batch_size=None, # Integer or None. Number of samples per gradient update. If unspecified, batch_size will default to 32. Do not specify the batch_size if your data is in the form of symbolic tensors, datasets, generators, or keras.utils.Sequence instances (since they generate batches).
    epochs=1, #  Integer. Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided. Note that in conjunction with initial_epoch, epochs is to be understood as "final epoch". The model is not trained for a number of iterations given by epochs, but merely until the epoch of index epochs is reached. (与initial_epoch一起使用的时候,意思有所不同!)
    verbose=1, #  0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. Note that the progress bar is not particularly useful when logged to a file, so verbose=2 is recommended when not running interactively (eg, in a production environment). 
    callbacks=None, # List of keras.callbacks.Callback instances. List of callbacks to apply during training. See tf.keras.callbacks. 
    validation_split=0.0, # Float between 0 and 1. Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch. The validation data is selected from the last samples in the x and y data provided, before shuffling. This argument is not supported when x is a dataset, generator or keras.utils.Sequence instance.
    validation_data=None, # Data on which to evaluate the loss and any model metrics at *the end of each epoch*. The model will not be trained on this data. *validation_data will override validation_split*. validation_data could be: 1) tuple (x_val, y_val) of Numpy arrays or tensors; 2) tuple (x_val, y_val, val_sample_weights) of Numpy arrays; 3) dataset For the first two cases, batch_size must be provided. For the last case, validation_steps must be provided.
    shuffle=True, # Boolean (whether to shuffle the training data before each epoch) or str (for 'batch'). 'batch' is a special option for dealing with the limitations of HDF5 data; it shuffles in batch-sized chunks. Has no effect when steps_per_epoch is not None.
    class_weight=None, # Optional dictionary mapping class indices (integers) to a weight (float) value, used for weighting the loss function (during training only). This can be useful to tell the model to "pay more attention" to samples from an under-represented class.
    sample_weight=None, 
    initial_epoch=0, # Integer. Epoch at which to start training (useful for resuming a previous training run).
    steps_per_epoch=None, # Integer or None. Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch. When training with input tensors such as TensorFlow data tensors, the default None is equal to the number of samples in your dataset divided by the batch size, or 1 if that cannot be determined. If x is a tf.data dataset, and 'steps_per_epoch' is None, the epoch will run until the input dataset is exhausted. This argument is not supported with array inputs.
    validation_steps=None, # Only relevant if validation_data is provided and is a tf.data dataset. Total number of steps (batches of samples) to draw before stopping when performing validation at the end of every epoch. If validation_data is a tf.data dataset and 'validation_steps' is None, validation will run until the validation_data dataset is exhausted.
    validation_freq=1, # Only relevant if validation data is provided. Integer or collections_abc.Container instance (e.g. list, tuple, etc.). If an integer, specifies how many training epochs to run before a new validation run is performed, e.g. validation_freq=2 runs validation every 2 epochs. If a Container, specifies the epochs on which to run validation, e.g. validation_freq=[1, 2, 10] runs validation at the end of the 1st, 2nd, and 10th epochs.
    max_queue_size=10, # Integer. Used for generator or keras.utils.Sequence input only. Maximum size for the generator queue. If unspecified, max_queue_size will default to 10.
    workers=1, # Integer. Used for generator or keras.utils.Sequence input only. Maximum number of processes to spin up when using process-based threading. If unspecified, workers will default to 1. If 0, will execute the generator on the main thread.
    use_multiprocessing=False, # Boolean. Used for generator or keras.utils.Sequence input only. If True, use process-based threading. If unspecified, use_multiprocessing will default to False. Note that because this implementation relies on multiprocessing, you should not pass non-picklable arguments to the generator as they can't be passed easily to children processes.
    **kwargs
)
```
### 3.3 tf.data输入数据
```python
dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
dataset = dataset.batch(32)
dataset = dataset.repeat()

val_dataset = tf.data.Dataset.from_tensor_slices((val_x, val_y))
val_dataset = val_dataset.batch(32)
val_dataset = val_dataset.repeat()

model.fit(dataset, epochs=5, steps_per_epoch=30, validation_data=val_dataset, validation_steps=3)
```
[`tf.data.Dataset`类](https://www.tensorflow.org/api_docs/python/tf/data/Dataset):  
- `from_tensor_slices`: Creates a ***Dataset*** whose elements are slices of the given tensors.  Note that if tensors contains a NumPy array, and eager execution is not enabled, the values will be embedded in the graph as one or more [tf.constant](https://www.tensorflow.org/api_docs/python/tf/constant) operations. For large datasets (> 1 GB), this can waste memory and run into byte limits of graph serialization. If tensors contains one or more large NumPy arrays, consider the alternative described in [this guide](https://tensorflow.org/guide/datasets#consuming_numpy_arrays).
- `batch`: Combines consecutive elements of this dataset into batches.  
    ```python
    batch(
        batch_size, # A tf.int64 scalar tf.Tensor, representing the number of consecutive elements of this dataset to combine in a single batch.
        drop_remainder=False # (Optional.) A tf.bool scalar tf.Tensor, representing whether the last batch should be dropped in the case it has fewer than batch_size elements; the default behavior is not to drop the smaller batch.
        )
    ```
- `repeat`: Repeats this dataset count times.
    ```python
    repeat(
        count=None # (Optional.) A tf.int64 scalar tf.Tensor, representing the number of times the dataset should be repeated. The default behavior (if count is None or -1) is for the dataset be repeated indefinitely.
        )  
    ```
### 3.4 评估与预测
```python
model.evaluate(val_x, val_y, batch_size=32) # 不使用tf.data.Dataset
```
或
```python
# 使用tf.data.Dataset创建数据集之后
test_data = tf.data.Dataset.from_tensor_slices((test_x, test_y))
test_data = test_data.batch(32).repeat()
model.evaluate(test_data, steps=30)
```
[`evaluate`](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential#evaluate) (Returns the loss value & metrics values for the model in test mode): 
```python
evaluate(
    x=None, # Input data. It could be: 1) A Numpy array (or array-like), or a list of arrays (in case the model has multiple inputs); 2) A TensorFlow tensor, or a list of tensors (in case the model has multiple inputs); 3) A dict mapping input names to the corresponding array/tensors, if the model has named inputs; 4) A tf.data dataset; 5) A generator or keras.utils.Sequence instance.
    y=None, # Target data. Like the input data x, it could be either Numpy array(s) or TensorFlow tensor(s). It should be consistent with x (you cannot have Numpy inputs and tensor targets, or inversely). If x is a dataset, generator or keras.utils.Sequence instance, y should not be specified (since targets will be obtained from the iterator/dataset).
    batch_size=None,
    verbose=1, # 0 or 1. Verbosity mode. 0 = silent, 1 = progress bar. 依个人经验，验证时最好设为0，否则出来很多'==='
    sample_weight=None,
    steps=None,
    callbacks=None,
    max_queue_size=10,
    workers=1,
    use_multiprocessing=False
    )
```
## 4. 构建高级模型
### 4.1 函数式API  
keras.Sequential 模型是层的简单堆叠，无法表示任意模型。使用 Keras 函数式 API 可以构建复杂的模型拓扑，例如: 多输入模型， 多输出模型，具有共享层的模型（同一层被调用多次），具有非序列数据流的模型（例如，残差连接）。使用函数式 API 构建的模型具有以下特征：
- 层实例可调用并返回张量。 输入张量和输出张量用于定义 tf.keras.Model 实例。 此模型的训练方式和 Sequential 模型一样。
```python
input_x = keras.Input(shape=(72,))
hidden1 = keras.layers.Dense(32, activation='relu')(input_x)
hidden2 = keras.layers.Dense(16, activation='relu')(hidden1)
pred = keras.layers.Dense(10, activation='softmax')(hidden2)

model = keras.Model(inputs=input_x, outputs=pred) # 
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
            loss=tf.keras.losses.categorical_crossentropy,
            metrics=['accuracy'])
model.fit(train_x, train_y, batch_size=32, epochs=5)
```
以上代码解析:  
- [`keras.Input`](https://www.tensorflow.org/api_docs/python/tf/keras/Input): Used to instantiate a Keras tensor.
    ```python
    tf.keras.Input(
        shape=None, # A shape tuple (integers), not including the batch size. For instance, shape=(32,) indicates that the expected input will be batches of 32-dimensional vectors. Elements of this tuple can be None; 'None' elements represent dimensions where the shape is not known.
        batch_size=None, # optional static batch size (integer).
        name=None,
        dtype=None,
        sparse=False, # A boolean specifying whether the placeholder to be created is sparse. Only one of 'ragged' and 'sparse' can be True.
        tensor=None, # Optional existing tensor to wrap into the Input layer. If set, the layer will not create a placeholder tensor.
        ragged=False, # A boolean specifying whether the placeholder to be created is ragged. Only one of 'ragged' and 'sparse' can be True. In this case, values of 'None' in the 'shape' argument represent ragged dimensions. For more information about RaggedTensors, see https://www.tensorflow.org/guide/ragged_tensors.
        **kwargs
    )
    # return a tensor
    ```
- [`keras.Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model)
### 4.2 模型子类化  
通过对 keras.Model 进行子类化并定义您自己的前向传播来构建完全可自定义的模型。在`__init__`方法中创建层并将它们设置为类实例的属性。在 call 方法中定义前向传播.  
```python
class MyModel(keras.Model):
    def __init__(self, num_classes=10):
        super(MyModel, self).__init__(name='my_model')
        self.num_classes = num_classes
        self.layer1 = layers.Dense(32, activation='relu')
        self.layer2 = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=None): # If you subclass Model, you can optionally have a training argument (boolean) in call, which you can use to specify a different behavior in training and inference:
        h1 = self.layer1(inputs)
        out = self.layer2(h1)
        return out

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)

model = MyModel(num_classes=10)

model.compile(optimizer=keras.optimizers.RMSprop(0.001),
            loss=keras.losses.categorical_crossentropy,
            metrics=['accuracy'])

model.fit(train_x, train_y, batch_size=16, epochs=5)
```
- [`keras.Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model) groups layers into an object with training and inference features.
### 4.3 自定义层  
通过对 keras.layers.Layer 进行子类化并实现以下方法来创建自定义层：
- build：创建层的权重。使用 add_weight 方法添加权重。
- call：定义前向传播。
- compute_output_shape：指定在给定输入形状的情况下如何计算层的输出形状。 或者，可以通过实现 get_config 方法和 from_config 类方法序列化层。 
```python
class MyLayer(keras.layers.Layer):
    def __init__(self, output_dim, **kwargs): #
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs) #

    def build(self, input_shape):
        shape = tf.TensorShape((input_shape[1], self.output_dim))
        self.kernel = self.add_weight(name='kernel1', shape=shape,
                                initializer='uniform', trainable=True)
        super(MyLayer, self).build(input_shape)  # 

    def call(self, inputs):
        return tf.matmul(inputs, self.kernel)

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.output_dim
        return tf.TensorShape(shape)

    def get_config(self):
        base_config = super(MyLayer, self).get_config()
        base_config['output_dim'] = self.output_dim
        return base_config

    @classmethod  # https://stackabuse.com/pythons-classmethod-and-staticmethod-explained/
    def from_config(cls, config):
        return cls(**config)

model = tf.keras.Sequential(
            [
            MyLayer(10),
            layers.Activation('softmax')
            ])


model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
            loss=tf.keras.losses.categorical_crossentropy,
            metrics=['accuracy'])

model.fit(train_x, train_y, batch_size=16, epochs=5)
```
### 4.4 回调
```python
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='./logs')
]
model.fit(train_x, train_y, batch_size=16, epochs=5, callbacks=callbacks, validation_data=(val_x, val_y))
```  

- [`tf.keras.callbacks.EarlyStopping`](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping): 
    ```python
    __init__(
        monitor='val_loss', # Quantity to be monitored.
        min_delta=0, # Minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute change of less than min_delta, will count as no improvement.
        patience=0, # Number of epochs with no improvement after which training will be stopped.
        verbose=0, # verbosity mode.
        mode='auto', # One of {"auto", "min", "max"}. In min mode, training will stop when the quantity monitored has stopped decreasing; in max mode it will stop when the quantity monitored has stopped increasing; in auto mode, the direction is automatically inferred from the name of the monitored quantity.
        baseline=None, # Baseline value for the monitored quantity. Training will stop if the model doesn't show improvement over the baseline.
        restore_best_weights=False # Whether to restore model weights from the epoch with the best value of the monitored quantity. If False, the model weights obtained at the last step of training are used.
    )
    ```
## 5. 保持和恢复
### 5.1 保存权重
```python
model = tf.keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax')])

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy'])

model.save_weights('./weights/model')
model.load_weights('./weights/model')
model.save_weights('./model.h5')
model.load_weights('./model.h5')
```
### 5.2 保存网络结构
```python
# 序列化成json
import json
import pprint
json_str = model.to_json()
pprint.pprint(json.loads(json_str))
fresh_model = tf.keras.models.model_from_json(json_str)
# 保持为yaml格式  #需要提前安装pyyaml
yaml_str = model.to_yaml()
print(yaml_str)
fresh_model = tf.keras.models.model_from_yaml(yaml_str)
```
### 5.3 保存整个模型
```python
model = keras.Sequential([
            keras.layers.Dense(10, activation='softmax', input_shape=(72,)),
            keras.layers.Dense(10, activation='softmax')
            ])
model.compile(optimizer='rmsprop',
            loss='categorical_crossentropy',
            metrics=['accuracy'])
model.fit(train_x, train_y, batch_size=32, epochs=5)
model.save('all_model.h5')
model = keras.models.load_model('all_model.h5')
```
## 6. 将keras用于Estimator  
Estimator API 用于针对分布式环境训练模型。它适用于一些行业使用场景，例如用大型数据集进行分布式训练并导出模型以用于生产.  
```python
model = keras.Sequential([
                    keras.layers.Dense(10,activation='softmax'),
                    keras.layers.Dense(10,activation='softmax')
                    ])

model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

estimator = tf.keras.estimator.model_to_estimator(model)
```

              
            
        


    