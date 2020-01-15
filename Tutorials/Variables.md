# 变量Variables
<!-- TOC -->

- [变量Variables](#变量variables)
    - [1. 创建一个变量](#1-创建一个变量)
    - [2. 使用一个变量](#2-使用一个变量)
    - [3. 变量变量](#3-变量变量)

<!-- /TOC -->
## 1. 创建一个变量
```python
import tensorflow as tf
tf.keras.backend.clear_session()  # Destroys the current TF graph and creates a new one.
import tensorflow.keras as keras


my_var = tf.Variable(tf.ones([2,3]))
print(my_var)
try:
    with tf.device("/device:GPU:0"):
        v = tf.Variable(tf.zeros([10, 10]))
        print(v)
except:
    print('no gpu')
```
## 2. 使用一个变量
```python
import tensorflow as tf
tf.keras.backend.clear_session()  # Destroys the current TF graph and creates a new one.
import tensorflow.keras as keras

a = tf.Variable(1.0)
b = (a+2) * 3
print(b) # tf.Tensor(9.0, shape=(), dtype=float32)

a = tf.Variable(1.0)
b = (a.assign_add(2)) * 3
print(b) # tf.Tensor(9.0, shape=(), dtype=float32)
```
## 3. 变量变量
```python
import tensorflow as tf
tf.keras.backend.clear_session()  # Destroys the current TF graph and creates a new one.
import tensorflow.keras as keras

class MyModuleOne(tf.Module):
    def __init__(self):
        self.v0 = tf.Variable(1.0)
        self.vs = [tf.Variable(x) for x in range(10)]

class MyOtherModule(tf.Module):
    def __init__(self):
        self.m = MyModuleOne()
        self.v = tf.Variable(10.0)

m = MyOtherModule()
print(m.variables)
print(len(m.variables)) # 12
```