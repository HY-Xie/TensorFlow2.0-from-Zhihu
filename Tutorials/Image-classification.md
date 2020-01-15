# 图像分类
[原代码](../Codes/Image_Classification_1.py)
<!-- TOC -->

- [图像分类](#图像分类)
    - [1. 获取Fashion MNIST数据集](#1-获取fashion-mnist数据集)
    - [2. 探索数据](#2-探索数据)
    - [3. 处理数据](#3-处理数据)
    - [4. 构造网络](#4-构造网络)
    - [5. 训练与验证](#5-训练与验证)
    - [6. 预测及可视化](#6-预测及可视化)

<!-- /TOC -->
## 1. 获取Fashion MNIST数据集
本指南使用Fashion MNIST数据集，该数据集包含10个类别中的70,000个灰度图像。 图像显示了低分辨率（28 x 28像素）的单件服装。Fashion MNIST旨在替代经典的MNIST数据集，通常用作计算机视觉机器学习计划的“Hello，World”。我们将使用60,000张图像来训练网络和10,000张图像，以评估网络学习图像分类的准确程度。
```python
(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
```
图像是28x28 NumPy数组，像素值介于0到255之间。标签是一个整数数组，范围从0到9. 这些对应于图像所代表的服装类别：  
`LabelClass0T-shirt/top1Trouser2Pullover3Dress4Coat5Sandal6Shirt7Sneaker8Bag9Ankle boot`  
每个图像都映射到一个标签。 由于类名不包含在数据集中，因此将它们存储在此处以便在绘制图像时使用：  
```python
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
```
## 2. 探索数据
我们在训练模型之前探索数据集的格式。 以下显示训练集中有60,000个图像，每个图像表示为28 x 28像素：  
```python
print(train_images.shape) # (60000, 28, 28)
print(train_labels.shape) # (60000,)
print(test_images.shape)  # (10000, 28, 28)
print(test_labels.shape)  # (10000,)
```
## 3. 处理数据
展示图片：
```python
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
```
```python
train_images = train_images / 255.0
test_images = test_images / 255.0
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
```
## 4. 构造网络
```python
model = keras.Sequential(
    [
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
    ])
model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
```
## 5. 训练与验证
```python
model.fit(train_images, train_labels, epochs=5)
```

## 6. 预测及可视化
```python
predictions = model.predict(test_images)
print(predictions[0])
print(np.argmax(predictions[0]))
print(test_labels[0])

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                    100*np.max(predictions_array),
                                    class_names[true_label]),
                                    color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1]) 
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()


# 可视化结果
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions, test_labels)
plt.show()


img = test_images[0]
img = (np.expand_dims(img,0))
print(img.shape)
predictions_single = model.predict(img)
print(predictions_single)
plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
```
