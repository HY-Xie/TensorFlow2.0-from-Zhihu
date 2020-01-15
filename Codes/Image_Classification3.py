import tensorflow as tf 
import tensorflow.keras as keras
import numpy as np 
import matplotlib.pyplot as plt 



class MyNetwork(keras.Model):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.layer1 = keras.layers.Dense(128, activation='relu')
        self.layer2 = keras.layers.Dense(10)
    
    def call(self, inputs):
        inputs = tf.reshape(inputs, [-1,784])
        out = self.layer1(inputs)
        logits = self.layer2(out)
        return logits


if __name__ == '__main__':
    keras.backend.clear_session()
    tf.random.set_seed(666)

    BATCH_SIZE = 128
    LR = 1e-3 

    (train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0
    train_labels = train_labels.astype(np.int32)
    test_labels = test_labels.astype(np.int32)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    train_dataset = train_dataset.shuffle(10000).batch(BATCH_SIZE)
    val_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    val_dataset = val_dataset.batch(BATCH_SIZE)

    network = MyNetwork()
    network.build(input_shape=(None, 28,28))
    print(network.summary())

   
    network.compile(optimizer=keras.optimizers.Adam(),
                    loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])
    network.fit(train_dataset,  epochs=5)
    









