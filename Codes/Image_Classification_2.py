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
    
    
    

    # sample = next(iter(train_dataset))
    # print(sample[0].shape) # (128, 28, 28)
    # print(sample[1].shape) # (128, )
    # logits = network(sample[0])



    # print(logits.shape)

    optimizer = keras.optimizers.Adam()
 
    for epoch in range(5):
        losses = []
        for step, (x,y) in enumerate(train_dataset):  # y's shape: [128,]
            with tf.GradientTape() as tape:
                logits = network(x) # TensorShape([128, 10])
                y_one_hot = tf.one_hot(y, depth=10)  # y_one_hot's shape: [128, 10]
                
                loss = tf.keras.losses.categorical_crossentropy(y_one_hot, logits, from_logits=True)
                # print(loss.shape) # (128,)
                loss = tf.reduce_mean(loss)
                losses.append(float(loss))
                grads = tape.gradient(loss, network.trainable_variables)
                optimizer.apply_gradients(zip(grads, network.trainable_variables))
            
            if step % 100 == 0:
                print('Epoch: {} Step: {} Loss: {:.4f}'.format(epoch+1, step+1, float(loss)))
        print("Epoch Mean Loss: {:.4f}".format(sum(losses)/len(losses)))

        val_losses = []
        for (x, y) in val_dataset:
            logits = network(x)
            y_one_hot = tf.one_hot(y, depth=10)
            val_loss = tf.keras.losses.categorical_crossentropy(y_one_hot, logits, from_logits=True)
            val_loss = tf.reduce_mean(val_loss)
            val_losses.append(float(val_loss))
        print("Val Mean Loss: {:.4f}".format(sum(val_losses)/len(val_losses)))

        

    print(network.summary())







