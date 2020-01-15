# Miscellaneous
- [`tf.keras.backend.clear_session()`](https://www.tensorflow.org/api_docs/python/tf/keras/backend/clear_session)
    - Destroys the current TF graph and creates a new one.
- [`tf.random_normal_initializer()`](https://www.tensorflow.org/api_docs/python/tf/random_normal_initializer)
    - Initializer that generates tensors with a normal distribution.
    - Example:
        ```python
        w_init = tf.random_normal_initializer()
        self.weight = tf.Variable(initial_value=w_init(
            shape=(input_dim, unit), dtype=tf.float32), trainable=True)
        ```
- [`tf.zeros_initializer()`](https://www.tensorflow.org/api_docs/python/tf/zeros_initializer)
    - Initializer that generates tensors initialized to 0.
    - Example:
        ```python
        b_init = tf.zeros_initializer()
        self.bias = tf.Variable(initial_value=b_init(
            shape=(unit,), dtype=tf.float32), trainable=True)
        ```
- [`tf.random.set_seed(seed)`](https://www.tensorflow.org/api_docs/python/tf/random/set_seed): Sets the graph-level random seed.