import tensorflow as tf

class CustomModel(tf.keras.Model):
    def __init__(self, units=32, activation='relu'):
        super(CustomModel, self).__init__()
        self.h1 = tf.keras.layers.Dense(units=units, activation=activation)
        self.h2 = tf.keras.layers.Dense(units=units, activation=activation)
        self.out = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.h1(inputs)
        x = self.h2(x)
        out = self.out(x)
        return out


