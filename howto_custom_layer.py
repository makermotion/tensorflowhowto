import tensorflow as tf

class CustomDenseLayer(tf.keras.layers.Layer):
	def __init__(self, units, activation):
		self.super(CustomDenseLayer, self).__init__()
		self.units = units
		self.activation = activation
	
	def build(self, input_shape):
		w_init = tf.random_normal_initializer()
		b_init = tf.zeros_initializer()
		self.w = tf.Variable(name='kernel',
							 initial_value=w_init(shape=(input_shape[-1], self.units), dtype=tf.float32),
							 trainable=True,
							)
		self.b = tf.Variable(name='bias',
							 initial_value=b_init(shape=(self.units,), dtype=tf.float32),
							 trainable=True,
							)
							 
	
	def call(self, inputs):
		return self.activtion(tf.matmul(inputs, self.w) + self.b)
		
