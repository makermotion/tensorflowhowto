import tensorflow as tf


# this custom data generator example is made for image data
class CustomDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, batch_size):
        self.batch_size = batch_size 

    def __getitem__(self):
        pass
    
    def __len__(self):
        pass


