#%%

import importlib
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

import CapsuleLayer


#%%
BATCH_SIZE = 64
EPOCHS = 1

dataset, info = tfds.load('mnist', with_info=True, as_supervised=True)
mnist_train, mnist_test = dataset['train'], dataset['test']

def convert_types(image, label):
  image = tf.cast(image, tf.float32)
  image /= 255
  return image, label

mnist_train = mnist_train.map(convert_types).shuffle(10000).batch(BATCH_SIZE)
mnist_test = mnist_test.map(convert_types).batch(BATCH_SIZE)

#%%
importlib.reload(CapsuleLayer)
#importlib.reload(MaskingLayer)
#importlib.reload(DecodingLayer)

mnist_capsule_struct = [(32, 8, 9), (16, 10)]
mnist_conv2d_struct =  (256, 9, 1, "relu")
mnist_fc_struct = [(512, "relu"),(1024, "relu")]

class CapsNet:

  def __init__(self, batch_size, capsule_struct, conv2d_struct, fc_struct):
    super(CapsNet, self).__init__()
    self.batch_size = batch_size
    self.capsule_struct = capsule_struct
    self.conv2d_struct = conv2d_struct
    self.n_capsules = len(capsule_struct)

    # Initial Convolution
    self.conv1 = tf.keras.layers.Conv2D(filters=self.conv2d_struct[0],kernel_size=self.conv2d_struct[1],
                                        strides=self.conv2d_struct[2],activation=self.conv2d_struct[3])

    # Capsule Layers
    self.capsule_layers = []
    for i in range(self.n_capsules):
      t = capsule_struct[i]
      name = "caps" + str(i)
      c_type = "CONV" if (len(t) == 3) else "FC"
      kernel_size = t[2] if (len(t) == 3) else 0
      vec_length = t[1]
      num_c = t[0]
      isPrimary = (i==0)

      self.capsule_layers.append(CapsuleLayer.CapsuleLayer(num_c, vec_length, kernel_size,
                                            name, c_type, isPrimary))


    # Masking Layer. Assert output shaoe: [batch_size, 16] ([batch_size, 1, 16])
     


    # Decoding layer. Assert output shape: [batch_size, 784] 
  
      

  """
  'Call' propagates the input through the layers. It returns the decoded images and
  the longest vector in the last capsule since these two enteties are used in the loss function.
  """
  def __call__(self, x):
    conv1_out = self.conv1(x)
    caps_out = self.capsule_layers[0](conv1_out)

    for capsLayer in self.capsule_layers[1:]:
      caps_out = capsLayer(caps_out)


    

capsnet = CapsNet(BATCH_SIZE, mnist_capsule_struct, mnist_conv2d_struct, mnist_fc_struct)

@tf.function
def train_step(image, label):
  #with tf.GradientTape() as tape:
  out = capsnet(image)



# Note: Image will have tensor dimension [batch_size, 28, 28, 1]
#       Label will have dimension [batch_size]

for _ in range(EPOCHS):
    for idx,(image, label) in enumerate(mnist_train):
      if(idx < 1):
        train_step(image, label)
        


#%%


#%%
