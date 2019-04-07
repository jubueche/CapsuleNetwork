#%%

import importlib
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

import CapsuleLayer


#%%
BATCH_SIZE = 64
EPOCHS = 1
IMAGE_DIM = 28
R_ITER = 3

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

"""
CapsNet class.
  This is the class that defines the structure of the Capsule network.

  Arguments:
    batch_size: Batch size.
    capsule_struct: The capsule struct is a list of triples or tuples. If the element
                    is a triple, then it is a convolutional capsule with
                    number of filters in the first, vector length in the second and
                    kernel size in the last dimension.
                    If it is a tuple, then we have a fully connected capsule.
                    The tuple specifies the number of capsules and the vector length
                    of the capsule in the respective order.
    con2d_struct:   This is the structure of the initial convolution.
                    It is only one triple, since we have only one initial convolution.
                    One can expand this network to support multiple initial convolutions
                    easily. The shape is (num_filters, kernel_size, stride)
    fc_struct:      The fully connected struct is a list of tuples comprising
                    [shape, activation], where shape is a scalar and activation a string.
                    For example 'sigmoid' or 'relu'.
                    Attention: The last layer should have the correct shape regarding the
                    input shape. For MNIST, the input is 28x28 so the last shape should
                    be 784, since this layer is the reconstructed digit. 
     
"""

mnist_capsule_struct = [(32, 8, 9), (10, 16)]
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

    h_prev = np.floor((IMAGE_DIM - self.conv2d_struct[1])/self.conv2d_struct[2] + 1)

    for i in range(self.n_capsules):
      t = capsule_struct[i]
      name = "caps" + str(i)
      c_type = "CONV" if (len(t) == 3) else "FC"
      kernel_size = t[2] if (len(t) == 3) else 0
      vec_length = t[1]
      num_c = t[0]
      isPrimary = (i==0)
      
      if(c_type == "CONV"):
        if(isPrimary == True):
          prev_vec_length = vec_length
        # Need to compute H/W using IMAGE_DIM and kernel size of conv1
        # This is 20 for MNIST where first conv. is 9x9
        h = np.floor((h_prev-kernel_size)/2+1) # From https://pytorch.org/docs/stable/nn.html
        h_prev = h
        num_per_capsule = int(h**2 * num_c) # e.g. 1152 for MNIST PrimaryCaps layer

      else:
        num_per_capsule = self.capsule_layers[i-1].num_per_capsule # e.g. 1152 for DigiCaps layer

      self.capsule_layers.append(CapsuleLayer.CapsuleLayer(num_c, vec_length, kernel_size,
                                            name, c_type, num_per_capsule, prev_vec_length, R_ITER, isPrimary))

      prev_vec_length = vec_length
    

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
      

    # caps_out [batch_size,10,16]

    


capsnet = CapsNet(BATCH_SIZE, mnist_capsule_struct, mnist_conv2d_struct, mnist_fc_struct)

@tf.function
def train_step(image, label):
  with tf.GradientTape() as tape:
    out = capsnet(image)
    #out = pg()
  

# Note: Image will have tensor dimension [batch_size, 28, 28, 1]
#       Label will have dimension [batch_size]

for _ in range(EPOCHS):
    for idx,(image, label) in enumerate(mnist_train):
      if(idx < 1):
        train_step(image, label)
        


#%%

