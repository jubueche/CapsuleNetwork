import tensorflow as tf


# TODO distinguish between layer that needs routing and not
class CapsuleLayer:

    """ Create layer of multiple capsules.
    By calling CapsuleLayer(args), a new object of type
    CapsuleLayer shall be created and the result of the
    operation of that layer returned. At the same time,
    the object should be accessible by name for later
    investigation.
    This design is similar to the tensorflow design of
    layers.
    Arguments:
        input:        Input from either a capsule layer
                      or the first convolutional layer.
        num_capsules: Number of capsules in the layer.
                      In the paper it was 32 for the
                      PrimaryCaps and 10 for the DigiCaps.
        vec_length:   Corresponds to the depth of each
                      capsule.
        kernel_size:  Kernel size for convolution.
                      In the paper, the PrimaryCaps each
                      perform convolution with a 9x9 kernel
                      on a 20x20 input, resulting in 6x6.
        name:         Name of the capsule layer.
        c_type:       Either "FC" or "CONV"
    In total the output dimension should be
    [batch_size, 32 x 6 x 6, 8]"""

    def __init__(self, num_capsules, vec_length, kernel_size, name, c_type, isPrimary = False):

        # Store the information in fields
        self.num_capsules = num_capsules
        self.vec_length = vec_length
        self.kernel_size = kernel_size
        self.name = name
        self.isPrimary = isPrimary
        self.c_type = c_type

        
        self.convs = []
        for _ in range(self.num_capsules):
            conv = tf.keras.layers.Conv2D(filters=self.vec_length, kernel_size=self.kernel_size,strides=(2,2),activation='relu')
            self.convs.append(conv)

    # TODO Make stride variable
    def __call__(self, input):

    # 1) Compute the output for each pixel with size
    # [batch_size, num_capsules x output_size x output_size x vec_length]
    # 2) Squash the output of each vector using the total input to the capsule.
    # Apparently, this is not the case for the primary caps.
    # Simply check isPrimary field.

        if(self.c_type == "CONV"):
            if(self.isPrimary):
                print('Is primary')

                res = tf.expand_dims(self.convs[0](input), axis=1)
                for conv in self.convs[1:]:
                    tmp = tf.expand_dims(conv(input), axis=1)
                    res = tf.concat([res, tmp], axis=1)

                # MNIST: Shape is now [batch_size, 32, 6, 6, 8]
                # No need to reshape. Squas takes care of that.
                return squash(res)

            else:
                raise ValueError("Convolutional non-primary capsule not implemented yet.")

        elif(self.c_type == "FC"):
            print('Fully connected')
            
            # Obtain u_hat using the weights matrices
            # Do a linear combination of the capsules using the coupling coefs.
            # At the beginning, each beta's of each layers are 0.

            # Squash, linear combination, 

        else:
            raise ValueError("Type %s is not supported." % self.c_type)
        

# TODO Check correctnes
def squash(input):
    # v_j = ||s_j||²/(1+||s_j||²)*s_j/||s_j||

    # Check if we have more than 3 dimensions:
    old_shape = input.get_shape()

    if(tf.size(input) > 3):
        input = tf.reshape(input, shape=[input.get_shape()[0], -1, input.get_shape()[-1]])

    l2_squared = tf.reduce_sum(tf.pow(input,2), axis=2) # [batch_size, 1152, 8] -> [batch_size, 1152]
    res = tf.multiply(tf.divide(l2_squared, tf.multiply(1+l2_squared, tf.sqrt(l2_squared))),input)

    return tf.reshape(res, shape=old_shape)

    