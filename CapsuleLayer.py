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

            else:
                raise ValueError("Convolutional non-primary capsule not implemented yet.")

        elif(self.c_type == "FC"):
            print('Fully connected')

        else:
            raise ValueError("Type %s is not supported." % self.c_type)
        

