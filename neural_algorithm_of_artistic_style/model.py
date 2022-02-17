import tensorflow as tf


def gram_matrix(input_tensor):
    '''
    Calculates the Gram matrix, the correlation between the CNN filter responses in image which indicates consistency of style
    :param input_tensor (tensor): feature map from intermediate layers
    :return:
    '''
    result = tf.linalg.einsum('bijc, bijd -> bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations


# Define model
def vgg_layers(layer_names):
    '''
    Loads pretrained vgg model and uses intermediate layers as representation of each contents and styles of images
    :param layer_names (list): list of layers for intermediate layers
    :return:
    '''
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(layer).output for layer in layer_names]

    model = tf.keras.Model(inputs=vgg.inputs, outputs=outputs)
    return model


class StyleContentModel(tf.keras.models.Model):
    '''
    Extracts tensors for style and content
    '''
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg_trainable = False

    def __call__(self, inputs):
        '''
        :param inputs (tensor): takes preprocessed inputs
        :return:
        '''
        outputs = self.vgg(inputs)
        style_outputs = outputs[:self.num_style_layers]
        content_outputs = outputs[self.num_style_layers:]

        style_features = [gram_matrix(style_output) for style_output in style_outputs]
        content_features = [gram_matrix(content_output) for content_output in content_outputs]

        return style_features, content_features

