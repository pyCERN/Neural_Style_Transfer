import tensorflow as tf
from utils import preprocess_image


def gram_matrix(input_tensor):
    '''
    Calculates the Gram matrix, the correlation between the CNN filter responses in image which indicates consistency of style
    :param input_tensor (tensor): feature map from intermediate layers
    :return:
    '''
    gram = tf.linalg.einsum('bijc, bijd -> bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    scaled_gram = gram / num_locations

    return scaled_gram


# Define model
def vgg_model(layer_names):
    '''
    Loads pretrained vgg model and uses intermediate layers as representation of each contents and styles of images
    :param layer_names (list): list of layers for intermediate layers
    :return:
    '''
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(layer).output for layer in layer_names]

    model = tf.keras.Model(inputs=vgg.input, outputs=outputs)

    return model


class StyleContentModel(tf.keras.models.Model):
    '''
    Extracts style and content features
    '''
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()

        self.vgg = vgg_model(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg_trainable = False

    def __call__(self, inputs):
        '''
        :param inputs (tensor):
        :return:
        '''
        preprocessed_input = preprocess_image(inputs)
        outputs = self.vgg(preprocessed_input)
        style_features = outputs[:self.num_style_layers]
        content_features = outputs[self.num_style_layers:]

        gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]

        return gram_style_features, content_features
