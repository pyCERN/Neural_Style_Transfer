# Following the instruction of https://www.tensorflow.org/tutorials/generative/style_transfer?hl=ko
# Gatys, Leon A., Alexander S. Ecker, and Matthias Bethge. "A neural algorithm of artistic style." arXiv preprint arXiv:1508.06576 (2015).
import tensorflow as tf
from utils import load_img
from model import StyleContentModel
from train import fit_style_transfer


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), 'Physical GPUs, ', len(logical_gpus), 'Logical GPUs')
    except RuntimeError as e:
        print(e)

if __name__ == '__main__':
    style_path = './images/cafe.jpg'
    content_path = './images/painting.jpg'
    style_image = load_img(style_path)
    content_image = load_img(content_path)

    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']
    content_layers = ['block5_conv2']

    feature_extractor = StyleContentModel(style_layers, content_layers)

    style_targets, _ = feature_extractor(style_image)
    _, content_targets = feature_extractor(content_image)

    image = tf.Variable(content_image) # image to be optimized

    style_weight = 2e-2
    content_weight = 1e-2

    optimizer = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    stylized_image = fit_style_transfer(feature_extractor, style_image, content_image, style_weight, content_weight,
                                        optimizer, epochs=10, steps_per_epoch=100)