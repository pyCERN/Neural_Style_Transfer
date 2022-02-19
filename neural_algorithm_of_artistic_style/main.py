# Following the instruction of https://www.tensorflow.org/tutorials/generative/style_transfer?hl=ko
# Gatys, Leon A., Alexander S. Ecker, and Matthias Bethge. "A neural algorithm of artistic style." arXiv preprint arXiv:1508.06576 (2015).
import numpy as np
import tensorflow as tf
from keras import backend as K
from utils import load_img
from model import StyleContentModel
from train import fit_style_transfer


if __name__ == '__main__':
    style_path = './images/painting.jpg'
    content_path = './images/cafe.jpg'

    style_image = load_img(style_path)
    content_image = load_img(content_path)

    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']
    content_layers = ['block5_conv2']
    num_style_layers, num_content_layers = len(style_layers), len(content_layers)

    feature_extractor = StyleContentModel(style_layers, content_layers)

    image = tf.Variable(content_image) # image to be optimized

    style_weight = 2e-2
    content_weight = 1e-2

    optimizer = tf.optimizers.Adam(
        tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=20., decay_steps=100, decay_rate=0.5
        )
    )

    stylized_image, display_images = fit_style_transfer(feature_extractor, style_image, content_image, num_style_layers, num_content_layers,
                                        style_weight, content_weight, optimizer, epochs=10, steps_per_epoch=100)

    GIF_PATH = './image/style_transfer.gif'
    gif_images = [np.squeeze(image.numpy().astype(np.uint8), axis=0) for image in display_images]
    gif_path = create_gif(GIF_PATH, gif_images)
    display_gif(gif_path)
