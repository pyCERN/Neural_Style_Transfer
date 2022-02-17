import tensorflow as tf
from utils import tensor_to_img
from loss import calculate_gradient


def update_image(feature_extractor, image, style_targets, content_targets,
                 style_weight, content_weight, optimizer):
    '''
    :param feature_extractor:
    :param image:
    :param style_targets:
    :param content_targets:
    :param style_weight:
    :param content_weight:
    :param optimizer:
    :return:
    '''
    gradients = calculate_gradient(feature_extractor, image, style_targets, content_targets, style_weight, content_weight)
    optimizer.apply_gradient([(gradients, image)])


def fit_style_transfer(feature_extractor, style_image, content_image, style_weight=1e-2, content_weight=1e-4,
                       optimizer='adam', epochs=1, steps_per_epoch=1):
    images = []
    step = 0

    style_targets, _ = feature_extractor(style_image)
    _, content_targets = feature_extractor(content_image)

    generated_image = tf.cast(content_image, dtype=tf.float32)
    generated_image = tf.Variable(generated_image)

    for epoch in range(epochs):
        for step in range(steps_per_epoch):
            step += 1

            update_image(feature_extractor, generated_image, style_targets, content_targets,
                         style_weight, content_weight, optimizer)
            print('.', end='')

        display_image = tensor_to_img(generated_image)
        print('Train step {}'.format(step))

    generated_image = tf.cast(generated_image, tf.uint8)

    return generated_image