import tensorflow as tf
from PIL import Image
from utils import tensor_to_img, clip_image_values
from loss import calculate_gradient
from IPython.display import display as display_fn
from IPython.display import Image, clear_output


def update_image(feature_extractor, image, style_targets, content_targets,
                 style_weight, content_weight, num_style_layers, num_content_layers, optimizer):
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
    gradients = calculate_gradient(feature_extractor, image, style_targets, content_targets,
                                    style_weight, content_weight, num_style_layers, num_content_layers)
    optimizer.apply_gradients([(gradients, image)])

    image.assign(clip_image_values(image, min_value=0., max_value=255.))


def fit_style_transfer(feature_extractor, style_image, content_image, num_style_layers, num_content_layers,
                       style_weight=1e-2, content_weight=1e-4, optimizer='adam', epochs=1, steps_per_epoch=1):
    images = []

    style_targets, _ = feature_extractor(style_image)
    _, content_targets = feature_extractor(content_image)

    generated_image = tf.cast(content_image, dtype=tf.float32)
    generated_image = tf.Variable(generated_image)

    images.append(content_image)

    for epoch in range(epochs):
        for step in range(steps_per_epoch):
            step += 1

            update_image(feature_extractor, generated_image, style_targets, content_targets,
                         style_weight, content_weight, num_style_layers, num_content_layers, optimizer)
            print('.', end='')

            if (step + 1) % 10 == 0:
                images.append(generated_image)

        print('')
        im = tensor_to_img(generated_image)
        im.save('./images/generated.jpg')

        clear_output(wait=True)
        display_image = tensor_to_img(generated_image)
        display_fn(display_image)
        images.append(generated_image)
        print('Train epoch {}'.format(epoch))

    generated_image = tf.cast(generated_image, tf.uint8)

    return generated_image