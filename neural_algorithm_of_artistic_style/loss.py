import tensorflow as tf


def get_style_loss(features, targets):
    '''
    :param features (tensor): Gram matrix features
    :param targets (tensor):
    :return:
    '''
    style_loss = tf.reduce_mean(tf.square(features - targets))

    return style_loss


def get_content_loss(features, targets):
    '''
    :param features:
    :param targets:
    :return:
    '''
    content_loss = 0.5 * tf.reduce_sum(tf.square(features -targets))

    return content_loss


def get_style_content_loss(style_features, content_features, style_targets, content_targets,
                           style_weight, content_weight, num_style_layers, num_content_layers):
    '''
    :param style_outputs:
    :param content_outputs:
    :return:
    '''
    style_loss = tf.add_n([get_style_loss(style_feature, style_target)
                           for style_feature, style_target in zip(style_features, style_targets)])
    content_loss = tf.add_n([get_content_loss(content_feature, content_target)
                             for content_feature, content_target in zip(content_features, content_targets)])
    style_loss *= style_weight / num_style_layers
    content_loss *= content_weight / num_content_layers
    total_loss = style_loss + content_loss

    return total_loss


def calculate_gradient(feature_extractor, image, style_targets, content_targets,
                       style_weight, content_weight, num_style_layers, num_content_layers):
    '''
    Calcuates the gradients of the loss w.r.t. the generated image
    :param image:
    :param style_targets:
    :return:
    '''
    with tf.GradientTape() as tape:
        style_features, content_features = feature_extractor(image)

        loss = get_style_content_loss(style_features, content_features, style_targets, content_targets,
                                      style_weight, content_weight, num_style_layers, num_content_layers)

    gradients = tape.gradient(loss, image)

    return gradients