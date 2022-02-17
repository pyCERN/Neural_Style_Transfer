import tensorflow as tf
import cv2


def load_img(path_to_img):
    '''
    Loads an image and convert it to a tensor with longer side to 512
    :param path_to_img (str): directory path to image
    :return:
    '''
    max_size = 512
    img = cv2.imread(path_to_img)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    longer_size = max(shape)
    scale = max_size / longer_size

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape) # additional dim for batch
    img = img[tf.newaxis, :]

    return img


def tensor_to_img(tensor):
    '''
    Converts a tensor to an image
    :param tensor:
    :return:
    '''
    tensor_shape = tf.shape(tensor)
    num_dim = tf.shape(tensor_shape)
    if num_dim > 3:
        assert tensor_shape[0] == 1
        tensor = tensor[0]

    return tf.keras.preprocessing.image.array_to_img(tensor)