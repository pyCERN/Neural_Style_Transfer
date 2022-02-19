import tensorflow as tf
import cv2
from imageio import mimsave
from IPython.display import display as display_fn
from IPython.display import Image


def load_img(path_to_img):
    '''
    Loads an image and convert it to a tensor with longer side to 512
    :param path_to_img (str): directory path to image
    :return:
    '''
    max_size = 512
    img = cv2.imread(path_to_img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)

    shape = tf.shape(img)[:-1]
    shape = tf.cast(shape, tf.float32)
    longer_size = max(shape)
    scale = max_size / longer_size

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape) # additional dim for batch
    img = img[tf.newaxis, :]
    img = tf.image.convert_image_dtype(img, tf.uint8)

    return img


def preprocess_image(image):
    image = tf.cast(image, dtype=tf.float32)
    image = tf.keras.applications.vgg19.preprocess_input(image)

    return image


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


def clip_image_values(image, min_value=0., max_value=255.):
        return tf.clip_by_value(image, clip_value_min=min_value, clip_value_max=max_value)


def display_gif(gif_path):
  '''displays the generated images as an animated gif'''
  with open(gif_path,'rb') as f:
    display_fn(Image(data=f.read(), format='png'))


def create_gif(gif_path, images):
  '''creates animation of generated images'''
  mimsave(gif_path, images, fps=1)
  
  return gif_path