import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras import Sequential


def create_vgg16_layers():
    vgg16_conv4 = [
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPool2D(2, 2, padding='same'),

        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPool2D(2, 2, padding='same'),

        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.MaxPool2D(2, 2, padding='same'),

        layers.Conv2D(512, 3, padding='same', activation='relu'),
        layers.Conv2D(512, 3, padding='same', activation='relu'),
        layers.Conv2D(512, 3, padding='same', activation='relu'),
        layers.MaxPool2D(2, 2, padding='same'),

        layers.Conv2D(512, 3, padding='same', activation='relu'),
        layers.Conv2D(512, 3, padding='same', activation='relu'),
        layers.Conv2D(512, 3, padding='same', activation='relu'),
    ]

    x = layers.Input(shape=[None, None, 3])
    out = x
    for layer in vgg16_conv4:
        out = layer(out)

    vgg16_conv4 = tf.keras.Model(x, out)

    vgg16_conv7 = [
        # Difference from original VGG16:
        # 5th maxpool layer has kernel size = 3 and stride = 1
        layers.MaxPool2D(3, 1, padding='same'),
        # atrous conv2d for 6th block
        layers.Conv2D(1024, 3, padding='same',
                      dilation_rate=6, activation='relu'),
        layers.Conv2D(1024, 1, padding='same', activation='relu'),
    ]

    x = layers.Input(shape=[None, None, 512])
    out = x
    for layer in vgg16_conv7:
        out = layer(out)

    vgg16_conv7 = tf.keras.Model(x, out)

    return vgg16_conv4, vgg16_conv7


def create_extra_layers():
    """ Create extra layers
        8th to 11th blocks
    """
    model_1 = Sequential()
    model_1.add(layers.Conv2D(256, 1, activation='relu'))
    model_1.add(layers.Conv2D(512, 3, strides=2, padding='same',
                      activation='relu'))

    model_2 = Sequential()
    model_2.add(layers.Conv2D(128, 1, activation='relu'))
    model_2.add(layers.Conv2D(256, 3, strides=2, padding='same',
                          activation='relu'))

    model_3 = Sequential()
    model_3.add(layers.Conv2D(128, 1, activation='relu'))
    model_3.add(layers.Conv2D(256, 3, activation='relu'))

    model_4 = Sequential()
    model_4.add(layers.Conv2D(128, 1, activation='relu'))
    model_4.add(layers.Conv2D(256, 3, activation='relu'))

    model_5 = Sequential()
    model_5.add(layers.Conv2D(128, 1, activation='relu'))
    model_5.add(layers.Conv2D(256, 4, activation='relu'))

    extra_layers = [
        model_1,
        model_2,
        model_3,
        model_4,
        model_5
    ]

    return extra_layers


def create_conf_head_layers(num_classes):
    """ Create layers for classification
    """
    conf_head_layers = [
        layers.Conv2D(4 * num_classes, kernel_size=3,
                      padding='same'),  # for 4th block
        layers.Conv2D(6 * num_classes, kernel_size=3,
                      padding='same'),  # for 7th block
        layers.Conv2D(6 * num_classes, kernel_size=3,
                      padding='same'),  # for 8th block
        layers.Conv2D(6 * num_classes, kernel_size=3,
                      padding='same'),  # for 9th block
        layers.Conv2D(4 * num_classes, kernel_size=3,
                      padding='same'),  # for 10th block
        layers.Conv2D(4 * num_classes, kernel_size=3,
                      padding='same'),  # for 11th block
        layers.Conv2D(4 * num_classes, kernel_size=1)  # for 12th block
    ]

    return conf_head_layers


def create_loc_head_layers():
    """ Create layers for regression
    """
    loc_head_layers = [
        layers.Conv2D(4 * 4, kernel_size=3, padding='same'),
        layers.Conv2D(6 * 4, kernel_size=3, padding='same'),
        layers.Conv2D(6 * 4, kernel_size=3, padding='same'),
        layers.Conv2D(6 * 4, kernel_size=3, padding='same'),
        layers.Conv2D(4 * 4, kernel_size=3, padding='same'),
        layers.Conv2D(4 * 4, kernel_size=3, padding='same'),
        layers.Conv2D(4 * 4, kernel_size=1)
    ]

    return loc_head_layers

