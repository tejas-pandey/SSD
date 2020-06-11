import tensorflow as tf


def mobilenet_ssd():
    mobilenet = tf.keras.applications.MobileNet(
        weights=None, include_top=False)

    for layer in mobilenet.layers:
        print(layer.name)

    ssd_layers = ['conv_pw_11_relu',
                  'conv_pw_13_relu',
                  ]
    # mobilenet.summary()


if __name__ == "__main__":

    mobilenet_ssd()
