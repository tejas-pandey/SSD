
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Layer

import sys
sys.path.append("/home/sean/work/tf_files/tensorflow/models/research/")
from object_detection.anchor_generators.multiple_grid_anchor_generator import create_ssd_anchors



def LeNet():
    
    # Number of classes.
    n_classes = 2
    # Number of layers being used for ssd.
    n_ssd_layers = 3

    # Generating boxes for ssd layers.
    n_boxes = []
    list_of_aspect_ratios = [1, 2, 3]
    aspect_ratios = [list_of_aspect_ratios] * n_ssd_layers
    scales = [1, 2, 3]
    n_boxes = len(aspect_ratios)
    n_boxes = [n_boxes] * n_ssd_layers
       

    # BASE MODEL.
    model = tf.keras.models.Model()

    conv1 = Conv2D(16, 5, padding='same', activation='relu', input_shape=(None, None, 3))
    conv1 = MaxPool2D()(conv1)
    
    conv2 = Conv2D(32, 5, padding='same', activation='relu')(conv1)
    conv2 = MaxPool2D()(conv2)
    
    conv3 = Conv2D(32, 5, padding='same', activation='relu')(conv2)
    conv3 = MaxPool2D()(conv3)

    conv4 = Conv2D(64, 5, padding='same', activation='relu')(conv3)
    conv4 = MaxPool2D()(conv4)

    conv5 = Conv2D(64, 5, padding='same', activation='relu')(conv4)
    conv5 = MaxPool2D()(conv5)

    conv6 = Conv2D(128, 5, padding='same', activation='relu')(conv5)
    conv6 = MaxPool2D()(conv6)

    # Build the convolutional predictor layers on top of conv layers 4, 5, 6.
    # We build two predictor layers on top of each of these layers: One for class prediction (classification), one for box coordinate prediction (localization)
    # We predict `n_classes` confidence values for each box, hence the `classes` predictors have depth `n_boxes * n_classes`
    # We predict 4 box coordinates for each box, hence the `boxes` predictors have depth `n_boxes * 4`
    # Output shape of `classes`: `(batch, height, width, n_boxes * n_classes)`

    # Classification
    classes4 = Conv2D(n_boxes[0] * n_classes, 3, padding='same')(conv4)
    classes5 = Conv2D(n_boxes[1] * n_classes, 3, padding='same')(conv5)
    classes6 = Conv2D(n_boxes[2] * n_classes, 3, padding='same')(conv6)

    # Bounding boxes.Conv2D
    box4 = Conv2D(n_boxes[0] * 4, 3, padding='same')(conv4)
    box5 = Conv2D(n_boxes[1] * 4, 3, padding='same')(conv5)
    box6 = Conv2D(n_boxes[2] * 4, 3, padding='same')(conv6)

    # Anchors or network generated groundtruth.
    anchors4 = anchor(aspect_ratios=(1.0, 2.0 ,3.0), scales)(boxes4)
    anchors5 = anchor(aspect_ratios=(1.0, 2.0 ,3.0), scales)(boxes5)
    anchors6 = anchor(aspect_ratios=(1.0, 2.0 ,3.0), scales)(boxes6)

    classes4_flattened = tf.reshape(classes4, [-1, n_classes])(classes4)
    classes5_flattened = tf.reshape(classes4, [-1, n_classes])(classes5)
    classes6_flattened = tf.reshape(classes4, [-1, n_classes])(classes6)

    
    