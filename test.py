import os
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from PIL import Image
from PIL import ImageDraw
from PIL import ImageColor

# Have to add background class ourselves.
labels = ["background", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", " ", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", " ", "backpack", "umbrella", " ", " ", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
          "bottle", " ", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", " ", "dining table", " ", " ", "toilet", " ", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", " ", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
image_path = "val2017/000000252219.jpg"
bbox_list = [[326.28, 174.56, 71.24, 197.25],
             [9.79, 167.06, 121.94, 226.45],
             [510.44, 171.27, 123.66, 215.76],
             [560.73, 90.25, 79.27, 67.32],
             [46.01, 211.11, 33.55, 51.85],
             [345.13, 226.41, 11.06, 22.14],
             [337.06, 44.11, 61.36, 57.17]
             ]
labels_list = [[1], [1], [1], [28], [31], [47], [10]]


def draw_bounding_box(image, bbox, category_id, labels):
    draw = ImageDraw.Draw(image)
    colors = list(ImageColor.colormap.values())

    x, y, w, h = bbox

    left = x
    top = y
    right = x + w
    bot = y + h

    color = colors[np.random.randint(0, len(colors))]

    draw.line([(left, top), (left, bot),
               (right, bot), (right, top),
               (left, top)],
              width=np.random.randint(1, 5),
              fill=color)

    draw.text((left, top), labels[category_id - 1], fill=color)


def draw_my_bounding_box(image, bbox, category_id, labels):
    draw = ImageDraw.Draw(image)
    colors = list(ImageColor.colormap.values())

    x, y, w, h = bbox

    left = x
    top = y
    right = w
    bot = h

    color = colors[np.random.randint(0, len(colors))]

    draw.line([(left, top), (left, bot),
               (right, bot), (right, top),
               (left, top)],
              width=np.random.randint(1, 5),
              fill=color)

    draw.text((left, top), labels[category_id - 1], fill=color)


def intersection_area(boxes1, boxes2):
    m = boxes1.shape[0]
    n = boxes2.shape[0]

    xmin = 0
    ymin = 1
    xmax = 2
    ymax = 3

    # https://stackoverflow.com/questions/27152904/calculate-overlapped-area-between-two-rectangles
    # https://drive.google.com/file/d/1VXzinaV-ifFIKeH2lwErLKodQzJSk_zR/view?usp=sharing

    min_xy = np.maximum(np.tile(np.expand_dims(boxes1[:, [xmin, ymin]], axis=1), (1, n, 1)),
                        np.tile(np.expand_dims(boxes2[:, [xmin, ymin]], axis=0), (m, 1, 1)))


    max_xy = np.minimum(np.tile(np.expand_dims(boxes1[:, [xmax, ymax]], axis=1), (1, n, 1)),
                        np.tile(np.expand_dims(boxes2[:, [xmax, ymax]], axis=0), (m, 1, 1)))

    side_lengths = np.maximum(0, max_xy - min_xy)

    return side_lengths[:, :, 0] * side_lengths[:, :, 1]


def iou(boxes1, boxes2):
    xmin = 0
    ymin = 1
    xmax = 2
    ymax = 3

    # https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/

    intersection = intersection_area(boxes1, boxes2)

    m = boxes1.shape[0]
    n = boxes2.shape[0]

    boxes1_areas = np.tile(np.expand_dims(
        (boxes1[:, xmax] - boxes1[:, xmin]) * (boxes1[:, ymax] - boxes1[:, ymin]), 1), (1, n))

    boxes2_areas = np.tile(np.expand_dims(
        (boxes2[:, xmax] - boxes2[:, xmin]) * (boxes2[:, ymax] - boxes2[:, ymin]), 0), (m, 1))

    union_areas = boxes1_areas + boxes2_areas - intersection
    return intersection / union_areas


def convert_coordinates(tensor, start_index, conversion, border_pixels='half'):
    if border_pixels == 'half':
        d = 0
    elif border_pixels == 'include':
        d = 1
    elif border_pixels == 'exclude':
        d = -1

    ind = start_index
    tensor1 = np.copy(tensor).astype(np.float)
    if conversion == 'minmax2centroids':
        tensor1[..., ind] = (tensor[..., ind] +
                             tensor[..., ind+1]) / 2.0  # Set cx
        tensor1[..., ind+1] = (tensor[..., ind+2] +
                               tensor[..., ind+3]) / 2.0  # Set cy
        tensor1[..., ind+2] = tensor[..., ind+1] - \
            tensor[..., ind] + d  # Set w
        tensor1[..., ind+3] = tensor[..., ind+3] - \
            tensor[..., ind+2] + d  # Set h
    elif conversion == 'centroids2minmax':
        tensor1[..., ind] = tensor[..., ind] - \
            tensor[..., ind+2] / 2.0  # Set xmin
        tensor1[..., ind+1] = tensor[..., ind] + \
            tensor[..., ind+2] / 2.0  # Set xmax
        tensor1[..., ind+2] = tensor[..., ind+1] - \
            tensor[..., ind+3] / 2.0  # Set ymin
        tensor1[..., ind+3] = tensor[..., ind+1] + \
            tensor[..., ind+3] / 2.0  # Set ymax
    elif conversion == 'corners2centroids':
        tensor1[..., ind] = (tensor[..., ind] +
                             tensor[..., ind+2]) / 2.0  # Set cx
        tensor1[..., ind+1] = (tensor[..., ind+1] +
                               tensor[..., ind+3]) / 2.0  # Set cy
        tensor1[..., ind+2] = tensor[..., ind+2] - \
            tensor[..., ind] + d  # Set w
        tensor1[..., ind+3] = tensor[..., ind+3] - \
            tensor[..., ind+1] + d  # Set h
    elif conversion == 'centroids2corners':
        tensor1[..., ind] = tensor[..., ind] - \
            tensor[..., ind+2] / 2.0  # Set xmin
        tensor1[..., ind+1] = tensor[..., ind+1] - \
            tensor[..., ind+3] / 2.0  # Set ymin
        tensor1[..., ind+2] = tensor[..., ind] + \
            tensor[..., ind+2] / 2.0  # Set xmax
        tensor1[..., ind+3] = tensor[..., ind+1] + \
            tensor[..., ind+3] / 2.0  # Set ymax
    elif (conversion == 'minmax2corners') or (conversion == 'corners2minmax'):
        tensor1[..., ind+1] = tensor[..., ind+2]
        tensor1[..., ind+2] = tensor[..., ind+1]
    else:
        raise ValueError("Wtf unexpected conversion string.")

    return tensor1


"""
Don't need. Convert keras ops to tensorflow.
def expanded_shape(orig_shape, start_dim, num_dims):
    start_dim = tf.expand_dims(start_dim, 0)  # scalar to rank-1
    before = tf.slice(orig_shape, [0], start_dim)
    add_shape = tf.ones(tf.reshape(num_dims, [1]), dtype=tf.int32)
    after = tf.slice(orig_shape, start_dim, [-1])
    new_shape = tf.concat([before, add_shape, after], 0)
    return new_shape


def meshgrid(x, y):
    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)
    x_exp_shape = expanded_shape(tf.shape(x), 0, tf.rank(y))
    y_exp_shape = expanded_shape(tf.shape(y), tf.rank(y), tf.rank(x))

    xgrid = tf.tile(tf.reshape(x, x_exp_shape), y_exp_shape)
    ygrid = tf.tile(tf.reshape(y, y_exp_shape), x_exp_shape)
    new_shape = y.get_shape().concatenate(x.get_shape())
    xgrid.set_shape(new_shape)
    ygrid.set_shape(new_shape)
    return xgrid, ygrid


def _center_size_bbox_to_corners_bbox(centers, sizes):
    return tf.concat([centers - .5 * sizes, centers + .5 * sizes], 1)
"""


def match_bipartite_greedy(similarities):

    num_gt_boxes = similarities.shape[0]
    all_gt_indices = list(range(num_gt_boxes))

    matches = np.zeros(num_gt_boxes)

    for _ in range(num_gt_boxes):
        anchor_indices = np.argmax(similarities, axis=1)
        overlaps = similarities[all_gt_indices, anchor_indices]
        ground_truth_index = np.argmax(overlaps)
        anchor_index = anchor_indices[ground_truth_index]
        matches[ground_truth_index] = anchor_index
        similarities[ground_truth_index] = 0
        similarities[:, anchor_index] = 0

    return matches


def generate_anchor_boxes_for_layer(img_size, ft_map_size,
                                    aspect_ratios, two_boxes_for_ar1,
                                    this_scale, next_scale,
                                    this_steps=None, this_offsets=None,
                                    clip_boxes=True, normalize_coords=True):

    feature_width = ft_map_size[1]
    feature_height = ft_map_size[0]

    img_width = img_size[1]
    img_height = img_size[0]

    size = min(img_width, img_height)

    if two_boxes_for_ar1:
        n_boxes = len(aspect_ratios) + 1
    else:
        n_boxes = len(aspect_ratios)

    boxes = []
    for ar in aspect_ratios:
        if ar == 1:
            box_height = box_width = this_scale * size
            boxes.append(np.stack([box_width, box_height]))

            if two_boxes_for_ar1:
                box_height = box_width = np.sqrt(
                    this_scale * next_scale) * size
                boxes.append(np.stack([box_width, box_height]))

        else:
            box_height = this_scale * size / np.sqrt(ar)
            box_width = this_scale * size * np.sqrt(ar)
            boxes.append(np.stack([box_width, box_height]))

    if this_steps is None:
        step_height = img_height / feature_height
        step_width = img_width / feature_width

    else:
        raise NotImplementedError

    if this_offsets is None:
        offset_height = 0.5
        offset_width = 0.5

    else:
        raise NotImplementedError

    start = offset_height * step_height
    end = (offset_height + feature_height - 1) * step_height
    cy = np.linspace(start, end, int(feature_height))

    start = offset_width * step_width
    end = (offset_width + feature_width - 1) * step_width
    cx = np.linspace(start, end, int(feature_width))

    cx_grid, cy_grid = np.meshgrid(cx, cy)
    cx_grid = np.expand_dims(cx_grid, -1)
    cy_grid = np.expand_dims(cy_grid, -1)

    boxes = np.asarray(boxes)
    boxes_tensor = np.zeros(
        (int(feature_height), int(feature_width), n_boxes, 4))

    boxes_tensor[:, :, :, 0] = np.tile(cx_grid, (1, 1, n_boxes))
    boxes_tensor[:, :, :, 1] = np.tile(cy_grid, (1, 1, n_boxes))
    boxes_tensor[:, :, :, 2] = boxes[:, 0]
    boxes_tensor[:, :, :, 3] = boxes[:, 1]

    boxes_tensor = convert_coordinates(
        boxes_tensor, start_index=0, conversion='centroids2corners')

    """

    width_grid, x_center_grid = meshgrid(boxes[:, 0], cx_grid)
    height_grid, y_center_grid = meshgrid(boxes[:, 1], cy_grid)

    # blah = tf.stack([x_center_grid, y_center_grid], axis=3)

    bbox_centers = tf.stack([y_center_grid, x_center_grid], axis=3)
    bbox_sizes = tf.stack([height_grid, width_grid], axis=3)
    bbox_centers = tf.reshape(bbox_centers, [-1, 2])
    bbox_sizes = tf.reshape(bbox_sizes, [-1, 2])
    bbox_corners = _center_size_bbox_to_corners_bbox(bbox_centers, bbox_sizes)
    """

    if clip_boxes:
        x_coords = boxes_tensor[:, :, :, [0, 2]]
        x_coords[x_coords >= img_width] = img_width - 1
        x_coords[x_coords < 0] = 0
        boxes_tensor[:, :, :, [0, 2]] = x_coords

        y_coords = boxes_tensor[:, :, :, [1, 3]]
        y_coords[y_coords >= img_height] = img_height - 1
        y_coords[y_coords < 0] = 0
        boxes_tensor[:, :, :, [1, 3]] = y_coords

    if normalize_coords:
        boxes_tensor[:, :, :, [0, 2]] /= img_width
        boxes_tensor[:, :, :, [1, 3]] /= img_height

    return boxes_tensor


if __name__ == "__main__":

    img = Image.open(image_path)

    predictor_sizes = np.asarray([[19, 19],
                                  [10, 10],
                                  [5, 5]])

    aspect_ratios = [0.5, 1.0, 2.0]
    variances = [0.1, 0.1, 0.2, 0.2]

    two_boxes_for_ar1 = True

    n_classes = len(labels)
    print("Number of classes :\t", n_classes)
    gt_labels = np.append(labels_list, bbox_list, axis=1)
    gt_labels = np.expand_dims(gt_labels, 0)

    if two_boxes_for_ar1:
        n_boxes = len(aspect_ratios) + 1
    else:
        n_boxes = len(aspect_ratios)

    normalize_coords = True
    neg_iou_limit = 0.3

    aspect_ratios = [aspect_ratios] * len(predictor_sizes)

    min_scale = 0.2
    max_scale = 0.95
    scales = np.linspace(min_scale, max_scale, len(predictor_sizes)+1)

    class_id = 0
    xmin = 1
    ymin = 2
    xmax = 3
    ymax = 4

    img_size = img.size

    boxes_list = []
    for i in range(len(predictor_sizes)):
        boxes_list.append(generate_anchor_boxes_for_layer(
            img_size, predictor_sizes[i],
            aspect_ratios[i], two_boxes_for_ar1,
            this_scale=scales[i], next_scale=scales[i+1],
            clip_boxes=False, normalize_coords=False
        ))

        print("Layer {} boxes: {}\t".format(i, boxes_list[-1].shape))
        for j in range(boxes_list[-1].shape[-1]):
            box = boxes_list[-1][:, :, j]
            box = np.reshape(box, (-1, 4))
            for b in box:
                # tmp_img = img.copy()
                draw_my_bounding_box(img, b, 0, labels)
                # plt.imshow(tmp_img)
                # plt.pause(1e-10)
                # plt.clf()
                
            # print(box)
#        plt.imshow(img)
#        plt.show()


    batch_size = len(gt_labels)

    boxes_batch = []
    for boxes in boxes_list:
        boxes = np.expand_dims(boxes, 0)
        boxes = np.tile(boxes, (batch_size, 1, 1, 1, 1))
        boxes = np.reshape(boxes, (batch_size, -1, 4))
        boxes_batch.append(boxes)

    boxes_tensor = np.concatenate(boxes_batch, 1)
    classes_tensor = np.zeros((batch_size, boxes_tensor.shape[1], n_classes))

    variances_tensor = np.zeros_like(boxes_tensor)
    variances_tensor += variances

    y_encoded = np.concatenate(
        (classes_tensor, boxes_tensor, boxes_tensor, variances_tensor), 2)


    y_encoded[:, :, 0] = 1
    temp_n_boxes = y_encoded.shape[1]

    class_vectors = np.eye(n_classes)

    for i in range(batch_size):
        if gt_labels[0].size == 0:
            continue
        labels = gt_labels[i].astype(np.float)
        labels[:, xmax] += labels[:, xmin]
        labels[:, ymax] += labels[:, ymin]

        if np.any(labels[:, [xmax]] - labels[:, [xmin]] <= 0) or np.any(labels[:, [ymax]] - labels[:, [ymin]] <= 0):
            raise ValueError("Eh")

        if normalize_coords:
            labels[:, [ymin, ymax]] /= img_size[1]
            labels[:, [xmin, xmax]] /= img_size[0]

        classes_one_hot = class_vectors[labels[:, class_id].astype(int)]
        labels_one_hot = np.concatenate(
            [classes_one_hot, labels[:, [xmin, ymin, xmax, ymax]]], -1)


        similarities = iou(
            labels[:, [xmin, ymin, xmax, ymax]], y_encoded[i, :, -12: -8])

        bipartite_matches = match_bipartite_greedy(similarities)
        bipartite_matches = bipartite_matches.astype(int)

        y_encoded[i, bipartite_matches, :-8] = labels_one_hot
        similarities[:, bipartite_matches] = 0

        max_background_similarities = np.amax(similarities, 0)
        neutral_boxes = np.nonzero(
            max_background_similarities >= neg_iou_limit)[0]
        y_encoded[i, neutral_boxes, 0] = 0

        y_encoded[:, :, -12:-8] -= y_encoded[:, :, -8:-4]
        # (xmin(gt) - xmin(anchor)) / w(anchor), (xmax(gt) - xmax(anchor)) / w(anchor)
        y_encoded[:, :, [-12, -10]
                  ] /= np.expand_dims(y_encoded[:, :, -6] - y_encoded[:, :, -8], axis=-1)
        # (ymin(gt) - ymin(anchor)) / h(anchor), (ymax(gt) - ymax(anchor)) / h(anchor)
        y_encoded[:, :, [-11, -9]] /= np.expand_dims(
            y_encoded[:, :, -5] - y_encoded[:, :, -7], axis=-1)
        # (gt - anchor) / size(anchor) / variance for all four coordinates, where 'size' refers to w and h respectively
        # y_encoded[:, :, -12:-8] /= y_encoded[:, :, -4:]
