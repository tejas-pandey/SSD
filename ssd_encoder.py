import tensorflow as tf


def convert_coordinates(tensor, start_index, conversion, border_pixels='half'):
    """
    Support configs:
    1. (xmin, xmax, ymin, ymax) - minmax.
    2. (xmin, ymin, xmax, ymax) - corners.
    3. (cx, cy, w, h) - centroids.

    Arguments:
        tensor {[type]} -- [description]
        start_index {[type]} -- [description]
        conversion {[type]} -- [description]

    Keyword Arguments:
        border_pixels {str} -- [description] (default: {'half'})
    """

    if border_pixels == 'half':
        d = 0
    elif border_pixels == 'include':
        d = 1
    elif border_pixels == 'exclude':
        d = -1

    ind = start_index
    tensor1 = tf.identity(tensor)
    if conversion == 'minmax2centroids':
        tensor1[..., ind] = (tensor[..., ind] + tensor[..., ind+1]) / 2.0 # Set cx
        tensor1[..., ind+1] = (tensor[..., ind+2] + tensor[..., ind+3]) / 2.0 # Set cy
        tensor1[..., ind+2] = tensor[..., ind+1] - tensor[..., ind] + d # Set w
        tensor1[..., ind+3] = tensor[..., ind+3] - tensor[..., ind+2] + d # Set h
    elif conversion == 'centroids2minmax':
        tensor1[..., ind] = tensor[..., ind] - tensor[..., ind+2] / 2.0 # Set xmin
        tensor1[..., ind+1] = tensor[..., ind] + tensor[..., ind+2] / 2.0 # Set xmax
        tensor1[..., ind+2] = tensor[..., ind+1] - tensor[..., ind+3] / 2.0 # Set ymin
        tensor1[..., ind+3] = tensor[..., ind+1] + tensor[..., ind+3] / 2.0 # Set ymax
    elif conversion == 'corners2centroids':
        tensor1[..., ind] = (tensor[..., ind] + tensor[..., ind+2]) / 2.0 # Set cx
        tensor1[..., ind+1] = (tensor[..., ind+1] + tensor[..., ind+3]) / 2.0 # Set cy
        tensor1[..., ind+2] = tensor[..., ind+2] - tensor[..., ind] + d # Set w
        tensor1[..., ind+3] = tensor[..., ind+3] - tensor[..., ind+1] + d # Set h
    elif conversion == 'centroids2corners':
        tensor1[..., ind] = tensor[..., ind] - tensor[..., ind+2] / 2.0 # Set xmin
        tensor1[..., ind+1] = tensor[..., ind+1] - tensor[..., ind+3] / 2.0 # Set ymin
        tensor1[..., ind+2] = tensor[..., ind] + tensor[..., ind+2] / 2.0 # Set xmax
        tensor1[..., ind+3] = tensor[..., ind+1] + tensor[..., ind+3] / 2.0 # Set ymax
    elif (conversion == 'minmax2corners') or (conversion == 'corners2minmax'):
        tensor1[..., ind+1] = tensor[..., ind+2]
        tensor1[..., ind+2] = tensor[..., ind+1]
    else:
        raise ValueError("Wtf unexpected conversion string.")

    return tensor1


<<<<<<< HEAD
def intersection_area_(boxes1, boxes2, coords='corners', mode='outer_product', border_pixels='half'):
    m = boxes1.shape[0]
    n = boxes2.shape[1]

    xmin = 0
    ymin = 1
    xmax = 2
    ymax = 3

    min_xy = tf.maximum(tf.tile(tf.expand_dims(boxes1[:, [xmin, ymin]], axis=1), (1, n, 1)),
                tf.tile(tf.expand_dims(boxes2[:, [xmin, ymin]], axis=0), (m, 1, 1)))

    max_xy = tf.minimum(tf.tile(tf.expand_dims(boxes1[:, [xmax, ymax]], axis=1), (1, n, 1)),
                tf.tile(tf.expand_dims(boxes2[:, [xmax, ymax]], axis=0), (m, 1, 1)))

    side_lengths = tf.maximum(0, max_xy - min_xy)

    return side_lengths[:, :, 0] * side_lengths[:, :, 1]


def iou(boxes1, boxes2, coords='centroids', mode='outer_product', border_pixels='half'):
=======
def intersecion_area(boxes1, boxes2):
    m = boxes1.shape[0]
    n = boxes2.shape[0]

    xmin = 0
    ymin = 1
    xmax  =2
    ymax = 3

    min_xy = 

def iou(boxes1, boxes2, coords='centroids', mode='outer_product'):
>>>>>>> 9b4197c21069c222fb988fbdf685e6927689f684

    if coords == 'centroids':
        boxes1 = convert_coordinates(boxes1, start_index=0, conversion='centroids2corners')
        boxes2 = convert_coordinates(boxes2, start_index=0, conversion='centroids2corners')
        coords = 'corners'
<<<<<<< HEAD

    intersection_areas = intersection_area_(boxes1, boxes2)

    m = boxes1.shape[0]
    n = boxes2.shape[0]

    xmin = 0
    ymin = 1
    xmax = 2
    ymax = 3

    boxes1_areas = tf.tile(tf.expand_dims((boxes1[:, xmax] - boxes1[:, xmin]) * (boxes1[:, ymax] - boxes1[:, ymin]), 1), (1, n))
    boxes2_areas = tf.tile(tf.expand_dims((boxes2[:, xmax] - boxes2[:, xmin]) * (boxes2[:, ymax] - boxes2[:, ymin]), 0), (m, 1))

    union_areas = boxes1_areas + boxes2_areas - intersection_areas

    return intersection_areas / union_areas


def match_bipartite_greedy(weight_matrix):

    num_gt_boxes = weight_matrix.shape[0]
    all_gt_indices = list(range(num_gt_boxes))

    matches = tf.zeros(num_gt_boxes)

    for _ in range(num_gt_boxes):

        anchor_indices = tf.argmax(weight_matrix, axis=1)
        overlaps = weight_matrix(all_gt_indices, anchor_indices)
        ground_truth_index = tf.argmax(overlaps)
        anchor_index = anchor_indices[ground_truth_index]
        matches[ground_truth_index] = anchor_index

        weight_matrix[ground_truth_index] = 0
        weight_matrix[:, anchor_index] = 0

    return matches

=======
    
    
>>>>>>> 9b4197c21069c222fb988fbdf685e6927689f684

def encode_ssd(gt_labels, *args):
    
    n_classes = args[1]

    class_id = 0
    xmin = 1
    ymin = 2
    xmax = 3
    ymax = 4

    batch_size = len(gt_labels)

    y_encoded = generate_encoding_template(batch_size, args)

    y_encoded[:, :, background_id] = 1
    n_boxes = y_encoded.shape[1]

    class_vectors = tf.eye(n_classes)

    for i in range(batch_size):
        labels = gt_labels[1]

        classes_one_hot = class_vectors[labels[:, class_id]]
        labels_one_hot = tf.concatenate([classes_one_hot, labels[:, [xmin, ymin, xmax, ymax]]], -1)

        similarities = iou(labels[:,[xmin, ymin, xmax, ymax]], y_encoded[i,:,-12:-8])

        bipartite_matches = match_bipartite_greedy(weight_matrix=similarities)

        y_encoded[i, bipartite_matches, :-8] = labels_one_hot

        similarities[:, bipartite_matches] = 0

        max_background_similiarites = tf.amax(similarities, 0)
        neutral_boxes = tf.nonzero(max_background_similiarites >= neg_iou_limit)[0]
        y_encoded[i, neutral_boxes, 0] = 0

        y_encoded[:,:,-12:-8] -= y_encoded[:,:,-8:-4]
        y_encoded[:,:,[-12,-10]] /= tf.expand_dims(y_encoded[:,:,-6] - y_encoded[:,:,-8], axis=-1) # (xmin(gt) - xmin(anchor)) / w(anchor), (xmax(gt) - xmax(anchor)) / w(anchor)
        y_encoded[:,:,[-11,-9]] /= tf.expand_dims(y_encoded[:,:,-5] - y_encoded[:,:,-7], axis=-1) # (ymin(gt) - ymin(anchor)) / h(anchor), (ymax(gt) - ymax(anchor)) / h(anchor)
        y_encoded[:,:,-12:-8] /= y_encoded[:,:,-4:] # (gt - anchor) / size(anchor) / variance for all four coordinates, where 'size' refers to w and h respectively

        return y_encoded



def generate_encoding_template(batch_size, *args):
    boxes_batch = []

    boxes_list = args[0]
    n_classes = args[1]
    variances = args[2]

    # Create boxes_list first.
    for boxes in boxes_list:
        boxes = tf.expand_dims(boxes, 0)
        boxes = tf.tile(boxes, (batch_size, 1, 1, 1, 1))
        
        # Reshape -> (Batch, Feature_Height * Feature_Width * n_boxes, 4)
        boxes = tf.reshape(boxes, (batch_size, -1, 4))
        boxes_batch.append(boxes)
    
    boxes_tensor = tf.concatenate(boxes_batch, 1)

    classes_tensor = tf.zeros((batch_size, boxes_tensor.shape[1], n_classes))

    variances_tensor = tf.zeros_like(boxes_tensor)
    variances_tensor += variances

    y_encoding_template = tf.concatenate((classes_tensor, boxes_tensor, boxes_tensor, variances_tensor), 2)

    return y_encoding_template


def generate_anchor_boxes_for_layer(
    img_size,
    feature_size,
    aspect_ratios,
    this_scale,
    next_scale,
    two_boxes_for_ar1,
    this_steps=None,
    this_offsets=None,
    clip_boxes=False,
    coords='centroids',
    normalize_coords=True):

    # Figure out image dimension ordering.
    img_width, img_height = img_size
    size = min(img_size[0], img_size[1])
    boxes = []

    
    for ar in aspect_ratios:
        if(ar == 1):
            box_height = box_width = this_scale * size
            boxes.append((box_width, box_height))

            if two_boxes_for_ar1:
                box_height = box_width = tf.sqrt(this_scale * next_scale) * size
                boxes.append((box_width, box_height))
        
        else:
            box_width = this_scale * size * tf.sqrt(ar)
            box_height = this_scale * size / tf.sqrt(ar)
            boxes.append((box_width, box_height))

    n_boxes = len(boxes)

    if this_steps is None:
        step_height = img_size[0] / feature_size[0]
        step_width = img_size[1] / feature_size[1]
    elif isinstance(this_steps, (list, tuple)) and (len(this_steps) == 2):
        step_height = this_steps[0]
        step_width = this_steps[1]
    elif isinstance(this_steps, (int, float)):
        step_height = this_steps
        step_width = this_steps

    if this_offsets is None:
        offset_height = 0.5
        offset_width = 0.5
    elif isinstance(this_offsets, (list, tuple)) and (len(this_offsets) == 2):
        offset_height = this_offsets[0]
        offset_width = this_offsets[1]
    elif isinstance(this_offsets, (int, float)):
        offset_height = this_offsets
        offset_width = this_offsets

    cy = tf.linespace(offset_height * step_height, (offset_height + feature_size[0] - 1) * step_height, feature_size[0])
    cx = tf.linespace(offset_width * step_width, (offset_width + feature_size[1] - 1) * step_width, feature_size[1])

    cx_grid, cy_grid = tf.meshgrid(cx, cy)
    cx_grid = tf.expand_dims(cx_grid, -1)
    cy_grid = tf.expand_dims(cy_grid, -1)

    boxes_tensor = tf.zeros((feature_size[0], feature_size[1], n_boxes, 4))

    boxes_tensor[:, :, :, 0] = tf.tile(cx_grid, (1, 1, n_boxes))
    boxes_tensor[:, :, :, 1] = tf.tile(cy_grid, (1, 1, n_boxes))
    boxes_tensor[:, :, :, 2] = boxes[:, 0]
    boxes_tensor[:, :, :, 3] = boxes[:, 1]

    boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='centroids2corners')

    if clip_boxes:
        x_coords = boxes_tensor[:, :, :, [0, 2]]
        x_coords[x_coords >= img_size[1]] = img_size[1] - 1
        x_coords[x_coords < 0] = 0
        boxes_tensor[:, :, :, [0, 2]] = x_coords

        y_coords = boxes_tensor[:, :, :, [1, 3]]
        y_coords[y_coords >= img_size[0]] = img_size[0] - 1
        y_coords[y_coords < 0] = 0
        boxes_tensor[:, :, :, [1, 3]] = y_coords

    if normalize_coords:
        boxes_tensor[:, :, :, [0, 2]] / img_width
        boxes_tensor[:, :, :, [1, 3]] / img_height

    if coords == 'centroids':
        boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='corners2centroids', border_pixels='half')
    elif coords == 'minmax':
        boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='corners2minmax', border_pixels='half')

    return boxes_tensor


    


