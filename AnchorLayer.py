import tensorflow as tf
from tensorflow.keras.layers import Layer

"""
    Good references:
        https://stackoverflow.com/questions/56259670/difference-between-box-coordinate-and-anchor-boxes-in-keras

"""


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


class AnchorBoxes(Layer):

    def __init__(self, img_size, aspect_ratios, this_scale, next_scale, two_boxes_for_ar1, this_steps, this_offsets, variances, clip_boxes, normalize_coords, **kwargs):
        
        self.img_size = img_size
        self.aspect_ratios = [1.0]
        self.this_scale = this_scale
        self.next_scale = next_scale
        self.two_boxes_for_ar1 = two_boxes_for_ar1
        self.variances = variances
        self.clip_boxes = clip_boxes
        self.this_steps = this_steps
        self.this_offsets = this_offsets
        self.normalize_coords = normalize_coords

        if aspect_ratios:
            for ar in aspect_ratios:
                if ar in self.aspect_ratios:
                    continue
                self.aspect_ratios.append(ar)

        if two_boxes_for_ar1:
            self.n_boxes = len(self.aspect_ratios) + 1
        else:
            self.n_boxes = len(self.aspect_ratios)

        super(AnchorBoxes, self).__init__(**kwargs)
        
 
    def call(self, x):
        input_shape = tf.shape(x)
        feature_width = input_shape[2]
        feature_height = input_shape[1]

        img_width = self.img_size[1]
        img_height = self.img_size[0]

        size = min(self.img_width, self.img_height)

        boxes = []
        for ar in self.aspect_ratios:
            if ar == 1:
                box_height = box_width = self.this_scale * size
                boxes.append((box_width, box_height))

                if self.two_boxes_for_ar1:
                    box_height = box_width = tf.sqrt(self.this_scale * self.next_scale) * size

            else:
                box_height = self.this_scale * size / tf.sqrt(ar)
                box_width = self.this_scale * size * tf.sqrt(ar)
                boxes.append((box_width, box_height))


        # DO THIS IN INIT. ?? WTF IS WRONG WITH NOOBS!?!            
        if self.this_steps is None:
            step_height = self.img_height / feature_height
            step_width = self.img_width / feature_width

        else:
            if isinstance(self.this_steps, (list, tuple)) and (len(self.this_steps) == 2):
                step_height = self.this_steps[0]
                step_width = self.this_steps[1]
            elif isinstance(self.this_steps, (int, float)):
                step_height = self.this_steps
                step_width = self.this_steps

        if self.this_offsets is None:
            offset_height = 0.5
            offset_width = 0.5
        
        else:
            if isinstance(self.this_offsets, (list, tuple)) and (len(self.this_offsets) == 2):
                offset_height = self.this_offsets[0]
                offset_width = self.this_offsets[1]
            elif isinstance(self.this_offsets, (int, float)):
                offset_height = self.this_offsets
                offset_width = self.this_offsets

        
        cy = tf.linespace(offset_height * step_height, (offset_height + feature_height - 1) * step_height, feature_height)
        cx = tf.linespace(offset_width * step_width, (offset_width + feature_width - 1) * step_width, feature_width)

        cx_grid, cy_grid = tf.meshgrid(cx, cy)
        cx_grid = tf.expand_dims(cx_grid, -1)
        cy_grid = tf.expand_dims(cy_grid, -1)

        boxes_tensor = tf.zeros((feature_height, feature_width, self.n_boxes, 4))

        boxes_tensor[:, :, :, 0] = tf.tile(cx_grid, (1, 1, self.n_boxes))
        boxes_tensor[:, :, :, 1] = tf.tile(cy_grid, (1, 1, self.n_boxes))
        boxes_tensor[:, :, :, 2] = boxes[:, 0]
        boxes_tensor[:, :, :, 3] = boxes[:, 1]

        boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='centroids2corners')

        if self.clip_boxes:
            x_coords = boxes_tensor[:, :, :, [0, 2]]
            x_coords[x_coords >= img_width] = img_width - 1
            x_coords[x_coords < 0] = 0
            boxes_tensor[:, :, :, [0,2]] = x_coords

            y_coords = boxes_tensor[:, :, :, [1, 3]]
            y_coords[y_coords >= img_height] = img_height - 1
            y_coords[y_coords < 0] = 0
            boxes_tensor[:, :, :, [1, 3]] = y_coords

        if self.normalize_coords:
            boxes_tensor[:, :, :, [0, 2]] /= img_width
            boxes_tensor[:, :, :, [1, 3]] /= img_height


        # Ensure coords are still valid.

        variances_tensor = tf.zeros_like(boxes_tensor)
        variances_tensor += self.variances
        boxes_tensor = tf.concatenate((boxes_tensor, variances_tensor), axis=-1)

        boxes_tensor = tf.expand_dims(boxes_tensor, 0)
        boxes_tensor = tf.tile(boxes_tensor, (input_shape[0], 1, 1, 1, 1))

        return boxes_tensor