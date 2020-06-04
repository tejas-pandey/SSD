import os
import tensorflow as tf

from pycocotools.coco import COCO


_THREADS = 4
labels = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant"," ","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe"," ","backpack","umbrella"," "," ","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle"," ","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed"," ","dining table"," "," ","toilet"," ","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator"," ","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]


def create_dataset_list(json_file):
    coco = COCO(json_file)

    image_list = []
    label_list = []
    annotation_list = []

    for img_id in coco.getImgIds():
        img_data = coco.loadImgs(img_id)[0]
        ann_id = coco.getAnnIds(img_id)
        anns = coco.loadAnns(ann_id)
        print(anns)

        exit(0)

        for ann in anns:
            image_list.append(os.path.abspath(os.path.join("val2017", img_data['file_name'])))
            label_list.append(labels[ann['category_id']-1])
            annotation_list.append(ann['bbox'])

    return image_list, label_list, annotation_list


def image_preprocessing(img_filepath):
    image = tf.io.read_file(img_filepath)
    image = tf.image.decode_jpeg(image)
    image = tf.image.resize(image, (300, 300))

    # Augmentations here
    return image

def annotation_preprocessing(*bbox):
    # print(bbox)
    return bbox

def create_tf_dataset(image_list, label_list, annotation_list):
    image_dataset = tf.data.Dataset.from_tensor_slices(image_list)
    label_dataset = tf.data.Dataset.from_tensor_slices(label_list)
    annotation_dataset = tf.data.Dataset.from_tensor_slices(annotation_list)

    image_dataset = image_dataset.map(image_preprocessing, _THREADS)
    annotation_dataset = annotation_dataset.map(annotation_preprocessing, _THREADS)

    for ann in annotation_dataset.take(10):
        print(ann)


if __name__ == "__main__":
    json_file = "/home/tejaspan/gits/SSD/instances_val2017.json"
    image_list, label_list, annotation_list = create_dataset_list(json_file)
    # create_tf_dataset(image_list, label_list, annotation_list)
