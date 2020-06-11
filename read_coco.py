import os
import numpy as np
import matplotlib.pyplot as plt

from pycocotools.coco import COCO

from PIL import Image
from PIL import ImageDraw
from PIL import ImageColor

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("filename", help="coco annotation file name.")
args = parser.parse_args()


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

    draw.text((left, top), labels[category_id - 1],fill=color)

              
if __name__ == '__main__':

    label_file = 'mscoco_labels.names'

    labels = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant"," ","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe"," ","backpack","umbrella"," "," ","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle"," ","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed"," ","dining table"," "," ","toilet"," ","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator"," ","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]
    coco = COCO(args.filename)

    img_ids = coco.getImgIds()

    for imd_id in img_ids[2:]:
        img_data = coco.loadImgs(imd_id)
        ann_id = coco.getAnnIds(imd_id)
        anns = coco.loadAnns(ann_id)
        break
    
    img_full_path = os.path.join("val2017", img_data[0]['file_name'])
    img = Image.open(img_full_path)

    for ann in anns:
        # img = Image.open(img_full_path)
        draw_bounding_box(img, ann['bbox'], ann['category_id'], labels)
    
    # plt.imshow(img)
    # plt.show()

    print(img_full_path)
    for ann in anns:
        print(ann['bbox'])
        print(ann['category_id'])

