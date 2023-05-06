import numpy as np
import xmltodict 
import pprint
import json
from tqdm import tqdm
from pathlib import Path
import cv2
from pprint import pprint
from utils import visualize_annotation


if __name__ == '__main__':

    datasets_dir = Path('C:\\Users\\user\\Desktop\\datasets\\VOCtrainval_11-May-2012\\VOCdevkit\\VOC2012')
    images_dir = datasets_dir / 'JPEGImages'
    annotations_dir = datasets_dir / 'Annotations'
    save_dir = Path('C:\\Users\\user\\Desktop')/'Results'
    save_dir.mkdir(exist_ok=True)


    images_paths = list(images_dir.glob('*.jpg'))
    annotations_paths = list(annotations_dir.glob('*.xml'))
    
    for i in tqdm(range(len(images_paths))):
        image_path = images_paths[i]
        image_name = image_path.name

        img_path = str(image_path)
        ann_path = annotations_paths[i]

        img = visualize_annotation(img_path, ann_path)

        save_path = save_dir / image_name
        cv2.imwrite(str(save_path), img)



