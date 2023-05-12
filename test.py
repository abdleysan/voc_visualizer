import numpy as np
import xmltodict 
import pprint
import json
from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd
import cv2
from pprint import pprint
from utils import visualize_annotation, get_objects_from_annotations, get_bbox_from_obj
import imagesize


if __name__ == '__main__':

    datasets_dir = Path('C:\\Users\\user\\Desktop\\datasets\\VOCtrainval_11-May-2012\\VOCdevkit\\VOC2012')
    images_dir = datasets_dir / 'JPEGImages'
    annotations_dir = datasets_dir / 'Annotations'
    save_dir = Path('C:\\Users\\user\\Desktop')/'Results'
    save_dir.mkdir(exist_ok=True)


    images_paths = list(images_dir.glob('*.jpg'))
    annotations_paths = list(annotations_dir.glob('*.xml'))
    
    class_names = {}
    analysis_obj_bbox = []
    for i in range(10):
    #for i in tqdm(range(len(images_paths))):
        image_path = images_paths[i]
        image_name = image_path.name

        img_path = str(image_path)
        ann_path = annotations_paths[i]

        #img = visualize_annotation(img_path, ann_path)
        image_width, image_height = imagesize.get(images_paths[i])
        objects = get_objects_from_annotations(ann_path)
        for obj in objects:
            [x1, y1, x2, y2], class_name_obj = get_bbox_from_obj(obj)
            w = x2-x1
            h = y2-y1
            c_x = x1+w/2
            c_y = y1+h/2
            analysis_obj_bbox.append([image_name, 
                                      image_width, 
                                      image_height,
                                      round(c_x, 3), 
                                      round(c_y, 3), 
                                      w, 
                                      h, 
                                      round(c_x/image_width, 4),
                                      round(c_y/image_height, 4),
                                      round(w/image_width, 4),
                                      round(h/image_height, 4),
                                      class_name_obj])
        
            #class_names[class_name_obj] = class_names[class_name_obj]+1 if class_name_obj in class_names else 1
    
    
    df = pd.DataFrame(analysis_obj_bbox, columns=['file_name', 
                                                  'image_heigh',
                                                  'image_width',
                                                  'x_center',
                                                  'y_center',
                                                  'widths',
                                                  'heights',
                                                  'x_center_norm',
                                                  'y_center_norm',
                                                  'w_norm',
                                                  'h_norm',
                                                  'class_name'])
    
    # print(df)
    # for i in range(len(analysis_obj_bbox)):
    #     axis([-0.05,1.05,-0.05,1.05])
    #     plt.scatter(analysis_obj_bbox[i][7], analysis_obj_bbox[i][8])
    #     plt.savefig('results/analyses.png')
    df1 = pd.DataFrame(analysis_obj_bbox)
    plt.axis([0,1,0,1])
    plt.scatter(df1[7],df1[8])
    plt.legend(loc=2)
    plt.show()

            #save_path = save_dir / image_name
            #cv2.imwrite(str(save_path), img)

    '''TODO 1) implement pandas dataframe base analytics
    Create pandas dataframe and analyse each column

    file_name | image_heigh | image_width | x_center | y_center | widths | heights | x_center_norm | y_center_norm | w_norm | h_norm | class_name

    NOTE: get image size using https://github.com/shibukawa/imagesize_py

    ''' 
    # print(class_names)

    # h=plt.bar(class_names.keys(), class_names.values(), align='center')
    # plt.savefig('results/classes_distribution.png')

