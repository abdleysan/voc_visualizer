import cv2
import xmltodict 

def get_bbox_from_obj(obj):
    b_box_dict = obj['bndbox']
    x1, y1, x2, y2 = get_dict_to_bbox(b_box_dict)
    class_name = obj['name']
    return ([x1, y1, x2, y2], class_name)

def get_dict_to_bbox(bbox_dict: dict) -> list:
    return list(int(float(x)) for x in bbox_dict.values())

def draw_bbox_from_obj(img, obj):
    [x1, y1, x2, y2], class_name = get_bbox_from_obj(obj)
    cv2.rectangle(img, (x1,y1), (x2,y2), color=(255,0,0))
    cv2.putText(img, class_name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (0,255,0), 1, cv2.LINE_AA)


def get_objects_from_annotations(annotation_path):
    with open(annotation_path) as fd:
        ann_text = fd.read()
    
    ann_xmltodict = xmltodict.parse(ann_text)
    
    objects = ann_xmltodict['annotation']['object']
    if isinstance(objects, dict):
        objects = [objects]
    return objects    


def visualize_annotation(image_path, annotation_path):
    img = cv2.imread(str(image_path))
    objects = get_objects_from_annotations(annotation_path)
    for obj_i in objects:
        draw_bbox_from_obj(img, obj_i)

    return img
