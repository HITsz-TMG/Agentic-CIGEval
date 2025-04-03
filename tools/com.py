import numpy as np
from PIL import Image, ImageDraw


def mark(image, bbox_list):
    if isinstance(image, str):
        image = Image.open(image)
    image_copy = image.copy()
    draw = ImageDraw.Draw(image_copy)
    for bbox in bbox_list:
        draw.rectangle(bbox, outline='red', width=3)
    return image_copy
    
    
def highlight(image, bbox_list):
    if isinstance(image, str):
        image = Image.open(image)
    np_img = np.array(image)
    np_ori = np_img.copy()
    if len(bbox_list)>0:
        np_img //= 4
    for bbox in bbox_list:
        np_img[bbox[1]:bbox[3], bbox[0]:bbox[2]] = np_ori[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    image_h = Image.fromarray(np_img)
    return image_h


def segment(image, bbox):
    if isinstance(image, str):
        image = Image.open(image)
    np_img = np.array(image)
    bbox_area = np_img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    image_s = Image.fromarray(bbox_area)
    return image_s


def split2part(image):
    if isinstance(image, str):
        img = Image.open(image)
    else:
        img = image
    w = img.width
    h = img.height
    box_L = (0,0,w*0.5,h)
    box_R = (w*0.5,0,w,h)
    img_L = img.crop(box_L)
    img_R = img.crop(box_R)
    return img_L,img_R