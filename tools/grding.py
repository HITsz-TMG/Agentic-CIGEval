import numpy as np
import os
import re
import torch
from collections import defaultdict
from groundingdino.util.inference import load_model, load_image, predict
from torchvision.ops import box_convert
from PIL import Image, ImageDraw


os.environ["CUDA_VISIBLE_DEVICES"] = "6"

class Grounding_Module():
    def __init__(self, base_dir):
        self.model = load_model(
            os.path.join(base_dir, "groundingdino/config/GroundingDINO_SwinT_OGC.py"), 
            os.path.join(base_dir, "weights/groundingdino_swint_ogc.pth")
            )

    def forward(self, img, prompt, bbox_thrd, text_thrd, do_clean=True):
        img_source, img = load_image(image_path=img)
        w, h = img_source.shape[1], img_source.shape[0]
        boxes, logits, phrases = predict(
            model=self.model,
            image=img,
            caption=prompt,
            box_threshold=bbox_thrd,
            text_threshold=text_thrd,
            )
        boxes = boxes * torch.Tensor([w, h, w, h])
        boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        boxes = list(boxes)
        logits = logits.numpy()
        logits = list(logits)
        res = []
        for bbox, logit, phrase in zip(boxes, logits, phrases):
            res.append((list([int(xy) for xy in bbox]), logit, phrase))
        if do_clean:
            res = self._clean_bbox(res)
        return sorted(res, key=lambda x: x[1], reverse=True)
    
    def _clean_bbox(self, bbox_list):
        def get_range(bbox):
            return (bbox[2]-bbox[0]) * (bbox[3]-bbox[1])
        def check_recap(bbox1, bbox2):
            if bbox2[0]<bbox1[0] and bbox2[1]<bbox1[1] and bbox2[2]>bbox1[2] and bbox2[3]>bbox1[3]:
                return True
            return False

        bbox_list = sorted(bbox_list, key=lambda x: get_range(x[0]))
        cleaned_bbox_list = []
        for bbox in bbox_list:
            if len(bbox_list) == 0:
                cleaned_bbox_list.append(bbox)
                continue

            flag = True
            for cleaned_bbox in cleaned_bbox_list:
                if check_recap(cleaned_bbox[0], bbox[0]):
                    flag = False
                    break
            if flag:
                cleaned_bbox_list.append(bbox)
        return cleaned_bbox_list





def filter_bboxes_by_max_logit(a):
    phrase_dict = defaultdict(lambda: (None, float('-inf')))
    for bbox, logit, phrase in a:
        if logit > phrase_dict[phrase][1]:
            phrase_dict[phrase] = (bbox, logit)
    filtered_list = [bbox for phrase, (bbox, logit) in phrase_dict.items()]
    return filtered_list



def tolist(a):
    box_list = []
    for bbox, logit, phrase in a:
        box_list.append(bbox)
    return box_list


grounding_module = Grounding_Module("PATH_TO_GroundingDINO")


def grding(img_path, text, function):
    res = grounding_module.forward(img_path, text, bbox_thrd=0.2, text_thrd=0.2, do_clean=True) # (bbox, logit, phrase)
        
    if len(res) == 0:
        print("Grounding: no result")
        if isinstance(img_path, str):
            img_path = Image.open(image_path).convert("RGB")
        if function=="highlight++":
            return(img_path, res)
        return img_path
    else:
        if function=="mark":
            res = filter_bboxes_by_max_logit(res)
            return(mark(img_path, res))
        if function=="highlight":
            res = filter_bboxes_by_max_logit(res)
            return(highlight(img_path, res))
        if function=="segment":
            return(segment(img_path, res[-1][0]))
        if function=="highlight++":
            res = filter_bboxes_by_max_logit(res)
            return(highlight(img_path, res), res)

