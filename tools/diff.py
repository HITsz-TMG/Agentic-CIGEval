import cv2
import numpy as np
from PIL import Image
from com import mark, highlight

threshold = 0.5
top_n = 1

def detect_diff(src_img, dst_img):
    
    height, width = dst_img.shape[:2]
    total_area = height * width
    # Calculate area threshold (0.5% of total image area)
    area_threshold = total_area * (threshold / 100)
    
    src_img = cv2.GaussianBlur(src_img, [5, 5], 0)
    dst_img = cv2.GaussianBlur(dst_img, [5, 5], 0)
    diff = cv2.absdiff(src_img, dst_img)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    # Apply thresholding to get a binary image (45)
    _, result = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)

    result = cv2.dilate(result, np.ones([3, 3]))
    contours, _ = cv2.findContours(result, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rect_pos = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area > area_threshold:
            rect_pos.append((x, y, w, h))
            
            
    # Merge overlapping rectangles
    merged_rects = []
    while rect_pos:
        rect = rect_pos.pop(0)
        x, y, w, h = rect
        
        merge = False
        for other_rect in rect_pos[:]:
            ox, oy, ow, oh = other_rect
            if x < ox + ow and x + w > ox and y < oy + oh and y + h > oy:
                x = min(x, ox)
                y = min(y, oy)
                w = max(x + w, ox + ow) - x
                h = max(y + h, oy + oh) - y
                
                rect_pos.remove(other_rect)
                merge = True
        
        merged_rects.append((x, y, w, h))
        
        if not merge:
            break

    areas = []
    for C in merged_rects:
        x, y, w, h = C
        area = w * h
        areas.append((x, y, w, h, area))
    areas.sort(key=lambda x: x[4], reverse=True)

    pos = []
    for i in range(min(top_n, len(areas))):
        x, y, w, h, _ = areas[i]
        pos.append((x, y, x+w, y+h))
    
    return pos


def imgs_diff(src_img_path, dst_img_path, function="highlight"):
    src_img = cv2.imread(src_img_path)
    dst_img = cv2.imread(dst_img_path)
    if src_img.shape != dst_img.shape:
        dst_img = cv2.resize(dst_img, (src_img.shape[1], src_img.shape[0]))
    
    # Get the rectangular coordinates of the difference area
    rects = detect_diff(src_img, dst_img)
    
    if function == "highlight":
        src_highlight_img = highlight(src_img_path, rects)
        dst_highlight_img = highlight(dst_img_path, rects)
        return src_highlight_img, dst_highlight_img

    else:
        # Mark the difference areas on the image
        src_mark_img = mark(src_img_path, rects)
        dst_mark_img = mark(dst_img_path, rects)
        return src_mark_img, dst_mark_img