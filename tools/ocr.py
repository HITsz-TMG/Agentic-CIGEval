from paddleocr import PaddleOCR,draw_ocr
ocr = PaddleOCR(lang='en') # need to run only once to download and load model into memory

def OCR(image):
    result = ocr.ocr(image, cls=False)
    texts = []
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            texts.append(line[1][0])
    return texts

