import json
import os
import re
from agent.openai import GPT4o
from typing import Union
import base64
import requests
import Levenshtein
from io import BytesIO
from PIL import Image, ImageOps

gpt = GPT4o()


def url2path(url, root):
    its = url.split("/")
    return os.path.join(root, its[-3], its[-2], its[-1])


def merge_images(images):
    if len(images) == 0:
        return None
    if len(images) == 1:
        return images[0]
    widths, heights = zip(*(i.size for i in images))
    average_height = sum(heights) // len(heights)
    for i, im in enumerate(images):
        # scale in proportion
        images[i] = im.resize((int(im.size[0] * average_height / im.size[1]), average_height))
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new("RGB", (total_width + 10 * (len(images) - 1), max_height))
    x_offset = 0
    for i, im in enumerate(images):
        if i > 0:
            # past a column of 1 pixel starting from x_offset width being black, 8 pixels being white, and 1 pixel being black
            new_im.paste(Image.new("RGB", (1, max_height), (0, 0, 0)), (x_offset, 0))
            x_offset += 1
            new_im.paste(Image.new("RGB", (8, max_height), (255, 255, 255)), (x_offset, 0))
            x_offset += 8
            new_im.paste(Image.new("RGB", (1, max_height), (0, 0, 0)), (x_offset, 0))
            x_offset += 1
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    return new_im


# Function to encode a PIL image
def encode_pil_image(pil_image, format="JPEG"):
    image_stream = BytesIO()
    pil_image.save(image_stream, format=format)
    image_data = image_stream.getvalue()
    base64_image = base64.b64encode(image_data).decode('utf-8')   
    return base64_image


def load_image(image: Union[str, Image.Image], format = "RGB") -> Image.Image:
    if isinstance(image, str):
        if image.startswith("http://") or image.startswith("https://"):
            image = Image.open(requests.get(image, stream=True).raw)
        elif os.path.isfile(image):
            image = Image.open(image)
        else:
            raise ValueError(
                f"{image} is not a valid path or url."
            )
    elif isinstance(image, Image.Image):
        image = image
    else:
        raise ValueError(
            "Incorrect format used for image. Should be an url linking to an image, a local path, or a PIL image."
        )
    image = ImageOps.exif_transpose(image)
    image = image.convert(format)
    return image


def read_json(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_json(output_path, output_data):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False)
    
    
def log_prompt(prompt_log_path, input):
    if not isinstance(input, str):
        input = toString(input)
    with open(prompt_log_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"{input}\n")
        log_file.write("#######################################################\n")
    
    
def cretae_new_path(path, filetype):
    files = os.listdir(path)
    if len(files) == 0:
        return os.path.join(path, f'0.{filetype}')
    else:
        max = 0
        for file in files:
            max = max if max>=int(os.path.splitext(file)[0]) else int(os.path.splitext(file)[0])
        return os.path.join(path, str(max+1) + f'.{filetype}')

def find_latest_file(path):
    files = os.listdir(path)
    max = 0
    f = ""
    for file in files:
        if max < int(os.path.splitext(file)[0]):
            max = int(os.path.splitext(file)[0])
            f = file

    return os.path.join(path, f)

def toString(input):
    return json.dumps(input, ensure_ascii=False, separators=(",", ":"))          
       
        
def prompt_format(prompt, params):
    text = prompt
    for key, value in params.items():
        if isinstance(value, (dict, list)):
            value = toString(value)
        if isinstance(value, (int, float)):
            value = str(value)
        text = text.replace(key, value)
    return text


def calculate_similarity(str1, str2):
    distance = Levenshtein.distance(str1.lower(), str2.lower())
    max_len = max(len(str1), len(str2))
    similarity = 1 - distance / max_len
    return similarity

    
def return_most_similar(string, string_list):
    max_similarity = 0
    tgt = 0
    for id,item in enumerate(string_list):
        current_similarity = calculate_similarity(string, item)
        if current_similarity > max_similarity:
            max_similarity = current_similarity
            tgt = id
    
    return string_list[tgt]


def check(src, keys):
    if isinstance(src, list):
        dst = []
        for item in src:
            dst_item = {}
            for item_key in item.keys():
                key = return_most_similar(item_key, keys)
                dst_item[key] = item[item_key]
            for _key in keys:
                if _key not in item.keys():
                    dst_item[_key] = "None"
            dst.append(dst_item)
    else:
        dst = {}
        for _key in src.keys():
            key = return_most_similar(_key, keys)
            dst[key] = src[_key]
    
    return dst
    
def GPTResponse2JSON(response):
    json_string = clean_text(response)
    # json_string = firstjson(clean_text(response))
    prompt = f"Modify the following string so that it can be correctly parsed by the json.loads() method:\n{json_string}\n\nYou should just return the modified string."
    print(json_string)
    if "```json" in json_string:
        json_string = json_string.replace("```","")
        json_string = json_string.replace("json","")
        json_string = json_string.strip()
    try:
        result = json.loads(json_string)
    except:
        _prompt = gpt.prepare_prompt(text_prompt=prompt)
        result = json.loads(clean_text(gpt.get_result(_prompt)))

    return result


def clean_text(text):
    # Only keep json content
    pattern = r"```json(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        text =  match.group(1)
    return text


def get_number(text):
    if isinstance(text, str):
        pattern = r'[^0-9]'
        text = re.sub(pattern, '', text)
        return int(text)
    else:
        return text


def firstjson(text):
    match = re.search(r'\{([^}]*)\}', text)
    if match:
        return "{" + match.group(1) + "}"
    else:
        return text


def matchkey(json, key):
    for k in json.keys():
        if key in k:
            return k
    return None
