import base64
import requests
from io import BytesIO
from PIL import Image, ImageOps
import os
import openai
from openai import OpenAI


class GPT4o():
    def __init__(self, model_name="gpt-4o-2024-05-13", api_key = "API_KEY", base_url = "BASE_URL"):
        self.model_name = model_name
        self.client = OpenAI(api_key = api_key, base_url = base_url)

    def prepare_prompt(self, image_links = [], text_prompt = ""):
        if not isinstance(image_links, list):
            image_links = [image_links]
            
        prompt_content = []
        text_dict = {
                    "type": "text",
                    "text": text_prompt
                }
        prompt_content.append(text_dict)

        for image_link in image_links:
            if "base64" not in image_link:
                img = load_image(image_link)
                image_link = f"data:image/jpeg;base64,{encode_pil_image(img)}"
            visual_dict = {
                    "type": "image_url",
                    "image_url": {"url": image_link}
            }
            prompt_content.append(visual_dict)
        return prompt_content


    def get_result(self, prompt):
        try:
            response = self.client.chat.completions.create(
                model = self.model_name,
                messages = [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]                                      
            )

            out = response.choices[0].message.content
            return out
        except Exception as e:
            print(f"Error: {e}")
            return None


        
    

class QWen25(GPT4o):
    def __init__(self, model_name="qwen2.5-vl-72b-instruct", api_key = "API_KEY", base_url = "BASE_URL"):
        super().__init__(model_name, api_key, base_url)





class LlavaNext(GPT4o):
    def __init__(self):
        self.address = "http://127.0.0.1:8080/generate"

    def get_result(self, images, text):
        if not isinstance(images, list):
            images = [images]
        data = {'imgs':images, 'text':text}
        try:
            response = requests.post(self.address, json = data).json()
            out = response['response']
            return out

        except Exception as e:
            print(f"Error: {e}")
            return None



############################################################### Image Processing Functions
###############################################################
def load_image(image, format = "RGB"):
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


def encode_pil_image(pil_image, format="JPEG"):
    image_stream = BytesIO()
    pil_image.save(image_stream, format=format)
    image_data = image_stream.getvalue()
    base64_image = base64.b64encode(image_data).decode('utf-8')   
    return base64_image

############################################################### Image Processing Functions
###############################################################