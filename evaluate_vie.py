from agent.openai import GPT4o,QWen25
from agent.prompt_mm import *
from util import *
from tqdm import tqdm
import os



def test_Text_Guided_IE_evaluate(in_path, out_path):
    data = read_json(in_path)
    out = {}
    for key in tqdm(list(data.keys())):
        try:
            instruction = data[key]["prompt_input"].replace("Editing instruction:", "").strip()
            task = Text_Guided_IE.replace("{instruction}", instruction)
            img_links = [url2path(data[key]["vision_input"][0], Image_Root), url2path(data[key]["vision_input"][1], Image_Root)]
            json_1 = GPTResponse2JSON(agent_run.get_result(img_links, task))

            out[key] = {}
            out[key]['score'] = json_1['score']
            out[key]['reasoning'] = json_1['reasoning']
            out[key]['prompt_input'] = task
            out[key]['vision_input'] = data[key]["vision_input"]
            print(f"{key} over")
        except Exception as e:
            print(f"Error: {key} evaluation failed: {e}")
    
    write_json(out_path, out)



def test_Subject_Driven_IE_evaluate(in_path, out_path):
    data = read_json(in_path)
    out = {}
    for key in tqdm(list(data.keys())):
        try:
            subject = data[key]["prompt_input"].replace("Subject:", "").strip()
            task = Subject_Driven_IE.replace("{subject}", subject)
            im_1 = load_image(url2path(data[key]["vision_input"][0], Image_Root))
            im_2 = load_image(url2path(data[key]["vision_input"][1], Image_Root))
            im = merge_images([im_1, im_2])
            img_links = [f"data:image/jpeg;base64,{encode_pil_image(im)}", url2path(data[key]["vision_input"][2], Image_Root)]
            json_1 = GPTResponse2JSON(agent_run.get_result(img_links, task))
            out[key] = {}
            out[key]['score'] = json_1['score']
            out[key]['reasoning'] = json_1['reasoning']
            out[key]['prompt_input'] = task
            out[key]['vision_input'] = data[key]["vision_input"]
            print(f"{key} over")
        except Exception as e:
            print(f"Error: {key} evaluation failed: {e}")
    
    write_json(out_path, out)



def test_Mask_Guided_IE_evaluate(in_path, out_path):
    data = read_json(in_path)
    out = {}
    for key in tqdm(list(data.keys())):
        try:
            instruction = data[key]["prompt_input"].replace("Editing instruction:", "").strip()
            task = Mask_Guided_IE.replace("{instruction}", instruction)
            img_links = [url2path(data[key]["vision_input"][0], Image_Root), url2path(data[key]["vision_input"][1], Image_Root)]
            json_1 = GPTResponse2JSON(agent_run.get_result(img_links, task))
            
            out[key] = {}
            out[key]['score'] = json_1['score']
            out[key]['reasoning'] = json_1['reasoning']
            out[key]['prompt_input'] = task
            out[key]['vision_input'] = data[key]["vision_input"]
            print(f"{key} over")
        except Exception as e:
            print(f"Error: {key} evaluation failed: {e}")
    
    write_json(out_path, out)



def test_Multi_Concept_IC_evaluate(in_path, out_path):
    data = read_json(in_path)
    out = {}
    for key in tqdm(list(data.keys())):
        try:
            text = data[key]["prompt_input"].replace("Text Prompt:", "").strip()
            task = Multi_Concept_IC.replace("{text}", text)
            img_links = [url2path(data[key]["vision_input"][0], Image_Root), url2path(data[key]["vision_input"][1], Image_Root)]
            json_1 = GPTResponse2JSON(agent_run.get_result(img_links, task))
            out[key] = {}
            out[key]['score'] = json_1['score']
            out[key]['reasoning'] = json_1['reasoning']
            out[key]['prompt_input'] = task
            out[key]['vision_input'] = data[key]["vision_input"]
            print(f"{key} over")
        except Exception as e:
            print(f"Error: {key} evaluation failed: {e}")
    
    write_json(out_path, out)



def test_Text_Guided_IG_evaluate(in_path, out_path):
    data = read_json(in_path)
    out = {}
    for key in tqdm(list(data.keys())):
        try:
            text = data[key]["prompt_input"].replace("Text Prompt:", "").strip()
            task = Text_Guided_IG.replace("{text}", text)
            img_links = [url2path(data[key]["vision_input"][0], Image_Root)]
            json_1 = GPTResponse2JSON(agent_run.get_result(img_links, task))

            out[key] = {}
            out[key]['score'] = json_1['score']
            out[key]['reasoning'] = json_1['reasoning']
            out[key]['prompt_input'] = task
            out[key]['vision_input'] = data[key]["vision_input"]
            print(f"{key} over")
        except Exception as e:
            print(f"Error: {key} evaluation failed: {e}")

    write_json(out_path, out)
    


def test_Control_Guided_IG_evaluate(in_path, out_path):
    data = read_json(in_path)
    out = {}
    for key in tqdm(list(data.keys())):
        try:
            text = data[key]["prompt_input"].replace("Text Prompt:", "").strip()
            task = Control_Guided_IG.replace("{text}", text)
            img_links = [url2path(data[key]["vision_input"][0], Image_Root), url2path(data[key]["vision_input"][1], Image_Root)]
        
            json_2 = GPTResponse2JSON(agent_run.get_result(img_links, task))
            out[key] = {}
            out[key]['score'] = json_2['score']
            out[key]['reasoning'] = json_2['reasoning']
            out[key]['prompt_input'] = task
            out[key]['vision_input'] = data[key]["vision_input"]
            print(f"{key} over")
        except Exception as e:
            print(f"Error: {key} evaluation failed: {e}")
    
    write_json(out_path, out)



def test_Subject_Driven_IG_evaluate(in_path, out_path):
    data = read_json(in_path)
    out = {}
    for key in tqdm(list(data.keys())):
        try:
            text = data[key]["prompt_input"].replace("Text Prompt:", "").strip()
            task = Subject_Driven_IG.replace("{text}", text)
            img_links = [url2path(data[key]["vision_input"][0], Image_Root), url2path(data[key]["vision_input"][1], Image_Root)]
            json_1 = GPTResponse2JSON(agent_run.get_result(img_links, task))
            out[key] = {}
            out[key]['score'] = json_1['score']
            out[key]['reasoning'] = json_1['reasoning']
            out[key]['prompt_input'] = task
            out[key]['vision_input'] = data[key]["vision_input"]
            print(f"{key} over")
        except Exception as e:
            print(f"Error: {key} evaluation failed: {e}")

    write_json(out_path, out)



def evaluate(in_path, eva_out_path):
    if "ImagenHub_Control-Guided_IG" in in_path:
        test_Control_Guided_IG_evaluate(in_path, eva_out_path)
    if "ImagenHub_Mask-Guided_IE" in in_path:
        test_Mask_Guided_IE_evaluate(in_path, eva_out_path)
    if "ImagenHub_Multi-Concept_IC" in in_path:
        test_Multi_Concept_IC_evaluate(in_path, eva_out_path)
    if "ImagenHub_Subject-Driven_IE" in in_path:
        test_Subject_Driven_IE_evaluate(in_path, eva_out_path)
    if "ImagenHub_Subject-Driven_IG" in in_path:
        test_Subject_Driven_IG_evaluate(in_path, eva_out_path)
    if "ImagenHub_Text-Guided_IE" in in_path:
        test_Text_Guided_IE_evaluate(in_path, eva_out_path)
    if "ImagenHub_Text-Guided_IG" in in_path:
        test_Text_Guided_IG_evaluate(in_path, eva_out_path)




agent_run = QWen25()
Image_Root = "PATH_TO_ImageHub_DATA"

for task in os.listdir(Image_Root):
    task_path = os.path.join(Image_Root, task)
    if os.path.isfile(task_path):
        continue
    models = os.listdir(task_path)
    for dir in models:
        if dir!="input" and dir!="token":
            eva_out_path = os.path.join(task_path, dir, "SC_eva_qwen25_72b_vie.json")
            in_path =os.path.join(task_path, dir, "in.json")
            evaluate(in_path, eva_out_path)
