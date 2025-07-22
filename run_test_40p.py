from agent.llava_next import LlavaNext
from agent.prompt_agent import *
from util import *
from tools.scene_graph import sg_generate
from tools.grding import grding, highlight
from tools.diff import imgs_diff
from tools.split import split2part
from tqdm import tqdm


def get_tool_text(tool, img_links, log_path):
    if tool=="None":
        return ""
    if tool=="Highlight" or tool=="MaskFocus":
        return "Focus on the highlighted parts of the image."
    if tool=="SceneGraph":
        text = ""
        if len(img_links) == 2:
            text = """Two scene graphs in JSON format generated from two images will be provided:\nThe first scene graph:\n{scene_graph_1}\nThe second scene graph:\n{scene_graph_2}"""
            sg_1 = sg_generate(img_links[0], 0, log_path)
            sg_2 = sg_generate(img_links[1], 0, log_path)
            params = {"{scene_graph_1}": sg_1, "{scene_graph_2}": sg_2}
            text = prompt_format(text, params)
        if len(img_links) == 1:
            text = """The scene graph in JSON format generated from this image is as follows:\n{scene_graph}"""
            sg_1 = sg_generate(img_links[0], 0, log_path)
            params = {"{scene_graph}": sg_1}
            text = prompt_format(text, params)
        return text



def test_Text_Guided_IE_tool(in_path, out_path, log_path, test=None):
    data = read_json(in_path)
    out = {}
    for key in tqdm(list(data.keys())):
        if test:
            its = in_path.split("/")
            if key not in test[its[-3]][its[-2]]['sample']:
                continue
        counter = 0
        while counter < 3:
            try:
                instruction = data[key]["prompt_input"].replace("Editing instruction:", "").strip()
                # tool
                task_rule = Text_Guided_IE_Rule.replace("{instruction}", instruction)
                task = task_rule + Text_Guided_IE_Task_1 + Text_Guided_IE_Task_2
                prompt_tool = Tool_Decide.replace("{task}", task)
                img_links = [url2path(data[key]["vision_input"][0], Image_Root), url2path(data[key]["vision_input"][1], Image_Root)]
                
                result = GPTResponse2JSON(agent_run.get_result(img_links, prompt_tool))
                result = check(result, ["task_id", "reasoning", "used", "tool"])
                log_prompt(log_path, prompt_tool)
                log_prompt(log_path, result)
                # tool 

                out[key] = {}
                out[key]['tool_plan'] = result
                print(f"{key} over")
                break
            except Exception as e:
                print(f"Error: {key} evaluation failed: {e}")
                counter += 1
    
    write_json(out_path, out)



def test_Text_Guided_IE_evaluate(in_path, tool_path, out_path, log_path, test=None):
    data = read_json(in_path)
    data_tool = read_json(tool_path)
    out = {}
    for key in tqdm(list(data.keys())):
        if test:
            its = in_path.split("/")
            if key not in test[its[-3]][its[-2]]['sample']:
                continue
        counter = 0
        while counter < 3:
            try:
                instruction = data[key]["prompt_input"].replace("Editing instruction:", "").strip()
                # Task 1
                ## prompt
                if data_tool[key]["tool_plan"][0]["tool"] in ["Highlight", "SceneGraph"]:
                    tool = data_tool[key]["tool_plan"][0]["tool"]
                else:
                    tool = "None"
                task_1_eva = Text_Guided_IE_Task_1_evaluation.replace("{instruction}", instruction)
                tool_text = get_tool_text(tool, data[key]["vision_input"], log_path)
                task_1_eva = task_1_eva.replace("{tool_text}", tool_text)
                ## prompt
                ## image
                img_links = [url2path(data[key]["vision_input"][0], Image_Root), url2path(data[key]["vision_input"][1], Image_Root)]
                if tool == "Highlight":
                    img_1 = load_image(url2path(data[key]["vision_input"][0], Image_Root))
                    img_2 = load_image(url2path(data[key]["vision_input"][1], Image_Root))
                    img_1 = grding(img_1, instruction, "highlight")  
                    img_2 = grding(img_2, instruction, "highlight")  
                    img_links = [f"data:image/jpeg;base64,{encode_pil_image(img_1)}", f"data:image/jpeg;base64,{encode_pil_image(img_2)}"] 
                ## image
                    
                json_1 = GPTResponse2JSON(agent_run.get_result(img_links, task_1_eva))
                json_1['score'] = get_number(json_1['score'])
                log_prompt(log_path, task_1_eva)
                log_prompt(log_path, json_1)
                # Task 1
                
                # Task 2
                ## prompt
                if data_tool[key]["tool_plan"][1]["tool"] in ["Highlight", "SceneGraph"]:
                    tool = data_tool[key]["tool_plan"][1]["tool"]
                else:
                    tool = "None"
                task_2_eva = Text_Guided_IE_Task_2_evaluation.replace("{instruction}", instruction)
                tool_text = get_tool_text(tool, data[key]["vision_input"], log_path)
                task_2_eva = task_2_eva.replace("{tool_text}", tool_text)
                ## prompt
                ## image
                img_links = [url2path(data[key]["vision_input"][0], Image_Root), url2path(data[key]["vision_input"][1], Image_Root)]
                if tool == "Highlight":
                    img_1 = load_image(url2path(data[key]["vision_input"][0], Image_Root))
                    img_2 = load_image(url2path(data[key]["vision_input"][1], Image_Root))
                    img_1 = grding(img_1, instruction, "highlight")  
                    img_2 = grding(img_2, instruction, "highlight")  
                    img_links = [f"data:image/jpeg;base64,{encode_pil_image(img_1)}", f"data:image/jpeg;base64,{encode_pil_image(img_2)}"] 
                
                ## image
                
                json_2 = GPTResponse2JSON(agent_run.get_result(img_links, task_2_eva))
                json_2['score'] = get_number(json_2['score'])
                log_prompt(log_path, task_2_eva)
                log_prompt(log_path, json_2)
                # Task 2

                out[key] = {}
                out[key]['score'] = [json_1['score'], json_2['score']]
                out[key]['reasoning'] = [json_1['reasoning'], json_2['reasoning']]
                out[key]['prompt_input'] = [task_1_eva, task_2_eva]
                out[key]['vision_input'] = data[key]["vision_input"]
                print(f"{key} over")
                break
            except Exception as e:
                print(f"Error: {key} evaluation failed: {e}")
                counter += 1
    
    write_json(out_path, out)



def test_Subject_Driven_IE_tool(in_path, out_path, log_path, test=None):
    data = read_json(in_path)
    out = {}
    for key in tqdm(list(data.keys())):
        if test:
            its = in_path.split("/")
            if key not in test[its[-3]][its[-2]]['sample']:
                continue
        counter = 0
        while counter < 3:
            try:
                subject = data[key]["prompt_input"].replace("Subject:", "").strip()
                # tool
                task_rule = Subject_Driven_IE_Rule_llava.replace("{subject}", subject)
                task = task_rule + Subject_Driven_IE_Task_1_llava + Subject_Driven_IE_Task_2_llava
                prompt_tool = Tool_Decide.replace("{task}", task)
                im_1 = load_image(url2path(data[key]["vision_input"][0], Image_Root))
                im_2 = load_image(url2path(data[key]["vision_input"][1], Image_Root))
                im = merge_images([im_1, im_2])

                img_links = [f"data:image/jpeg;base64,{encode_pil_image(im)}", url2path(data[key]["vision_input"][2], Image_Root)]
                
                result = GPTResponse2JSON(agent_run.get_result(img_links, prompt_tool))
                result = check(result, ["task_id", "reasoning", "used", "tool"])
                log_prompt(log_path, prompt_tool)
                log_prompt(log_path, result)
                # tool 

                out[key] = {}
                out[key]['tool_plan'] = result
                print(f"{key} over")
                break
            except Exception as e:
                print(f"Error: {key} evaluation failed: {e}")
                counter += 1
    
    write_json(out_path, out)



def test_Subject_Driven_IE_evaluate(in_path, tool_path, out_path, log_path, test=None):
    data = read_json(in_path)
    data_tool = read_json(tool_path)
    out = {}
    for key in tqdm(list(data.keys())):
        if test:
            its = in_path.split("/")
            if key not in test[its[-3]][its[-2]]['sample']:
                continue
        counter = 0
        while counter < 3:
            try:
                subject = data[key]["prompt_input"].replace("Subject:", "").strip()
                # Task 1
                ## prompt
                if data_tool[key]["tool_plan"][0]["tool"] in ["Highlight", "SceneGraph"]:
                    tool = data_tool[key]["tool_plan"][0]["tool"]
                else:
                    tool = "None"
                task_1_eva = Subject_Driven_IE_Task_1_evaluation.replace("Subject: {subject}", "")
                tool_text = get_tool_text(tool, [data[key]["vision_input"][1], data[key]["vision_input"][2]], log_path)
                task_1_eva = task_1_eva.replace("{tool_text}", tool_text)
                ## prompt
                ## image
                img_links = [url2path(data[key]["vision_input"][1], Image_Root), url2path(data[key]["vision_input"][2], Image_Root)]
                if tool == "Highlight":
                    img_1 = load_image(url2path(data[key]["vision_input"][1], Image_Root))
                    img_2 = load_image(url2path(data[key]["vision_input"][2], Image_Root))
                    img_1 = grding(img_1, subject, "highlight")  
                    img_2 = grding(img_2, subject, "highlight")  
                    img_links = [f"data:image/jpeg;base64,{encode_pil_image(img_1)}", f"data:image/jpeg;base64,{encode_pil_image(img_2)}"] 
                ## image

                json_1 = GPTResponse2JSON(agent_run.get_result(img_links, task_1_eva))
                json_1['score'] = get_number(json_1['score'])
                log_prompt(log_path, task_1_eva)
                log_prompt(log_path, json_1)
                # Task 1
                
                # Task 2
                ## prompt
                if data_tool[key]["tool_plan"][1]["tool"] in ["Highlight", "SceneGraph"]:
                    tool = data_tool[key]["tool_plan"][1]["tool"]
                else:
                    tool = "None"
                tool_text = get_tool_text(tool, [data[key]["vision_input"][0], data[key]["vision_input"][2]], log_path)
                task_2_eva = Subject_Driven_IE_Task_2_evaluation.replace("{tool_text}", tool_text)
                ## prompt
                ## image
                img_links = [url2path(data[key]["vision_input"][0], Image_Root), url2path(data[key]["vision_input"][2], Image_Root)]
                if tool == "Highlight":
                    img_1 = load_image(url2path(data[key]["vision_input"][0], Image_Root))
                    img_2 = load_image(url2path(data[key]["vision_input"][2], Image_Root))
                    img_1 = grding(img_1, "background", "highlight")  
                    img_2 = grding(img_2, "background", "highlight")  
                    img_links = [f"data:image/jpeg;base64,{encode_pil_image(img_1)}", f"data:image/jpeg;base64,{encode_pil_image(img_2)}"] 
                
                ## image

                json_2 = GPTResponse2JSON(agent_run.get_result(img_links, task_2_eva))
                json_2['score'] = get_number(json_2['score'])
                log_prompt(log_path, task_2_eva)
                log_prompt(log_path, json_2)
                # Task 2

                out[key] = {}
                out[key]['score'] = [json_1['score'], json_2['score']]
                out[key]['reasoning'] = [json_1['reasoning'], json_2['reasoning']]
                out[key]['prompt_input'] = [task_1_eva, task_2_eva]
                out[key]['vision_input'] = data[key]["vision_input"]
                print(f"{key} over")
                break
            except Exception as e:
                print(f"Error: {key} evaluation failed: {e}")
                counter += 1
    
    write_json(out_path, out)



def test_Mask_Guided_IE_tool(in_path, out_path, log_path, test=None):
    data = read_json(in_path)
    out = {}
    st = False
    for key in tqdm(list(data.keys())):
        if test:
            its = in_path.split("/")
            if key not in test[its[-3]][its[-2]]['sample']:
                continue
        counter = 0
        while counter < 3:
            try:
                instruction = data[key]["prompt_input"].replace("Editing instruction:", "").strip()
                # tool
                task_rule = Mask_Guided_IE_Rule.replace("{instruction}", instruction)
                task = task_rule + Mask_Guided_IE_Task_1 + Mask_Guided_IE_Task_2
                prompt_tool = Tool_Decide.replace("{task}", task)
                img_links = [url2path(data[key]["vision_input"][0], Image_Root), url2path(data[key]["vision_input"][1], Image_Root)]

                result = GPTResponse2JSON(agent_run.get_result(img_links, prompt_tool))
                result = check(result, ["task_id", "reasoning", "used", "tool"])
                log_prompt(log_path, prompt_tool)
                log_prompt(log_path, result)
                # tool 

                out[key] = {}
                out[key]['tool_plan'] = result
                print(f"{key} over")
                break
            except Exception as e:
                print(f"Error: {key} evaluation failed: {e}")
                counter += 1
    
    write_json(out_path, out)



def test_Mask_Guided_IE_evaluate(in_path, tool_path, out_path, log_path, test=None):
    data = read_json(in_path)
    data_tool = read_json(tool_path)
    out = {}
    for key in tqdm(list(data.keys())):
        if test:
            its = in_path.split("/")
            if key not in test[its[-3]][its[-2]]['sample']:
                continue
        counter = 0
        while counter < 3:
            try:
                instruction = data[key]["prompt_input"].replace("Editing instruction:", "").strip()
                # Task 1
                ## prompt
                if data_tool[key]["tool_plan"][0]["tool"] in ["Highlight", "SceneGraph", "MaskFocus"]:
                    tool = data_tool[key]["tool_plan"][0]["tool"]
                else:
                    tool = "None"
                task_1_eva = Mask_Guided_IE_Task_1_evaluation.replace("{instruction}", instruction)
                tool_text = get_tool_text(tool, data[key]["vision_input"], log_path)
                task_1_eva = task_1_eva.replace("{tool_text}", tool_text)
                ## prompt
                ## image
                img_links = [url2path(data[key]["vision_input"][0], Image_Root), url2path(data[key]["vision_input"][1], Image_Root)]
                if tool == "Highlight":
                    img_1 = load_image(url2path(data[key]["vision_input"][0], Image_Root))
                    img_2 = load_image(url2path(data[key]["vision_input"][1], Image_Root))
                    img_1 = grding(img_1, instruction, "highlight")  
                    img_2 = grding(img_2, instruction, "highlight")  
                    img_links = [f"data:image/jpeg;base64,{encode_pil_image(img_1)}", f"data:image/jpeg;base64,{encode_pil_image(img_2)}"] 
                if tool == "MaskFocus":
                    img_1, img_2 = imgs_diff(url2path(data[key]["vision_input"][0], Image_Root), url2path(data[key]["vision_input"][1], Image_Root), "highlight")
                    img_links = [f"data:image/jpeg;base64,{encode_pil_image(img_1)}", f"data:image/jpeg;base64,{encode_pil_image(img_2)}"]
                ## image
 
                json_1 = GPTResponse2JSON(agent_run.get_result(img_links, task_1_eva))
                json_1['score'] = get_number(json_1['score'])
                log_prompt(log_path, task_1_eva)
                log_prompt(log_path, json_1)
                # Task 1
                
                # Task 2
                ## prompt
                if data_tool[key]["tool_plan"][1]["tool"] in ["Highlight", "SceneGraph", "MaskFocus"]:
                    tool = data_tool[key]["tool_plan"][1]["tool"]
                else:
                    tool = "None"
                task_2_eva = Mask_Guided_IE_Task_2_evaluation.replace("{instruction}", instruction)
                tool_text = get_tool_text(tool, data[key]["vision_input"], log_path)
                task_2_eva = task_2_eva.replace("{tool_text}", tool_text)
                ## prompt
                ## image
                img_links = [url2path(data[key]["vision_input"][0], Image_Root), url2path(data[key]["vision_input"][1], Image_Root)]
                if tool == "Highlight":
                    img_1 = load_image(url2path(data[key]["vision_input"][0], Image_Root))
                    img_2 = load_image(url2path(data[key]["vision_input"][1], Image_Root))
                    img_1 = grding(img_1, instruction, "highlight")  
                    img_2 = grding(img_2, instruction, "highlight")  
                    img_links = [f"data:image/jpeg;base64,{encode_pil_image(img_1)}", f"data:image/jpeg;base64,{encode_pil_image(img_2)}"] 
                if tool == "MaskFocus":
                    img_1, img_2 = imgs_diff(url2path(data[key]["vision_input"][0], Image_Root), url2path(data[key]["vision_input"][1], Image_Root), "highlight")
                    img_links = [f"data:image/jpeg;base64,{encode_pil_image(img_1)}", f"data:image/jpeg;base64,{encode_pil_image(img_2)}"]
                ## image

                json_2 = GPTResponse2JSON(agent_run.get_result(img_links, task_2_eva))
                json_2['score'] = get_number(json_2['score'])
                log_prompt(log_path, task_2_eva)
                log_prompt(log_path, json_2)
                # Task 2

                out[key] = {}
                out[key]['score'] = [json_1['score'], json_2['score']]
                out[key]['reasoning'] = [json_1['reasoning'], json_2['reasoning']]
                out[key]['prompt_input'] = [task_1_eva, task_2_eva]
                out[key]['vision_input'] = data[key]["vision_input"]
                print(f"{key} over")
                break
            except Exception as e:
                print(f"Error: {key} evaluation failed: {e}")
                counter += 1
    
    write_json(out_path, out)



def test_Multi_Concept_IC_tool(in_path, out_path, log_path, test=None):
    data = read_json(in_path)
    out = {}
    for key in tqdm(list(data.keys())):
        if test:
            its = in_path.split("/")
            if key not in test[its[-3]][its[-2]]['sample']:
                continue
        counter = 0
        while counter < 3:
            try:
                text = data[key]["prompt_input"].replace("Text Prompt:", "").strip()
                # tool
                task_rule = Multi_Concept_IC_Rule.replace("{text}", text)
                task = task_rule + Multi_Concept_IC_Task_1 + Multi_Concept_IC_Task_2
                prompt_tool = Tool_Decide.replace("{task}", task)
                img_links = [url2path(data[key]["vision_input"][0], Image_Root), url2path(data[key]["vision_input"][1], Image_Root)]

                result = GPTResponse2JSON(agent_run.get_result(img_links, prompt_tool))
                result = check(result, ["task_id", "reasoning", "used", "tool"])
                log_prompt(log_path, prompt_tool)
                log_prompt(log_path, result)
                # tool 

                out[key] = {}
                out[key]['tool_plan'] = result
                print(f"{key} over")
                break
            except Exception as e:
                print(f"Error: {key} evaluation failed: {e}")
                counter += 1
    
    write_json(out_path, out)



def test_Multi_Concept_IC_evaluate(in_path, tool_path, out_path, log_path, test=None):
    data = read_json(in_path)
    data_tool = read_json(tool_path)
    out = {}
    for key in tqdm(list(data.keys())):
        if test:
            its = in_path.split("/")
            if key not in test[its[-3]][its[-2]]['sample']:
                continue
        counter = 0
        while counter < 3:
            try:
                text = data[key]["prompt_input"].replace("Text Prompt:", "").strip()
                img_L, img_R = split2part(url2path(data[key]["vision_input"][0], Image_Root))
                img_LR_links = [f"data:image/jpeg;base64,{encode_pil_image(img_L)}", f"data:image/jpeg;base64,{encode_pil_image(img_R)}"]
                subject_L, subject_R = data[key]["concepts"][0], data[key]["concepts"][1]
                # Task 1
                ## prompt
                if data_tool[key]["tool_plan"][0]["tool"] in ["Highlight", "SceneGraph"]:
                    tool = data_tool[key]["tool_plan"][0]["tool"]
                else:
                    tool = "None"
                task_1_eva = Multi_Concept_IC_Task_1_evaluation.replace("{text}", text)
                task_1_eva = task_1_eva.replace("{subject}", subject_L)
                tool_text = get_tool_text(tool, [img_LR_links[0], data[key]["vision_input"][1]], log_path)
                task_1_eva = task_1_eva.replace("{tool_text}", tool_text)
                ## prompt

                img_links = [img_LR_links[0], url2path(data[key]["vision_input"][1], Image_Root)]
                if tool == "Highlight":
                    img_1 = img_L
                    img_2 = load_image(url2path(data[key]["vision_input"][1], Image_Root))
                    img_1 = grding(img_1, subject_L, "highlight")  
                    img_2 = grding(img_2, subject_L, "highlight")  
                    img_links = [f"data:image/jpeg;base64,{encode_pil_image(img_1)}", f"data:image/jpeg;base64,{encode_pil_image(img_2)}"] 
                ## image

                json_1 = GPTResponse2JSON(agent_run.get_result(img_links, task_1_eva))
                json_1['score'] = get_number(json_1['score'])
                log_prompt(log_path, task_1_eva)
                log_prompt(log_path, json_1)
                # Task 1
                
                # Task 1
                ## prompt
                if data_tool[key]["tool_plan"][0]["tool"] in ["Highlight", "SceneGraph"]:
                    tool = data_tool[key]["tool_plan"][0]["tool"]
                else:
                    tool = "None"
                task_1_eva_ = Multi_Concept_IC_Task_1_evaluation.replace("{text}", text)
                task_1_eva_ = task_1_eva_.replace("{subject}", subject_R)
                tool_text = get_tool_text(tool, [img_LR_links[1], data[key]["vision_input"][1]], log_path)
                task_1_eva_ = task_1_eva_.replace("{tool_text}", tool_text)
                ## prompt
                ## image
                img_links = [img_LR_links[1], url2path(data[key]["vision_input"][1], Image_Root)]
                if tool == "Highlight":
                    img_1 = img_R
                    img_2 = load_image(url2path(data[key]["vision_input"][1], Image_Root))
                    img_1 = grding(img_1, subject_R, "highlight")  
                    img_2 = grding(img_2, subject_R, "highlight")  
                    img_links = [f"data:image/jpeg;base64,{encode_pil_image(img_1)}", f"data:image/jpeg;base64,{encode_pil_image(img_2)}"] 
                ## image
               
                json_2 = GPTResponse2JSON(agent_run.get_result(img_links, task_1_eva_))
                json_2['score'] = get_number(json_2['score'])
                log_prompt(log_path, task_1_eva_)
                log_prompt(log_path, json_2)
                # Task 1
                
                # Task 2
                ## prompt
                if data_tool[key]["tool_plan"][1]["tool"] in ["Highlight", "SceneGraph"]:
                    tool = data_tool[key]["tool_plan"][1]["tool"]
                else:
                    tool = "None"
                task_2_eva = Multi_Concept_IC_Task_2_evaluation.replace("{text}", text)
                tool_text = get_tool_text(tool, [data[key]["vision_input"][1]], log_path)
                task_2_eva = task_2_eva.replace("{tool_text}", tool_text)
                ## prompt
                ## image
                img_links = [url2path(data[key]["vision_input"][1], Image_Root)]
                if tool == "Highlight":
                    img_2 = load_image(url2path(data[key]["vision_input"][1], Image_Root))
                    img_2 = grding(img_2, text, "highlight")  
                    img_links = [f"data:image/jpeg;base64,{encode_pil_image(img_2)}"]
                ## image

                json_3 = GPTResponse2JSON(agent_run.get_result(img_links, task_2_eva))
                json_3['score'] = get_number(json_3['score'])
                log_prompt(log_path, task_2_eva)
                log_prompt(log_path, json_3)
                # Task 2

                out[key] = {}
                out[key]['score'] = [json_1['score'], json_2['score'], json_3['score']]
                out[key]['reasoning'] = [json_1['reasoning'], json_2['reasoning'], json_3['reasoning']]
                out[key]['prompt_input'] = [task_1_eva, task_1_eva_, task_2_eva]
                out[key]['vision_input'] = data[key]["vision_input"]
                print(f"{key} over")
                break
            except Exception as e:
                print(f"Error: {key} evaluation failed: {e}")
                counter += 1
    
    write_json(out_path, out)



def test_Text_Guided_IG_tool(in_path, out_path, log_path, test=None):
    data = read_json(in_path)
    out = {}
    for key in tqdm(list(data.keys())):
        if test:
            its = in_path.split("/")
            if key not in test[its[-3]][its[-2]]['sample']:
                continue
        counter = 0
        while counter < 3:
            try:
                text = data[key]["prompt_input"].replace("Text Prompt:", "").strip()
                # tool
                task_rule = Text_Guided_IG_Rule.replace("{text}", text)
                task = task_rule + Text_Guided_IG_Task_1
                prompt_tool = Tool_Decide.replace("{task}", task)
                img_links = [url2path(data[key]["vision_input"][0], Image_Root)]

                result = GPTResponse2JSON(agent_run.get_result(img_links, prompt_tool))
                result = check(result, ["task_id", "reasoning", "used", "tool"])
                log_prompt(log_path, prompt_tool)
                log_prompt(log_path, result)
                # tool 

                out[key] = {}
                out[key]['tool_plan'] = result
                print(f"{key} over")
                break
            except Exception as e:
                print(f"Error: {key} evaluation failed: {e}")
                counter += 1
    
    write_json(out_path, out)



def test_Text_Guided_IG_evaluate(in_path, tool_path, out_path, log_path, test=None):
    data = read_json(in_path)
    data_tool = read_json(tool_path)
    out = {}
    for key in tqdm(list(data.keys())):
        if test:
            its = in_path.split("/")
            if key not in test[its[-3]][its[-2]]['sample']:
                continue
        counter = 0
        while counter < 3:
            try:
                text = data[key]["prompt_input"].replace("Text Prompt:", "").strip()
                # Task 1
                ## prompt
                if data_tool[key]["tool_plan"][0]["tool"] in ["Highlight", "SceneGraph"]:
                    tool = data_tool[key]["tool_plan"][0]["tool"]
                else:
                    tool = "None"
                task_1_eva = Text_Guided_IG_Task_1_evaluation.replace("{text}", text)
                tool_text = get_tool_text(tool, data[key]["vision_input"], log_path)
                task_1_eva = task_1_eva.replace("{tool_text}", tool_text)
                ## prompt
                ## image
                img_links = [url2path(data[key]["vision_input"][0], Image_Root)]
                if tool == "Highlight":
                    img_1 = load_image(url2path(data[key]["vision_input"][0], Image_Root))
                    img_1 = grding(img_1, text, "highlight")  
                    img_links = [f"data:image/jpeg;base64,{encode_pil_image(img_1)}"] 
                
                ## image
        
                json_1 = GPTResponse2JSON(agent_run.get_result(img_links, task_1_eva))
                json_1['score'] = get_number(json_1['score'])
                log_prompt(log_path, task_1_eva)
                log_prompt(log_path, json_1)
                # Task 1

                out[key] = {}
                out[key]['score'] = [json_1['score']]
                out[key]['reasoning'] = [json_1['reasoning']]
                out[key]['prompt_input'] = [task_1_eva]
                out[key]['vision_input'] = data[key]["vision_input"]
                print(f"{key} over")
                break
            except Exception as e:
                print(f"Error: {key} evaluation failed: {e}")
                counter += 1
    
    write_json(out_path, out)
    


def test_Control_Guided_IG_tool(in_path, out_path, log_path, test=None):
    data = read_json(in_path)
    out = {}
    for key in tqdm(list(data.keys())):
        if test:
            its = in_path.split("/")
            if key not in test[its[-3]][its[-2]]['sample']:
                continue
        counter = 0
        while counter < 3:
            try:
                text = data[key]["prompt_input"].replace("Text Prompt:", "").strip()
                # tool
                task_rule = Control_Guided_IG_Rule.replace("{text}", text)
                task = task_rule + Control_Guided_IG_Task_1 + Control_Guided_IG_Task_2
                prompt_tool = Tool_Decide.replace("{task}", task)
                img_links = [url2path(data[key]["vision_input"][0], Image_Root), url2path(data[key]["vision_input"][1], Image_Root)]
        
                result = GPTResponse2JSON(agent_run.get_result(img_links, prompt_tool))
                result = check(result, ["task_id", "reasoning", "used", "tool"])
                log_prompt(log_path, prompt_tool)
                log_prompt(log_path, result)
                # tool 

                out[key] = {}
                out[key]['tool_plan'] = result
                print(f"{key} over")
                break
            except Exception as e:
                print(f"Error: {key} evaluation failed: {e}")
                counter += 1
    
    write_json(out_path, out)



def test_Control_Guided_IG_evaluate(in_path, tool_path, out_path, log_path,test=None):
    data = read_json(in_path)
    data_tool = read_json(tool_path)
    out = {}
    for key in tqdm(list(data.keys())):
        if test:
            its = in_path.split("/")
            if key not in test[its[-3]][its[-2]]['sample']:
                continue
        counter = 0
        while counter < 3:
            try:
                text = data[key]["prompt_input"].replace("Text Prompt:", "").strip()
                # Task 1
                ## prompt
                if data_tool[key]["tool_plan"][0]["tool"] in ["Highlight", "SceneGraph"]:
                    tool = data_tool[key]["tool_plan"][0]["tool"]
                else:
                    tool = "None"
                task_1_eva = Control_Guided_IG_Task_1_evaluation
                tool_text = get_tool_text(tool, data[key]["vision_input"], log_path)
                task_1_eva = task_1_eva.replace("{tool_text}", tool_text)
                ## prompt
                ## image
                img_links = [url2path(data[key]["vision_input"][0], Image_Root), url2path(data[key]["vision_input"][1], Image_Root)]
                if tool == "Highlight":
                    img_1 = load_image(url2path(data[key]["vision_input"][0], Image_Root))
                    img_2 = load_image(url2path(data[key]["vision_input"][1], Image_Root))
                    img_1 = grding(img_1, text, "highlight")  
                    img_2 = grding(img_2, text, "highlight")  
                    img_links = [f"data:image/jpeg;base64,{encode_pil_image(img_1)}", f"data:image/jpeg;base64,{encode_pil_image(img_2)}"] 
                ## image
       
                json_1 = GPTResponse2JSON(agent_run.get_result(img_links, task_1_eva))
                json_1['score'] = get_number(json_1['score'])
                log_prompt(log_path, task_1_eva)
                log_prompt(log_path, json_1)
                # Task 1
                
                # Task 2
                ## prompt
                if data_tool[key]["tool_plan"][1]["tool"] in ["Highlight", "SceneGraph"]:
                    tool = data_tool[key]["tool_plan"][1]["tool"]
                else:
                    tool = "None"
                task_2_eva = Control_Guided_IG_Task_2_evaluation.replace("{text}", text)
                tool_text = get_tool_text(tool, [data[key]["vision_input"][1]], log_path)
                task_2_eva = task_2_eva.replace("{tool_text}", tool_text)
                ## prompt
                ## image
                img_links = [url2path(data[key]["vision_input"][1], Image_Root)]
                if tool == "Highlight":
                    img_2 = load_image(url2path(data[key]["vision_input"][1], Image_Root))
                    img_2 = grding(img_2, text, "highlight")  
                    img_links = [f"data:image/jpeg;base64,{encode_pil_image(img_2)}"] 
                ## image
            
                json_2 = GPTResponse2JSON(agent_run.get_result(img_links, task_2_eva))
                json_2['score'] = get_number(json_2['score'])
                log_prompt(log_path, task_2_eva)
                log_prompt(log_path, json_2)
                # Task 2

                out[key] = {}
                out[key]['score'] = [json_1['score'], json_2['score']]
                out[key]['reasoning'] = [json_1['reasoning'], json_2['reasoning']]
                out[key]['prompt_input'] = [task_1_eva, task_2_eva]
                out[key]['vision_input'] = data[key]["vision_input"]
                print(f"{key} over")
                break
            except Exception as e:
                print(f"Error: {key} evaluation failed: {e}")
                counter += 1
    
    write_json(out_path, out)



def test_Subject_Driven_IG_tool(in_path, out_path, log_path, test=None):
    data = read_json(in_path)
    out = {}
    for key in tqdm(list(data.keys())):
        if test:
            its = in_path.split("/")
            if key not in test[its[-3]][its[-2]]['sample']:
                continue
        counter = 0
        while counter < 3:
            try:
                text = data[key]["prompt_input"].replace("Text Prompt:", "").strip()
                # tool
                task_rule = Subject_Driven_IG_Rule.replace("{text}", text)
                task = task_rule + Subject_Driven_IG_Task_1 + Subject_Driven_IG_Task_2
                prompt_tool = Tool_Decide.replace("{task}", task)
                img_links = [url2path(data[key]["vision_input"][0], Image_Root), url2path(data[key]["vision_input"][1], Image_Root)]
               
                result = GPTResponse2JSON(agent_run.get_result(img_links, prompt_tool))
                result = check(result, ["task_id", "reasoning", "used", "tool"])
                log_prompt(log_path, prompt_tool)
                log_prompt(log_path, result)
                # tool 

                out[key] = {}
                out[key]['tool_plan'] = result
                print(f"{key} over")
                break
            except Exception as e:
                print(f"Error: {key} evaluation failed: {e}")
                counter += 1
    
    write_json(out_path, out)



def test_Subject_Driven_IG_evaluate(in_path, tool_path, out_path, log_path, test=None):
    data = read_json(in_path)
    data_tool = read_json(tool_path)
    out = {}
    for key in tqdm(list(data.keys())):
        if test:
            its = in_path.split("/")
            if key not in test[its[-3]][its[-2]]['sample']:
                continue
        counter = 0
        while counter < 3:
            try:
                text = data[key]["prompt_input"].replace("Text Prompt:", "").strip()
                subject = data[key]["subject"]
                # Task 1
                ## prompt
                if data_tool[key]["tool_plan"][0]["tool"] in ["Highlight", "SceneGraph"]:
                    tool = data_tool[key]["tool_plan"][0]["tool"]
                else:
                    tool = "None"
                task_1_eva = Subject_Driven_IG_Task_1_evaluation.replace("{subject}", subject)
                tool_text = get_tool_text(tool, data[key]["vision_input"], log_path)
                task_1_eva = task_1_eva.replace("{tool_text}", tool_text)
                ## prompt
                ## image
                img_links = [url2path(data[key]["vision_input"][0], Image_Root), url2path(data[key]["vision_input"][1], Image_Root)]
                if tool == "Highlight":
                    img_1 = load_image(url2path(data[key]["vision_input"][0], Image_Root))
                    img_2 = load_image(url2path(data[key]["vision_input"][1], Image_Root))
                    img_1 = grding(img_1, text, "highlight")  
                    img_2 = grding(img_2, text, "highlight")  
                    img_links = [f"data:image/jpeg;base64,{encode_pil_image(img_1)}", f"data:image/jpeg;base64,{encode_pil_image(img_2)}"] 
                ## image
                
                json_1 = GPTResponse2JSON(agent_run.get_result(img_links, task_1_eva))
                json_1['score'] = get_number(json_1['score'])
                log_prompt(log_path, task_1_eva)
                log_prompt(log_path, json_1)
                # Task 1
                
                # Task 2
                ## prompt
                if data_tool[key]["tool_plan"][1]["tool"] in ["Highlight", "SceneGraph"]:
                    tool = data_tool[key]["tool_plan"][1]["tool"]
                else:
                    tool = "None"
                task_2_eva = Subject_Driven_IG_Task_2_evaluation.replace("{text}", text)
                tool_text = get_tool_text(tool, [data[key]["vision_input"][1]], log_path)
                task_2_eva = task_2_eva.replace("{tool_text}", tool_text)
                ## prompt
                ## image
                img_links = [url2path(data[key]["vision_input"][1], Image_Root)]
                if tool == "Highlight":
                    img_2 = load_image(url2path(data[key]["vision_input"][1], Image_Root))
                    img_2 = grding(img_2, text, "highlight")  
                    img_links = [f"data:image/jpeg;base64,{encode_pil_image(img_2)}"] 
                ## image

                json_2 = GPTResponse2JSON(agent_run.get_result(img_links, task_2_eva))
                json_2['score'] = get_number(json_2['score'])
                log_prompt(log_path, task_2_eva)
                log_prompt(log_path, json_2)
                # Task 2

                out[key] = {}
                out[key]['score'] = [json_1['score'], json_2['score']]
                out[key]['reasoning'] = [json_1['reasoning'], json_2['reasoning']]
                out[key]['prompt_input'] = [task_1_eva, task_2_eva]
                out[key]['vision_input'] = data[key]["vision_input"]
                print(f"{key} over")
                break
            except Exception as e:
                print(f"Error: {key} evaluation failed: {e}")
                counter += 1
    
    write_json(out_path, out)




    
def evaluate(in_path, tool_out_path, tool_log_path, eva_out_path, eva_log_path, test):
    if "ImagenHub_Control-Guided_IG" in in_path:
        test_Control_Guided_IG_tool(in_path, tool_out_path, tool_log_path, test)
        test_Control_Guided_IG_evaluate(in_path, tool_out_path, eva_out_path, eva_log_path, test)
    if "ImagenHub_Mask-Guided_IE" in in_path:
        test_Mask_Guided_IE_tool(in_path, tool_out_path, tool_log_path, test)
        test_Mask_Guided_IE_evaluate(in_path, tool_out_path, eva_out_path, eva_log_path, test)
    if "ImagenHub_Multi-Concept_IC" in in_path:
        test_Multi_Concept_IC_tool(in_path, tool_out_path, tool_log_path, test)
        test_Multi_Concept_IC_evaluate(in_path, tool_out_path, eva_out_path, eva_log_path, test)
    if "ImagenHub_Subject-Driven_IE" in in_path:
        test_Subject_Driven_IE_tool(in_path, tool_out_path, tool_log_path, test)
        test_Subject_Driven_IE_evaluate(in_path, tool_out_path, eva_out_path, eva_log_path, test)
    if "ImagenHub_Subject-Driven_IG" in in_path:
        test_Subject_Driven_IG_tool(in_path, tool_out_path, tool_log_path, test)
        test_Subject_Driven_IG_evaluate(in_path, tool_out_path, eva_out_path, eva_log_path, test)
    if "ImagenHub_Text-Guided_IE" in in_path:
        test_Text_Guided_IE_tool(in_path, tool_out_path, tool_log_path, test)
        test_Text_Guided_IE_evaluate(in_path, tool_out_path, eva_out_path, eva_log_path, test)
    if "ImagenHub_Text-Guided_IG" in in_path:
        test_Text_Guided_IG_tool(in_path, tool_out_path, tool_log_path, test)
        test_Text_Guided_IG_evaluate(in_path, tool_out_path, eva_out_path, eva_log_path, test)



import os
agent_run = LlavaNext()
Image_task_path = "PATH_TO_ImageHub_DATA"
test = read_json("test_40p.json")
# test=None
for task in os.listdir(Image_task_path):
    task_path = os.path.join(Image_task_path, task)
    models = os.listdir(task_path)
    print(models)
    for dir in models:
        if dir!="input" and dir!="token":
            tool_log_path = f"{task_path}_{dir}_tool_test_40p.txt"
            tool_out_path = f"{task_path}/{dir}/SC_tool_test_40p.json"
            eva_log_path = f"{task_path}_{dir}_eva_test_40p.txt"
            eva_out_path = f"{task_path}/{dir}/SC_eva_test_40p.json"
            in_path = f"{task_path}/{dir}/in.json"
            evaluate(in_path, tool_out_path, tool_log_path, eva_out_path, eva_log_path, test)

