
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6,5"
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from modelscope import Qwen2VLForConditionalGeneration
from peft import PeftModel
from qwen_vl_utils import process_vision_info
import torch
import flask
from flask import Flask, request, jsonify

# torch.cuda.empty_cache()
app = Flask(__name__)
model_dir = "path_to/Qwen2-VL-7B-Instruct"
# model_dir = "path_to/Qwen2.5-VL-7B-Instruct"
# model_dir = "path_to/Qwen2.5-VL-7B-Instruct-sft"
# model_dir = "path_to/Qwen2-VL-7B-Instruct-sft"
# lora_path = ""




model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)


# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     model_dir,
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )
# if lora_path:
#     model = PeftModel.from_pretrained(model, lora_path, torch_dtype=torch.bfloat16, device_map="auto")

model.eval()
processor = AutoProcessor.from_pretrained(model_dir)


def format(images, text):
    content = []
    for img in images:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": text})
    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]
    return messages


@app.route('/generate', methods=['POST'])
def generate():
    msg = flask.request.get_json(force=True)
    imgs = msg['imgs']
    text = msg['text']
    messages = format(imgs, text)
    ########
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,add_vision_id=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    return jsonify({"response": output_text})

    # except Exception as e:
    #     return jsonify({"error": str(e)})



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)