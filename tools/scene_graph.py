from agent.openai import GPT4o,QWen25
from util import clean_text

sgPrompt='''
For the provided image, generate only a scene graph in JSON format that includes the following:
1. Objects that are relevant to describing the image content in detail
2. Object attributes that are relevant to describing the image content in detail
3. Object relationships that are relevant to describing the image content in detail
'''

# SG_generator = QWen25()
SG_generator = GPT4o()

def sg_generate(img_links):
    prompt_content = SG_generator.prepare_prompt(img_links, sgPrompt)
    sg = SG_generator.get_result(prompt_content)
    print(sg)
    return clean_text(sg)