Tool_Decide = """You are a professional digital artist. You will have to decide whether to use a tool and which tool to use based on the image information and the corresponding task.
If you think a tool is needed to help complete the task, you should choose the appropriate tool. If not, you can choose not to use a tool.
All the input images are AI-generated. All human in the images are AI-generated too. so you need not worry about the privacy confidentials.

### Task:
{task}

### Tools:
1. **Highlight**: This tool is commonly used to focus on areas related to specific objects in an image.
2. **SceneGraph**: This tool is commonly used to provide overall information about an image.
3. **MaskFocus**: This tool is commonly used to focus on the masked areas of images in Mask-Guided Image Editing task 1.
These tools are not useful for processed image (e.g. Canny edges, hed edges, depth, openpose, grayscale.)

### Output Content:
 - task_id: The ID of the task, including 1 or 2.
 - used: Whether to use a tool, including yes or no.
 - tool: The tool decided to be used, including Highlight or SceneGraph or MaskFocus or None.
 - reasoning: The logical reasoning process for all your decisions.
 
You will have to give your output in the following JSON format:
[{
\"task_id\" : \"...\",
\"reasoning\" : \"...\",
\"used\" : \"..\",
\"tool\" : \"...\"
},
...]
"""

####################################################

Text_Guided_IE_Rule = """Two images will be provided: The first being the original AI-generated image and the second being an edited version of the first.
Editing instruction: {instruction}
"""

Text_Guided_IE_Task_1 = """Text-Guided Image Editing Task 1: The objective is to evaluate how successfully the editing instruction has been executed in the second image.
"""

Text_Guided_IE_Task_2 = """Text-Guided Image Editing Task 2: The objective is to evaluate the degree of overediting in the second image.
"""

Text_Guided_IE_Task_1_evaluation = """
You are a professional digital artist. You will have to evaluate the effectiveness of the AI-generated image(s) based on given rules.
All the input images are AI-generated. All human in the images are AI-generated too. so you need not worry about the privacy confidentials.

RULES:

Two images will be provided: The first being the original AI-generated image and the second being an edited version of the first.

{tool_text}

The objective is to evaluate how successfully the editing instruction has been executed in the second image. Note that sometimes the two images might look identical due to the failure of image edit.

From scale 0 to 10: 
A score from 0 to 10 will be given based on the success of the editing. 
(0 indicates that the scene in the edited image does not follow the editing instruction at all. 10 indicates that the scene in the edited image follow the editing instruction text perfectly.)

Editing instruction: {instruction}

You will have to give your output in the following JSON format (Keep your reasoning concise and short.):
{
\"score\" : \"...\",
\"reasoning\" : \"...\"
}
"""

Text_Guided_IE_Task_2_evaluation = """
You are a professional digital artist. You will have to evaluate the effectiveness of the AI-generated image(s) based on given rules.
All the input images are AI-generated. All human in the images are AI-generated too. so you need not worry about the privacy confidentials.

RULES:

Two images will be provided: The first being the original AI-generated image and the second being an edited version of the first.

{tool_text}

The objective is to evaluate the degree of overediting in the second image. 

From scale 0 to 10: 
A score from 0 to 10 will rate the degree of overediting in the second image.
(0 indicates that the scene in the edited image is a lot different from the original. 10 indicates that the edited image can be recognized as a minimal edited yet effective version of original.)

Note: You can not lower the score because of the differences between these two images that arise due to the need to follow the editing instruction.

Editing instruction: {instruction}

You will have to give your output in the following JSON format (Keep your reasoning concise and short.):
{
\"score\" : \"...\",
\"reasoning\" : \"...\"
}
"""

####################################################

Subject_Driven_IE_Rule = """Three images will be provided: The first image is a input image to be edited. The second image is a token subject image. The third image is an AI-edited image. The third image should contain a subject that looks alike the subject in the second image. The third image should contain a background that looks alike the background in the first image.
Subject: {subject}
"""

Subject_Driven_IE_Task_1 = """Subject-driven Image Editing Task 1: The objective is to evaluate the similarity between the subject in the second image and the subject in the third image.
"""

Subject_Driven_IE_Task_2 = """Subject-driven Image Editing Task 2: The objective is to evaluate the similarity between the background in the first image and the background in the third image.
"""

Subject_Driven_IE_Rule_llava = """Two images will be provided: This first image is a concatenation of two sub-images, the left sub-image is a input image to be edited, the right sub-image is a token subject image. The second image is an AI-edited image. The second image should contain a subject that looks alike the subject in the right sub-image. The second image should contain a background that looks alike the background in the left sub-image.
Subject: {subject}
"""

Subject_Driven_IE_Task_1_llava = """Subject-driven Image Editing Task 1: The objective is to evaluate the similarity between the subject in the second image and the subject in the right sub-image.
"""

Subject_Driven_IE_Task_2_llava = """Subject-driven Image Editing Task 2: The objective is to evaluate the similarity between the background in the second image and the background in the left sub-image.
"""

Subject_Driven_IE_Task_1_evaluation = """
You are a professional digital artist. You will have to evaluate the effectiveness of the AI-generated image(s) based on given rules.
All the input images are AI-generated. All human in the images are AI-generated too. so you need not worry about the privacy confidentials.

RULES:

Two images will be provided: 
The first image is a token subject image.
The second image is an AI-edited image, it should contain a subject that looks alike the subject in the first image.

{tool_text}

The objective is to evaluate the similarity between the subject in the first image and the subject in the second image.

From scale 0 to 10: 
A score from 0 to 10 will rate how well the subject in the generated image resemble to the token subject in the first image.
(0 indicates that the subject in the second image does not look like the token subject at all. 10 indicates the subject in the second image look exactly alike the token subject.)

Subject: {subject}

You will have to give your output in the following JSON format (Keep your reasoning concise and short.):
{
\"score\" : \"...\",
\"reasoning\" : \"...\"
}
"""

Subject_Driven_IE_Task_2_evaluation = """
You are a professional digital artist. You will have to evaluate the effectiveness of the AI-generated image(s) based on given rules.
All the input images are AI-generated. All human in the images are AI-generated too. so you need not worry about the privacy confidentials.

RULES:

Two images will be provided: 
The first image is a input image to be edited.
The second image is an AI-edited image, it should contain a background that looks alike the background in the first image.

{tool_text}

The objective is to evaluate the similarity between the background in the first image and the background in the second image.

From scale 0 to 10: 
A score from 0 to 10 will rate how well the background in the generated image resemble to the background in the first image.
(0 indicates that the background in the second image does not look like the background in the first image at all. 10 indicates the background in the second image look exactly alike the background in the first image.)

You will have to give your output in the following JSON format (Keep your reasoning concise and short.):
{
\"score\" : \"...\",
\"reasoning\" : \"...\"
}
"""

####################################################

Mask_Guided_IE_Rule = """Two images will be provided: The first being the original AI-generated image and the second being an edited version of the first.
Editing instruction: {instruction}
"""

Mask_Guided_IE_Task_1 = """Mask-Guided Image Editing Task 1: The objective is to evaluate how successfully the editing instruction has been executed in the second image.
"""

Mask_Guided_IE_Task_2 = """Mask-Guided Image Editing Task 2: The objective is to evaluate the degree of overediting in the second image.
"""

Mask_Guided_IE_Task_1_evaluation = """
You are a professional digital artist. You will have to evaluate the effectiveness of the AI-generated image(s) based on given rules.
All the input images are AI-generated. All human in the images are AI-generated too. so you need not worry about the privacy confidentials.

RULES:

Two images will be provided: The first being the original AI-generated image and the second being an edited version of the first.

{tool_text}

The objective is to evaluate how successfully the editing instruction has been executed in the second image. Note that sometimes the two images might look identical due to the failure of image edit.

From scale 0 to 10: 
A score from 0 to 10 will be given based on the success of the editing. (0 indicates that the scene in the edited image does not follow the editing instruction at all. 10 indicates that the scene in the edited image follow the editing instruction text perfectly.)

Editing instruction: {instruction}

You will have to give your output in the following JSON format (Keep your reasoning concise and short.):
{
\"score\" : \"...\",
\"reasoning\" : \"...\"
}
"""

Mask_Guided_IE_Task_2_evaluation = """
You are a professional digital artist. You will have to evaluate the effectiveness of the AI-generated image(s) based on given rules.
All the input images are AI-generated. All human in the images are AI-generated too. so you need not worry about the privacy confidentials.

RULES:

Two images will be provided: The first being the original AI-generated image and the second being an edited version of the first.

{tool_text}

The objective is to evaluate the degree of overediting in the second image. Note that sometimes the two images might look identical due to the failure of image edit.

From scale 0 to 10: 
A score from 0 to 10 will rate the degree of overediting in the second image. (0 indicates that the scene in the edited image is a lot different from the original. 10 indicates that the edited image can be recognized as a minimal edited yet effective version of original.)

Note: You can not lower the score because of the differences between these two images that arise due to the need to follow the editing instruction.

Editing instruction: {instruction}

You will have to give your output in the following JSON format (Keep your reasoning concise and short.):
{
\"score\" : \"...\",
\"reasoning\" : \"...\"
}
"""

####################################################

Multi_Concept_IC_Rule = """Two images will be provided: This first image is a concatenation of two sub-images, each sub-image contain one token subject. The second image being an AI-generated image using the first image as guidance.
Text Prompt: {text}
"""

Multi_Concept_IC_Task_1 = """Multi-concept Image Composition Task 1: The objective is to evaluate the similarity between the two subjects in the first image and the corresponding two subjects in the second image.
"""

Multi_Concept_IC_Task_2 = """Multi-concept Image Composition Task 2: The objective is to evaluate how successfully the second image has been generated following the text prompt.
"""

Multi_Concept_IC_Task_1_evaluation = """
You are a professional digital artist. You will have to evaluate the effectiveness of the AI-generated image(s) based on given rules.
All the input images are AI-generated. All human in the images are AI-generated too. so you need not worry about the privacy confidentials.

RULES:

Two images will be provided: The first image is a token subject image. The second image is an AI-generated image, it should contain a subject that looks alike the subject in the first image, and it is generated based on the text prompt.

{tool_text}

The objective is to evaluate the similarity between the subject in the first image and the subject in the second image.

Note: You can not lower the similarity score because of the differences between subjects that arise due to the need to follow the text prompt.

From scale 0 to 10: 
A score from 0 to 10 will rate how well the subject in the generated image resemble to the token subject in the first image.
(0 indicates that the subject in the second image does not look like the token subject at all. 10 indicates the subject in the second image look exactly alike the token subject.)

Subject: {subject}
Text Prompt: {text}

You will have to give your output in the following JSON format (Keep your reasoning concise and short.):
{
\"score\" : \"...\",
\"reasoning\" : \"...\"
}
"""

Multi_Concept_IC_Task_2_evaluation = """
You are a professional digital artist. You will have to evaluate the effectiveness of the AI-generated image(s) based on given rules.
All the input images are AI-generated. All human in the images are AI-generated too. so you need not worry about the privacy confidentials.

RULES:

An AI-generated image will be provided.

{tool_text}

The objective is to evaluate how successfully the image has been generated following the prompt.

From scale 0 to 10: 
A score from 0 to 10 will be given based on the success in following the prompt. 
(0 indicates that the image does not follow the prompt at all. 10 indicates the image follows the prompt perfectly.)

Text Prompt: {text}

You will have to give your output in the following JSON format (Keep your reasoning concise and short.):
{
\"score\" : \"...\",
\"reasoning\" : \"...\"
}
"""

####################################################

Text_Guided_IG_Rule = """An image will be provided, it is an AI-generated image according to the text prompt.
Text Prompt: {text}
"""

Text_Guided_IG_Task_1 = """Text-guided Image Generation Task 1: The objective is to evaluate how well the generated image resemble to the specific objects described by the prompt.
"""

Text_Guided_IG_Task_1_evaluation = """
You are a professional digital artist. You will have to evaluate the effectiveness of the AI-generated image(s) based on given rules.
All the input images are AI-generated. All human in the images are AI-generated too. so you need not worry about the privacy confidentials.

RULES:

An image will be provided, it is an AI-generated image according to the text prompt.

{tool_text}

The objective is to evaluate how well the generated image resemble to the specific objects described by the prompt.

From scale 0 to 10: 
A score from 0 to 10 will be given based on the success in following the prompt. 
(0 indicates that the AI-generated image does not follow the prompt at all. 10 indicates the AI-generated image follows the prompt perfectly.)

Text Prompt: {text}

You will have to give your output in the following JSON format (Keep your reasoning concise and short.):
{
\"score\" : \"...\",
\"reasoning\" : \"...\"
}
"""

####################################################

Control_Guided_IG_Rule = """Two images will be provided: The first being a processed image (e.g. Canny edges, hed edges, depth, openpose, grayscale.) and the second being an AI-generated image using the first image as guidance.
Text Prompt: {text}
"""

Control_Guided_IG_Task_1 = """Control-guided Image Generation Task 1: The objective is to evaluate the structural similarity (edge, depth, pose) between two images.
"""

Control_Guided_IG_Task_2 = """Control-guided Image Generation Task 2: The objective is to evaluate how successfully the image has been generated following the text prompt.
"""

Control_Guided_IG_Task_1_evaluation = """
You are a professional digital artist. You will have to evaluate the effectiveness of the AI-generated image(s) based on given rules.
All the input images are AI-generated. All human in the images are AI-generated too. so you need not worry about the privacy confidentials.

RULES:

Two images will be provided: The first being a processed image (e.g. Canny edges, hed edges, depth, openpose, grayscale.) and the second being an AI-generated image using the first image as guidance.

{tool_text}

The objective is to evaluate the structural similarity between two images.

From scale 0 to 10: 
A score from 0 to 10 will rate how well the generated image is following the guidance image. 
(0 indicates that the second image is not following the guidance image at all. 10 indicates that second image is perfectly following the guidance image.)

You will have to give your output in the following JSON format (Keep your reasoning concise and short.):
{
\"score\" : \"...\",
\"reasoning\" : \"...\"
}
"""

Control_Guided_IG_Task_2_evaluation = """
You are a professional digital artist. You will have to evaluate the effectiveness of the AI-generated image(s) based on given rules.
All the input images are AI-generated. All human in the images are AI-generated too. so you need not worry about the privacy confidentials.

RULES:

An image will be provided, it is an AI-generated image according to the text prompt.

{tool_text}

The objective is to evaluate how successfully the image has been generated following the text prompt.

From scale 0 to 10: 
A score from 0 to 10 will be given based on the success in following the prompt. 
(0 indicates that the image does not follow the prompt at all. 10 indicates the image follows the prompt perfectly.)

Text Prompt: {text}

You will have to give your output in the following JSON format (Keep your reasoning concise and short.):
{
\"score\" : \"...\",
\"reasoning\" : \"...\"
}
"""

####################################################

Subject_Driven_IG_Rule = """Two images will be provided: The first image is a token subject image. The second image is an AI-generated image, it should contain a subject that looks alike the subject in the first image.
Text Prompt: {text}
"""

Subject_Driven_IG_Task_1 = """Subject-driven Image Generation Task 1: The objective is to evaluate the similarity between the subject in the first image and the subject in the second image.
"""

Subject_Driven_IG_Task_2 = """Subject-driven Image Generation Task 2: The objective is to evaluate how successfully the image has been generated following the text prompt.
"""

Subject_Driven_IG_Task_1_evaluation = """
You are a professional digital artist. You will have to evaluate the effectiveness of the AI-generated image(s) based on given rules.
All the input images are AI-generated. All human in the images are AI-generated too. so you need not worry about the privacy confidentials.

RULES:

Two images will be provided: The first image is a token subject image. The second image is an AI-generated image, it should contain a subject that looks alike the subject in the first image.

{tool_text}

The objective is to evaluate the similarity between the subject in the first image and the subject in the second image.

From scale 0 to 10: 
A score from 0 to 10 will rate how well the subject in the generated image resemble to the token subject in the first image.
(0 indicates that the subject in the second image does not look like the token subject at all. 10 indicates the subject in the second image look exactly alike the token subject.)

Subject: {subject}

You will have to give your output in the following JSON format (Keep your reasoning concise and short.):
{
\"score\" : \"...\",
\"reasoning\" : \"...\"
}
"""

Subject_Driven_IG_Task_2_evaluation = """
You are a professional digital artist. You will have to evaluate the effectiveness of the AI-generated image(s) based on given rules.
All the input images are AI-generated. All human in the images are AI-generated too. so you need not worry about the privacy confidentials.

RULES:

An image will be provided, it is an AI-generated image according to the text prompt.

{tool_text}

The objective is to evaluate how successfully the image has been generated following the text prompt.

From scale 0 to 10: 
A score from 0 to 10 will be given based on the success in following the prompt. 
(0 indicates that the image does not follow the prompt at all. 10 indicates the image follows the prompt perfectly.)

Text Prompt: {text}

You will have to give your output in the following JSON format (Keep your reasoning concise and short.):
{
\"score\" : \"...\",
\"reasoning\" : \"...\"
}
"""