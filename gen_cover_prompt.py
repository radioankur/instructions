import vertexai
import tempfile
import json

from vertexai.generative_models import GenerativeModel, ChatSession, GenerationConfig
from vertexai.preview.vision_models import ImageGenerationModel
from PIL import Image

vertexai.init(project="stations-243022", location="us-central1")

SYSTEM_INSTRUCTION = """Output 6 prompts that could be used to generate an image using Gen AI, that relates to a input question, using <Examples> as a guide.
Make sure that each output prompt is detailed and conveys something relevant to the input question.
Take your time and think step-by-step. Don’t be lazy.
Output a JSON object containing an array of 6 prompts. A prompt is a string.

  Using this JSON schema:
    Prompt = str
  Return a `list[Prompt]`

Overall Tone:
* Use clear language.

<Examples>
Question:Fish or Birds?
Prompts:[
    "split screen with fish on one side and birds on another"
]
</Examples>
"""
_INSPO_PACKS_COVER_PROMPT_SYSTEM_INSTRUCTION = """Output 6 prompts that could be used to generate an image using Gen AI.
The background of the generated image should relate to the input title.
The foreground of the generated image to be the input title as text, centered in a bold, fun, eye catching font.
Use <Examples> as a guide.
Make sure that each output prompt is detailed and conveys something relevant to the input title.
Take your time and think step-by-step. Don’t be lazy.
Output a JSON object containing an array of 6 prompts. A prompt is a string.

  Using this JSON schema:
    Prompt = str
  Return a `list[Prompt]`

Overall Tone:
* Use clear language.

<Examples>
Title:Would you rather
Prompts:[
    "The text 'WOULD YOU RATHER' in bold text, centered over a background of question marks"
]
</Examples>
"""

RESPONSE_SCHEMA = {
    "type": "array",
    "items": {
        "type": "string",
    },
}

#Update me.
TITLE = "Pawsome Pals"
PROMPT_TEMPLATE="""Title:{}
Prompts:"""

def get_chat_response(chat, prompt):
    text_response = []
    responses = chat.send_message(prompt, stream=True)
    for chunk in responses:
        text_response.append(chunk.text)
    return "".join(text_response)

text_model = GenerativeModel(
    model_name="gemini-1.5-flash-001",
    system_instruction=SYSTEM_INSTRUCTION,
    generation_config=GenerationConfig(
        response_mime_type="application/json", response_schema=RESPONSE_SCHEMA
    )
)

chat = text_model.start_chat()
prompt = PROMPT_TEMPLATE.format(TITLE)
text = get_chat_response(chat, prompt)
print("PROMPTS:")
print(text)

imagen_model = ImageGenerationModel.from_pretrained("imagen-3.0-fast-generate-001")

image_prompts = json.loads(text)
for prompt in image_prompts:
    images = imagen_model.generate_images(
        prompt=prompt,
        number_of_images=1,
        aspect_ratio="3:4",
        safety_filter_level="block_some",
        person_generation="allow_adult",
    )
    tmp = tempfile.NamedTemporaryFile(dir='/tmp')
    images[0].save(location=tmp.name, include_generation_parameters=False)
    img = Image.open(tmp.name)
    img.show()