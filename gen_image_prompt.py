import vertexai

from vertexai.generative_models import GenerativeModel, ChatSession, GenerationConfig

vertexai.init(project="stations-243022", location="us-central1")

SYSTEM_INSTRUCTION = """Output 5 prompts that could be used to generate an image using Gen AI, that relates to a input question, using <Examples> as a guide.
Target audience is gen z, gen alpha, and millennials.
Make sure that each output prompt is detailed and conveys something relevant to the input question.
Take your time and think step-by-step. Don’t be lazy.
Output a JSON object containing an array of 5 prompts. A prompt is a string.

  Using this JSON schema:
    Prompt = str
  Return a `list[Prompt]`

Overall Tone:
* Use clear language.
* Be thought-provoking and unexpected.

<Examples>
Question:Fish or Birds?
Prompts:[
    "split screen with fish on one side and birds on another"
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
QUESTION = "Cutest dog breed?"
PROMPT_TEMPLATE="""Question:{}
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
prompt = PROMPT_TEMPLATE.format(QUESTION)
text = get_chat_response(chat, prompt)
print(text)
