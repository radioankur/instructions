import vertexai
import json
import typing_extensions as typing

from vertexai.generative_models import GenerativeModel, ChatSession, GenerationConfig

vertexai.init(project="stations-243022", location="us-central1")

SYSTEM_INSTRUCTION = """Output 10 topic suggestions based on a list of one or more questions, using <Examples> as a guide.
Make sure output topic very relevant right now.
Target audience is gen z, get alpha, and millennials.
Make sure that each output topic contains 6 words or fewer.
Take your time and think step-by-step. Don’t be lazy.
Output a JSON object containing an array of 10 topics. A topic is a string.

  Using this JSON schema:
    Topic = str
  Return a `list[Topic]`

Overall Tone:
* Use clear, simple, and friendly language.
* Be thought-provoking and unexpected.
* Make sure output questions are concise and clever.

<Examples>
Questions:[
  "Cheetahs or leopards",
]
Topics:[
  "Big Cat Geography",
  "True Cat Facts",
  "This or That About Big Cats",
  "Favorite Cats",
  "Cat fashion",
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
QUESTIONS = [
    "What's the most populate pet in the world?",
    "Indoor pets or outdoor pets?"
]
PROMPT_TEMPLATE="""Questions:{}
Topics:"""

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
prompt = PROMPT_TEMPLATE.format(json.dumps(QUESTIONS))
text = get_chat_response(chat, prompt)
print(text)
