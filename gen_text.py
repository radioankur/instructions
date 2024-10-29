import vertexai
import json

from vertexai.generative_models import GenerativeModel, ChatSession, GenerationConfig

vertexai.init(project="stations-243022", location="us-central1")

def read_text_file(file_path):
  """
  Reads a text file and returns its content as a string.

  Args:
    file_path: The path to the text file.

  Returns:
    A string containing the file's content.
  """
  try:
    with open(file_path, 'r') as file:
      return file.read()
  except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    return None

SYSTEM_INSTRUCTION = read_text_file("gen_text_system_instruction.txt").replace(r"{{ prompt }}", "")

PROMPT_TEMPLATE_WITH_TOPIC="""Topic:{}
IDEAS:"""
PROMPT_TEMPLATE_WITH_TOPIC_AND_IDEAS="""Topic:{}
Existing_IDEAS:{}
IDEAS:"""
PROMPT_TEMPLATE_WITH_IDEAS="""Existing_IDEAS:{}
IDEAS:"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "rewrite_topic": { "type": "string" },
        "format": { "type": "string" },
        "ideas": { "type": "array", "items": { "type": "string" } }
    },
    "required": ["rewrite_topic", "format", "ideas"]
}

#Update me.
TOPIC = "Travel"
EXISTING_IDEAS = [
    "My favorite place to travel is",
    "Best places to backpack",
]

prompt = ""
if TOPIC and EXISTING_IDEAS:
    prompt = PROMPT_TEMPLATE_WITH_TOPIC_AND_IDEAS.format(TOPIC, ", ".join(EXISTING_IDEAS))
elif TOPIC:
    prompt = PROMPT_TEMPLATE_WITH_TOPIC.format(TOPIC)
elif EXISTING_IDEAS:
    prompt = PROMPT_TEMPLATE_WITH_IDEAS.format(", ".join(EXISTING_IDEAS))
else:
    prompt = "IDEAS:"

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
text = get_chat_response(chat, prompt)
result = json.loads(text)
for idea in result["ideas"]:
    print(idea)
