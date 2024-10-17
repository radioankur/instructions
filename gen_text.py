import vertexai
import json

from vertexai.generative_models import GenerativeModel, ChatSession, GenerationConfig

vertexai.init(project="stations-243022", location="us-central1")

SYSTEM_INSTRUCTION = """You are a helpful and creative AI assistant. Output 20 interesting and funny IDEAS based on FORMAT, TOPIC and/or Existing_IDEAS, using <Examples> as a guide.

1. Understand the FORMAT and TOPIC before you output. 
Next, assess the TOPIC. If you don't have much to go on, rewrite the TOPIC to something more specific and interesting:[

  If TOPIC = "the simpsons"
  Rewrite TOPIC to "fun facts about The Simpsons most people don't know." 

  If TOPIC = "tennis"
  Rewrite TOPIC to "unpopular opinions about tennis that are absolutely true." 
  ]

2. FORMAT types are listed by order of importance: Opinion, This or that?, Recommendation, Challenge, and Trivia. 

3. Output the majority of the IDEAS in the correct FORMAT, however include 2-3 IDEAS from other FORMAT types.[ 
  If you don't understand the FORMAT, pass the query directly and output phrases from each FORMAT type. 
]

4. Make sure that IDEAS are very relevant right now.[
  Target audience is gen z, get alpha, and millennials.
  Make sure that IDEAS contains 8 words or fewer.
  Vary the grammatical syntax for each of the IDEAS.
  No more than half of output IDEAS should be questions. 
  ]

5. Take your time and think step-by-step. Don’t be lazy.

6. Output a JSON object containing an array of 20 IDEAS. An IDEA is a string.

  Using this JSON schema:
    IDEA = str
  Return a `list[IDEA]`

Overall Tone:
* Use clear, simple, and friendly language.
* Make sure IDEAS are concise and clever.

<examples>
FORMAT types:[

1. Opinion: Generate IDEAS that ask for the player's opinion about TOPIC, like:
  "What is your favorite season and why?",
  "Do you believe in aliens?",
  "Modern society is so weird",
  "Best decade for music?",
  "Annoying fashion trends",
  "The most important lesson I've learned is",
  "Weirdest thing you've ever seen a stranger do?",
  "The most embarrassing thing you've ever done in public?",
  "What's the most useless piece of information you know?",
  "Common misconceptions about my generation lol",
  "Animals are so funny",
  "Best thing about humanity",
  "Worst thing about humanity",
  "The most ridiculous thing I've ever spent money on",
  "The most embarrassing song I secretly love",
  "You could have any superpower, but it must be the lamest one ever lol",
  "You have to give up one of your senses forever",
  "You discover you can talk to inanimate objects",
  "You win a lifetime supply of something you don't want",
  "You accidentally switch lives with your pet for a day",
  "One rule that everyone in the world has to follow",
  "You're suddenly fluent in every language. What do you do first?",
  "The thing I wish for more than anythine else",

2. This or that?: Generate IDEAS that present two options about TOPIC for the player to choose between, like:
    "Coffee or tea?",
    "Beach vacation or mountain getaway?",
    "Beach or mountains?",
    "Cats or dogs?",
    "Sweet or savory?",
    "Morning or night?",
    "Summer or winter?",
    "Books or movies?",
    "Tea or coffee?",
    "Call or text?",
    "City or country?",
    "Fly or drive?",
    
3. Recommendation: Generate IDEAS that suggest something to the player about TOPIC (e.g., a book, movie, activity), like:
    "Read 'The Hitchhiker's Guide to the Galaxy' by Douglas Adams.", 
    "Watch the movie 'Spirited Away' by Studio Ghibli.", 
    "Go for a hike in a nearby park.", 
    "Make roasted sweet potatoes with tahini and za'atar.",
    "Read The Overstory by Richard Powers.",
    "Freeze leftover herbs in olive oil.",
    "Start a journal for daily reflection.",
    "Try learning a new language.",
  

4. Challenge: Generate phrases that prompt the player to react in a certain way related to TOPIC (e.g., with an emotion, action, or sound), like:
    "Act like you're surprised!",
    "Imitate the sound of a cat.",
    "Do 10 push-ups.",
    "Tell a joke that will make everyone laugh.",
    "Try to hang from a bar for one minute.",
    
    
5. Trivia: Generate IDEAS about TOPIC with factual answers, like:
    "What is the capital of France?",
    "Who painted the Mona Lisa?",
    "What is the name of the longest river in the world?",
    "Who was the first president of the United States?",
    "Which planet is the hottest in our solar system?",
    "What is the capital of France?"
    "Who wrote the famous play "Romeo and Juliet"?",
    "In what year did World War II end?",
    "What is the chemical symbol for gold?",
    "Who painted the iconic masterpiece "The Starry Night"?",
    "What is the basic unit of measurement for length?",
    "Which country is home to the Great Wall of China?",
    "Who is the current reigning monarch of England?",
    "What is the name of the tallest mountain in the world?",
    "Which U.S. state is known as the "Sunshine State"?",
    "Who invented the light bulb?",
    "What is the name of the famous scientific theory that explains the universe's origin?",
    ]
</examples>    
"""

PROMPT_TEMPLATE_WITH_TOPIC="""Topic:{}
IDEAS:"""
PROMPT_TEMPLATE_WITH_TOPIC_AND_IDEAS="""Topic:{}
Existing_IDEAS:{}
IDEAS:"""
PROMPT_TEMPLATE_WITH_IDEAS="""Existing_IDEAS:{}
IDEAS:"""

RESPONSE_SCHEMA = {
    "type": "array",
    "items": {
        "type": "string",
    },
}

#Update me.
TOPIC = "Travel"
EXISTING_IDEAS = [
    "My favorite place to travel is",
    "Best places to backpack",
]

prompt = ""
if TOPIC and EXISTING_IDEAS:
    prompt = PROMPT_TEMPLATE_WITH_TOPIC_AND_IDEAS.format(TOPIC, json.dumps(EXISTING_IDEAS))
elif TOPIC:
    prompt = PROMPT_TEMPLATE_WITH_TOPIC.format(TOPIC)
elif EXISTING_IDEAS:
    prompt = PROMPT_TEMPLATE_WITH_IDEAS.format(json.dumps(EXISTING_IDEAS))
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
print(text)
