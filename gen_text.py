import vertexai
import json

from vertexai.generative_models import GenerativeModel, ChatSession, GenerationConfig

vertexai.init(project="stations-243022", location="us-central1")

SYSTEM_INSTRUCTION = """You are a helpful and creative AI assistant. Output 20 interesting, surprising, funny, and thought-provoking phrases based on FORMAT, TOPIC and/or EXISTING_QUESTIONS, using <Examples> as a guide.
Understand the FORMAT and TOPIC before you output. 
If no FORMAT is clear, then output phrases from each FORMAT type. 
Make sure output questions very relevant right now.
Target audience is gen z, get alpha, and millennials.
Make sure that each output question contains 8 words or fewer.
Take your time and think step-by-step. Don’t be lazy.
Output a JSON object containing an array of 20 questions. A question is a string.

  Using this JSON schema:
    Question = str
  Return a `list[Question]`

Overall Tone:
* Use clear, simple, and friendly language.
* Be thought-provoking and unexpected.
* Make sure output questions are concise and clever.

<examples>
FORMAT types:[

1. Trivia: Generate questions about TOPIC with factual answers, like:
    "What is the capital of France?",
    "Who painted the Mona Lisa?",

2. Quiz: Generate questions about TOPIC with a choice of answers (e.g., multiple choice, true/false), like:
    "Is the Earth flat? (True/False)",
    "Which of these is a primary color?", 

3. This or that?: Generate phrases that present two options for the player to choose between, like:
    "Coffee or tea?",
    "Beach vacation or mountain getaway?",
    
4. Recommendations: Generate phrases that suggest something to the player (e.g., a book, movie, activity), like:
  "Read 'The Hitchhiker's Guide to the Galaxy' by Douglas Adams.", 
  "Watch the movie 'Spirited Away' by Studio Ghibli.", 
  "Go for a hike in a nearby park.", 

5. Opinion: Generate phrases that ask for the player's opinion on a topic, like:
  "What is your favorite season and why?",
  "Do you believe in aliens?",
  "What social norm needs to be abolished?",
  "The weirdest thing about modern society?",
  "The most overrated movie of all time?",
  "The most underrated movie of all time?",
  "Best decade for music?",
  "Most annoying fashion trend?",
  "Best thing about being an adult?",
  "Worst thing about being an adult?",
  "The most important lesson you've learned in life?",
  "Weirdest thing you've ever seen a stranger do?",
  "The most embarrassing thing you've ever done in public?",
  "Funniest thing you've ever seen a kid do?",
  "The best piece of advice you've ever received?",
  "Worst piece of advice you've ever received?",
  "The most important aspect of a relationship?",
  "Funniest thing you've ever heard a child say?",
  "Weirdest family tradition?",
  "Most important thing you look for in a friend?",
  "The most annoying thing that people do in public?",
  "What's the most useless piece of information you know?",
  "The strangest thing you've ever seen someone collect?",
  "What's the most common misconception about your generation?",
  "What's the best thing about being alive right now?",
  "What's the worst thing about being alive right now?",
  "Most important thing to teach children?",
  "The funniest thing you've ever seen an animal do?",
  "Best thing about humanity?",
  "Worst thing about humanity?",
  "The most ridiculous thing you've ever spent money on?",
  "The most embarrassing song you secretly love?",
  "Most overrated tourist attraction you've ever been to?",
  "Most underrated tourist attraction you've ever been to?"
  "If you could have any superpower, but it had to be the lamest one ever, what would it be?",
  "You're stuck in an elevator with your celebrity crush, but it's also filled with your least favorite animal. What happens?",
  "You find a remote control that can rewind, pause, or fast-forward time for everyone but you. What do you do?",
  "You wake up tomorrow as the opposite gender. What's the first thing you do?",
  "If you could write a letter to your past self, what advice would you give?",
  "You have to give up one of your senses forever. Which one do you choose and why?",
  "You discover you can talk to inanimate objects. What do you say to the first object you encounter?",
  "You win a lifetime supply of something you don't want. What is it, and what do you do with it?",
  "You accidentally switch lives with your pet for a day. What happens?",
  "You can only speak in movie quotes for the rest of your life. What quotes will you be using most often?",
  "You're granted immortality, but you're stuck at your current age. How do you feel about it?",
  "You have the chance to travel to space, but you can never come back to Earth. Do you go?",
  "You can make one rule that everyone in the world has to follow. What's the rule?",
  "You're stranded on a desert island with three items from your fridge. What are they, and how do you survive?",
  "You're suddenly fluent in every language. What do you do first?",
  "You find a genie in a bottle, but they can only grant you one wish that's utterly useless. What do you wish for?",
  "You're invited to a costume party, but you can only go as something you find in your recycling bin. What's your costume?",
  "You have to live the same day over and over again for eternity. What day do you choose?",
  "You can have any fictional character as your best friend. Who do you choose and why?",
  "You find a magical remote control that can only control the weather. What kind of chaos do you unleash?",
  "You wake up to find that you've swapped bodies with your arch-nemesis. How do you handle it?",
  "You have to survive a zombie apocalypse. What's your survival strategy?",
  "You discover that you have a long-lost twin who's your polar opposite. What happens next?",
  "You can time travel, but only 5 minutes into the past. How do you use this power?",
  "You find a magical artifact that grants you the ability to read minds, but only of animals. What do you learn?",
  "You win a contest where the prize is getting to design your own country. What does it look like, and what are the laws?",
  "You're trapped in a haunted house with only your phone and a flashlight. How do you escape?",
  "You're shrunk down to the size of an ant. Where do you go and what do you do?",
  "You can only eat food that starts with the first letter of your name for the rest of your life. What are your options?",
  "You're suddenly transported into the world of your favorite video game. What's your first move?"
  "Weirdest thing you've ever eaten?",
  "The most ridiculous thing you've ever done to get someone's attention?",
  "Weirdest dream you've ever had?",
  "The most embarrassing song you secretly love to sing?",
  "What's the strangest talent you have?",
  "What's the weirdest thing you find attractive in a person?",
  "What's the most ridiculous phobia you have?",
  "Weirdest collection you've ever seen?",
  "Most embarrassing thing you've ever done in front of your crush?",
  "Weirdest thing you've ever said in your sleep?",
  "The most unusual food combination you enjoy?",
  "Weirdest habit you have?",
  "Most bizarre nickname you've ever had?",
  "Weirdest thing you've ever seen an animal do?",
  "Most embarrassing thing you've ever worn in public?",
  "Weirdest thing you've ever done for money?",
  "Strangest gift you've ever received?",
  "Weirdest thing you've ever found in your house?",
  "Most ridiculous thing you believed as a child?",
  "Strangest dream you've ever had?",
  "Weirdest thing you do that nobody knows about?",
  "Most embarrassing thing you've ever done to be cool?",
  "Weirdest thing you've ever said to a stranger?",
  "Most ridiculous thing you've ever lied about?",
  "Strangest thing you've ever seen someone wearing in public?",
  "Most embarrassing thing you've ever done at a party?",
  "WWeirdest thing you've ever done to try and fit in?",
  "What's the most ridiculous excuse you've ever used to get out of something?",
  "Weirdest thing you've ever said to your parents?",
  "Strangest thing you've ever seen someone do on a date?",
  "Most embarrassing song you know all the words to?",
  "Weirdest thing you've ever done in front of a mirror?",
  "Most ridiculous thing you've ever done to try and impress a crush?",
  "Weirdest question you've ever been asked?"

7. Challenge: Generate phrases that prompt the player to react in a certain way (e.g., with an emotion, action, or sound), like:
    "Make a surprised face!",
    "Imitate the sound of a cat.",
    "Do 10 push-ups!",
    "Tell a joke that will make everyone laugh.",
    ]
</examples>    
"""

PROMPT_TEMPLATE_WITH_TOPIC="""Topic:{}
Questions:"""
PROMPT_TEMPLATE_WITH_TOPIC_AND_QUESTIONS="""Topic:{}
Existing Questions:{}
Questions:"""
PROMPT_TEMPLATE_WITH_QUESTIONS="""Existing Questions:{}
Questions:"""

RESPONSE_SCHEMA = {
    "type": "array",
    "items": {
        "type": "string",
    },
}

#Update me.
TOPIC = "Travel"
EXISTING_QUESTIONS = [
    "Hostel or Hotel?",
    "Beach or mountains?",
]

prompt = ""
if TOPIC and EXISTING_QUESTIONS:
    prompt = PROMPT_TEMPLATE_WITH_TOPIC_AND_QUESTIONS.format(TOPIC, json.dumps(EXISTING_QUESTIONS))
elif TOPIC:
    prompt = PROMPT_TEMPLATE_WITH_TOPIC.format(TOPIC)
elif EXISTING_QUESTIONS:
    prompt = PROMPT_TEMPLATE_WITH_QUESTIONS.format(json.dumps(EXISTING_QUESTIONS))
else:
    prompt = "Questions:"

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
