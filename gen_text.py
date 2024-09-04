import vertexai

from vertexai.generative_models import GenerativeModel, ChatSession

vertexai.init(project="stations-243022", location="us-central1")

SYSTEM_INSTRUCTION = """Output 20 interesting, surprising, funny, and thought-provoking questions/musings based on <topic>, using <examples> as a guide.
Some examples of <topic> include 'The Olympics', 'travel hacks', 'world trivia', but it could be anything.
Make sure output are very relevant right now.
Target audience is gen z, get alpha, and millennials.
Make <output> questions/musings short and pithy.
Take your time and think step-by-step. Don’t be lazy.
Make <output> a JSON object containing an array of 20 questions. A question is a string.

  Using this JSON schema:
    Question = str
  Return a `list[Question]`

Overall Tone:
* Use clear, simple, and friendly language.
* Be thought-provoking and unexpected.
* Make your output entries concise and clever.

<Examples>
Topic:Culture & Society
Questions:[
  "What social norm needs to be abolished?",
  "What's the weirdest thing about modern society?",
  "What's the most overrated movie of all time?",
  "What's the most underrated movie of all time?",
  "What's the best decade for music?",
  "What's the most annoying fashion trend?",
  "What's the best thing about being an adult?",
  "What's the worst thing about being an adult?",
  "What's the most important lesson you've learned in life?",
  "What's the weirdest thing you've ever seen a stranger do?",
  "What's the most embarrassing thing you've ever done in public?",
  "What's the funniest thing you've ever seen a kid do?",
  "What's the best piece of advice you've ever received?",
  "What's the worst piece of advice you've ever received?",
  "What's the most important thing in a relationship?",
  "What's the funniest thing you've ever heard a child say?",
  "What's the weirdest tradition your family has?",
  "What's the best part about being part of a group?",
  "What's the worst part about being part of a group?",
  "What's the most important thing you look for in a friend?",
  "What's the most annoying thing that people do in public?",
  "What's the most ridiculous law you know of?",
  "What's the most useless piece of information you know?",
  "What's the strangest thing you've ever seen someone collect?",
  "What's the most common misconception about your generation?",
  "What's the best thing about being alive right now?",
  "What's the worst thing about being alive right now?",
  "What's the most important thing to teach children?",
  "What's the funniest thing you've ever seen an animal do?",
  "What's the best thing about humanity?",
  "What's the worst thing about humanity?",
  "What's the most ridiculous thing you've ever spent money on?",
  "What's the most embarrassing song you secretly love?",
  "What's the most overrated tourist attraction you've ever been to?",
  "What's the most underrated tourist attraction you've ever been to?"
]
Topic:Hypotheticals & Scenarios
Questions:[
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
]
Topic:Just Plain Weird
Questions:[
  "What's the weirdest thing you've ever eaten?",
  "What's the most ridiculous thing you've ever done to get someone's attention?",
  "What's the weirdest dream you've ever had?",
  "What's the most embarrassing song you secretly love to sing?",
  "What's the strangest talent you have?",
  "What's the weirdest thing you find attractive in a person?",
  "What's the most ridiculous phobia you have?",
  "What's the weirdest collection you've ever seen?",
  "What's the most embarrassing thing you've ever done in front of your crush?",
  "What's the weirdest thing you've ever said in your sleep?",
  "What's the most unusual food combination you enjoy?",
  "What's the weirdest habit you have?",
  "What's the most bizarre nickname you've ever had?",
  "What's the weirdest thing you've ever seen an animal do?",
  "What's the most embarrassing thing you've ever worn in public?",
  "What's the weirdest thing you've ever done for money?",
  "What's the strangest thing you've ever received as a gift?",
  "What's the weirdest thing you've ever found in your house?",
  "What's the most ridiculous thing you've ever believed as a child?",
  "What's the strangest dream you've ever had that felt weirdly real?",
  "What's the weirdest thing you do when you're home alone?",
  "What's the most embarrassing thing you've ever done while trying to be cool?",
  "What's the weirdest thing you've ever said to a stranger?",
  "What's the most ridiculous thing you've ever lied about?",
  "What's the strangest thing you've ever seen someone wearing in public?",
  "What's the weirdest thing you've ever done on a dare?",
  "What's the most embarrassing thing you've ever done at a party?",
  "What's the weirdest thing you've ever done to try and fit in?",
  "What's the most ridiculous excuse you've ever used to get out of something?",
  "What's the weirdest thing you've ever said to your parents?",
  "What's the strangest thing you've ever seen someone do on a date?",
  "What's the most embarrassing song you know all the words to?",
  "What's the weirdest thing you've ever done in front of a mirror?",
  "What's the most ridiculous thing you've ever done to try and impress a crush?",
  "What's the weirdest question you've ever been asked? (You're reading it!)"
]
</Examples>
"""

#Update me.
TOPIC = "Pets"
PROMPT_TEMPLATE="""Topic:{}
Questions:"""

def get_chat_response(chat, prompt):
    text_response = []
    responses = chat.send_message(prompt, stream=True)
    for chunk in responses:
        text_response.append(chunk.text)
    return "".join(text_response)

text_model = GenerativeModel(
    model_name="gemini-1.5-flash-001",
    system_instruction=SYSTEM_INSTRUCTION,
    generation_config={"response_mime_type": "application/json"}
)

chat = text_model.start_chat()
prompt = PROMPT_TEMPLATE.format(TOPIC)
text = get_chat_response(chat, prompt)
print(text)