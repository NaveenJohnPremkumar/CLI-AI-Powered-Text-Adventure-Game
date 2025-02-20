from ollama import generate
from langchain.schema import SystemMessage


def generate_story_text():
    prompt = """Generate the background story for a text-based adventure game.
    The player takes on the role of a character on a quest to uncover the secrets of a long-forgotten world or civilization.
    The story should include the player's current situation, their motivation for embarking on the journey, 
    and what items they have at the start. The player’s goal and the mysteries they must solve should unfold over time.
    Include elements of danger, intrigue, and discovery. The adventure should offer a sense of progression and 
    mystery as the player explores the world. The response should be 3 paragraphs maximum"""
    
    inside_think = False
    
    story = ""
    for part in generate(model='deepseek-r1:1.5b', prompt=prompt, stream=True):
        response = part['response']
        if '<think>' in response:
            inside_think = True
        if '</think>' in response:
            inside_think = False
            continue
        if not inside_think:
            #print(response, end="", flush=True)
            story += response
            
        
    return story
        
    
background = generate_story_text()
llm_prompt = """
You are the Game Master of a text-based adventure game.
The player is an adventurer in a mysterious dungeon filled with secrets, treasures, and dangers.
Your job is to describe the world, react to player actions, and make the game engaging.
Keep responses immersive, creative, and interactive. I will provide the background story now. Only respond when you get
"User:" in the prompt.
""" + background


system_message = SystemMessage(content=llm_prompt)