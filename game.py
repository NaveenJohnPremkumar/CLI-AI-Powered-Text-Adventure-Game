# import whisper
# import ollama
#from langchain.chains import ConversationChain
from langchain_ollama import OllamaLLM
#from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from ollama import generate
#from langchain.schema import SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
#from langchain.chains import create_history_aware_retriever
from PIL import Image
import base64
import io

def encode_image_to_base64(image_path):
    with Image.open(image_path) as img:
        # Convert to RGB if image is in RGBA format
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        
        # Convert image to base64
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str

def generate_story_from_image(image_path):
    img_base64 = encode_image_to_base64(image_path)
    
    prompt = """Look carefully at the provided image and create a detailed 3-paragraph background story for a text adventure game that is deeply rooted in what you observe.

    First paragraph: Analyze the visual elements, atmosphere, and details in the image to vividly describe the setting and time period. Include specific details about the environment, architecture, technology level, weather, or other notable features you can see.

    Second paragraph: Based on the mood, objects, characters, or situations depicted in the image, construct a central conflict or challenge. Focus on elements that are either directly visible or strongly suggested by the scene - perhaps a looming threat, a mysterious object, signs of recent conflict, or environmental hazards.

    Third paragraph: Drawing from the scene shown, establish the player character's role and immediate circumstances. Consider their apparent position in the image, any visible equipment or clothing they might have, and their relationship to the environment and conflict you've described.

    Keep your descriptions grounded in what you can actually see in the image. Use present tense and refer to the player as "you" or "the player". Ensure every detail you include is supported by visual evidence from the provided image."""
    
    story = ""
    inside_think = False
    
    for part in generate(
        model='bsahane/Qwen2.5-VL-7B-Instruct:Q4_K_M_benxh',
        prompt=prompt,
        images=[img_base64],
        stream=True
    ):
        response = part['response']
        if '<think>' in response:
            inside_think = True
        if '</think>' in response:
            inside_think = False
            continue
        if not inside_think:
            print(response, end="", flush=True)
            story += response
            
    return story

def generate_story_text():
    game_prompt = """
    Generate the background for a text-based adventure game. The world should be filled with mystery and challenges, and the player will uncover its history as they explore. 

    ### Structure:
    1. **Setting:** 
    - Describe a unique world (e.g., fantasy, sci-fi, post-apocalyptic).  
    - Establish the current state of the world and any major conflicts.  

    2. **Main Objective:**  
    - Define the player’s overarching goal (e.g., escape, find an artifact, solve a mystery).  

    3. **Key Locations & NPCs:**  
    - Create 3-5 major locations with distinct challenges.  
    - Introduce key NPCs and factions that shape the story.  

    4. **Dynamic Elements:**  
    - Ensure the world evolves based on the player’s choices.  
    - Leave room for AI-generated details to emerge during gameplay.  

    Keep descriptions immersive but brief. The game world should unfold naturally as the player interacts with it.
    Output should be in paragraph format. Limit the output to 3 paragraphs.
    """

    
    inside_think = False
    
    story = ""
    for part in generate(model='deepseek-r1:1.5b', prompt=game_prompt, stream=True):
        response = part['response']
        if '<think>' in response:
            inside_think = True
        if '</think>' in response:
            inside_think = False
            continue
        if not inside_think:
            print(response, end="", flush=True)
            story += response

    return story

def generate_help_options(current_context=""):
    prompt = f"""Based on the current game situation, generate 4 numbered command options that would be most relevant for the player right now.
    Each option should be a single sentence command followed by a brief description.
    Format each option on a new line starting with a number (1-4).
    Make the commands practical and useful for the current game situation.
    Keep the descriptions clear and concise.
    Only output the 4 numbered options and descriptions, nothing else.
    
    Current game context: {current_context}"""
    
    help_text = ""
    inside_think = False
    
    for part in generate(model='deepseek-r1:1.5b', prompt=prompt, stream=True):
        response = part['response']
        if '<think>' in response:
            inside_think = True
        if '</think>' in response:
            inside_think = False
            continue
        if not inside_think:
            help_text += response
            
    return help_text

def main():
    print("Welcome to the AI-Powered Adventure Game!")
    print("Type 'exit' or 'quit' to end the game.")
    print("Type 'help' to see available commands.")
    
    # Ask user for story generation preference
    while True:
        print("\nHow would you like to generate the game's background story?")
        print("1. Let the AI create a story")
        print("2. Generate story from an image")
        choice = input("Enter your choice (1 or 2): ").strip()
        
        if choice in ['1', '2']:
            break
        print("Invalid choice. Please enter 1 or 2.")
    
    # Generate background story based on user choice
    if choice == '1':
        print("\nGenerating AI story...")
        background = generate_story_text()
    else:
        while True:
            image_path = input("\nEnter the path to your image file: ").strip()
            try:
                print("\nGenerating story from image...")
                background = generate_story_from_image(image_path)
                break
            except Exception as e:
                print(f"Error loading image: {e}")
                print("Please try again with a valid image path.")
    
    # Initialize help options with the background context
    current_help_options = generate_help_options(background)
    
    llm_prompt = f"""
    You are the Game Master of a text-based adventure game.
    The player is an adventurer in a mysterious dungeon filled with secrets, treasures, and dangers.
    Your job is to describe the world, react to player actions, and make the game engaging.
    Keep responses immersive, creative, and interactive.
    
    The game starts with the following background story:
    {background}

    Remember the player's actions and use them to shape the story dynamically. Only respond when you receive a user input.
    """

    llm = OllamaLLM(model="deepseek-r1:1.5b",
                     temperature=0.1)
    
    memory = ConversationBufferMemory(return_messages=True)

    # Create the conversation chain
    prompt = ChatPromptTemplate.from_messages([
        ("system", llm_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    def show_help():
        print("\nAvailable commands:")
        print(current_help_options)
        print("\nYou can also type 'help' to see this menu again or 'exit' to quit the game.")
    
    while True:
        user_input = input(">> ")
        
        if user_input.lower() in ["exit", "quit"]:
            print("\nGame Master: Thanks for playing! Goodbye!")
            break
            
        if user_input.lower() == "help":
            show_help()
            continue
        
        ai_response = ""
        inside_think = False
        
        # Get the chat history
        chat_history = memory.load_memory_variables({})["history"]
        
        # Generate response using the chain
        for part in chain.stream({"input": user_input, "history": chat_history}):
            if '<think>' in part:
                inside_think = True
            if '</think>' in part:
                inside_think = False
                continue
            if not inside_think:
                print(part, end="", flush=True)
                ai_response += part
        
        print()  # Print a newline after the streamed response
        
        # Generate new help options based on the latest response
        current_help_options = generate_help_options(ai_response)
        
        # Save the interaction to memory
        memory.save_context({"input": user_input}, {"output": ai_response})

if __name__ == "__main__":
    main()