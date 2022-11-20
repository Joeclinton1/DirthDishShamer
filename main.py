import os
from bot import DishDetectorBlocking
from dotenv import load_dotenv
import pyttsx3
import random

load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")


bot = DishDetectorBlocking(DISCORD_TOKEN)

def shoutORsiren():
    voice = [0, 1]
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    ran = random.choice(voice)
    engine.setProperty('voice', voices[ran].id)
    engine.say("Shame! Shame! Dishes still remain!")
    engine.runAndWait()
