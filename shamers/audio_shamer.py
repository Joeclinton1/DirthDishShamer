import pyttsx3
import random


def shame(img, name):
    voice = [0, 1]
    engine = pyttsx3.init()
    voice = random.choice(engine.getProperty('voices'))
    engine.setProperty('voice', voice)
    engine.say(f"Shame! Shame! Dishes still remain!")
    engine.runAndWait()
