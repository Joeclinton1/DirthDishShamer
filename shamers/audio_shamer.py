import pyttsx3
import random


def shame():
    voice = [0, 1]
    engine = pyttsx3.init()
    voice = random.choice(engine.getProperty('voices'))
    engine.setProperty('voice', voice)
    engine.say("Shame! Shame! Dishes still remain!")
    engine.runAndWait()
