import pyttsx3
import random


class AudioShamer():
    def __init__(self):
        self.engine = pyttsx3.init()

    def shame(self):
        voice = random.choice(self.engine.getProperty('voices'))
        self.engine.setProperty('voice', voice)
        self.engine.say("Shame! Shame! Dishes still remain!")
        self.engine.runAndWait()
