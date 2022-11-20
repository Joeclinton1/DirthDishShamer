import os
from shamers import discord_shamer, audio_shamer
from logic import World
from dotenv import load_dotenv
import cv2
import torch
import torchvision

load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

discord_bot = discord_shamer.DishDetectorBlocking(DISCORD_TOKEN)
world = World()

# start main loop
video_capture = cv2.VideoCapture(0)
count = 0
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # if count%4 == 0:
    obj_df = model.forward(frame)
    obj_df.show()
    # cv2.imshow("vid", frame)
    # count += 1

    # update world state
    shameful_people = world.new_frame(obj_df, frame)

    # shame the shameful
    audio_shamer.shame()
    discord_bot.shame()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
