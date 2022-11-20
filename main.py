import os
from shamers import discord_shamer, audio_shamer
from ML import facerec as frec, objrec as orec
from logic import World
from dotenv import load_dotenv
import cv2


load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

discord_bot = discord_shamer.DishDetectorBlocking(DISCORD_TOKEN)

# Initialise ML models
facerec = frec.FaceRec()
objrec = orec.ObjectDetector()

# start main loop
video_capture = cv2.VideoCapture(0)

print(video_capture)

# Grab first frame in order to get the table
ret, frame = video_capture.read()
world = World(frame)

count = 0
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # if count%4 == 0:
    obj_df = objrec.rec_objs(frame)
    obj_df.show()
    # cv2.imshow("vid", frame)
    # count += 1

    # update world state
    is_dirty_dish_placed = world.new_frame(obj_df, frame)

    if is_dirty_dish_placed:
        # get name of shameful
        people_df = obj_df[obj_df["class"] == 0]
        bbox = [people_df["x-min"], people_df["x-max"], people_df["y_min"], people_df["y_max"]]
        face_name = facerec.face_rec(frame[bbox[1]:bbox[3], bbox[0]:bbox[2]])

        # shame the shameful
        audio_shamer.shame()
        discord_bot.shame()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
