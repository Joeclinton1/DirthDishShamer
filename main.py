import os
from shamers import discord_shamer, audio_shamer
from logic import World
from dotenv import load_dotenv
import cv2
from ML.objrec import ObjectDetector
from ML.facerec import FaceRec

# load_dotenv()
# DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

# discord_bot = discord_shamer.DishDetectorBlocking(DISCORD_TOKEN)
audio_shamer = audio_shamer.AudioShamer()

facerec = FaceRec()
objrec = ObjectDetector()
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
    obj_df = objrec.recognize(frame, True)
    # obj_df.show()
    # cv2.imshow("vid", frame)
    # count += 1

    # update world state
    is_dirty_dish_placed = world.new_frame(obj_df, frame)

    if is_dirty_dish_placed:
        # get name of shameful
        people_df = obj_df[obj_df["class"] == 0].iloc[0]
        bbox = list(map(int, [people_df["xmin"], people_df["xmax"], people_df["ymin"], people_df["ymax"]]))
        face_name = facerec.face_rec(frame[bbox[2]:bbox[3], bbox[0]:bbox[1]])

        # shame the shameful
        print("SHAME! SHAME UPON", face_name)
        audio_shamer.shame()
        # discord_bot.shame()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
