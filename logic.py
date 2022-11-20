# import cv2
# import torch
# import torchvision
import pandas as pd

# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
# # model.load_weights("weightdirectory")
# # weights = ResNet50_Weights.DEFAULT
# # model = resnet50(weights=weights)

class Person:
    def __init__(self, name, discordID):
        self.name = name
        self.id = discordID

class Human:
    def __init__(self, xmin, ymin, xmax, ymax, i):
        self.location = (xmin, ymin, xmax, ymax)
        self.identity = None
        self.dishes = []
        self.lifetime = 0
        self.deathtime = -1
        self.prev_index = i

    def enter(self):
        pass

    def stay(self):
        pass

    def leave(self):
        pass

class Dish:
    def __init__(self, xmin, ymin, xmax, ymax, i):
        self.location = (xmin, ymin, xmax, ymax)
        self.bringer = None
        self.lifetime = 0
        self.deathtime = -1
        self.prev_index = i

    def enter(self):
        pass

    def stay(self):
        pass

    def leave(self):
        pass

def attempt_ID(human):
    # We've found a person in the image and cropped it down to just them, now pass it to the face recogniser
    pass

def get_similarity(df1, img1, df2, img2):
    # dummy function for Clinton's method
    pass

class World():
    def __init__(self):
        self.objects = set()
        # might have to fill these with the actual values of the first frame
        self.prev_frame_df = pd.DataFrame(columns=["xmin","ymin","xmax","ymax","confidence","class","name"])
        self.prev_frame_img = None

    def new_frame(self, frame_df: pd.DataFrame, frame_img):
        pairs = get_similarity(self.prev_frame_df, self.prev_frame_img, frame_df, frame_img)
        for (i,j) in pairs:
            if i is None:
                # Enter
                if frame_df[j, "name"] == "person":
                    # TODO find the relevant Human item (if this is a repeat person)
                    pass
                else:
                    # TODO find the relevant Dish item (if this is a repeat dish)
                    pass
            elif j is None:
                # Leave
                for o in self.objects:
                    if o.prev_index == i:
                        o.leave()
                        break
            else:
                # Stay
                for o in self.objects:
                    if o.prev_index == i:
                        o.stay()
                        break

# video_capture = cv2.VideoCapture(0)
# count = 0
# while True:
#     # Grab a single frame of video
#     ret, frame = video_capture.read()
#     # if count%4 == 0:
#     out = model.forward(frame)
#     out.show()
#     # cv2.imshow("vid", frame)
#     # count += 1
#     #out.show()
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break




