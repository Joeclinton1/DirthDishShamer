# import cv2
# import torch
# import torchvision
import pandas as pd
import objrec
from facerec import face_rec

#model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
# # model.load_weights("weightdirectory")
# # weights = ResNet50_Weights.DEFAULT
# # model = resnet50(weights=weights)

def crop(img, coords):
    # x1, x2, y1, y2 = bbox[0], bbox[2], bbox[1], bbox[3]
    img = img[coords[1]:coords[3], coords[0]:coords[2]]
    return img

class Person:
    def __init__(self, name, discordID):
        self.name = name
        self.id = discordID



class Human:
    def __init__(self, xmin, ymin, xmax, ymax):
        self.location = (xmin, ymin, xmax, ymax)
        self.identity = None
        self.dishes = []
        self.lifetime = 0
        self.deathtime = -1

    def enter(self, ):
        

class Dish:
    def __init__(self, xmin, ymin, xmax, ymax):
        self.location = (xmin, ymin, xmax, ymax)
        self.bringer = None
        self.lifetime = 0
        self.deathtime = -1

def attempt_ID(image, human):
    local = human.location
    img_crop = crop(image, local)
    out = objrec.rec_objs(img_crop)
    df = out.pandas().xyxy[0]
    for row in df.values:
        if row[-1]=="person" and row[-3] > 0.7:
            face_name = face_rec(img_crop)
            if face_name != "unknown":
                human.name = face_name

def distance(obj1, obj2):
    pass

def get_nearest(obj, obj_list):
    min_dist = float("inf")
    closest = None
    for i in obj_list:
        # IRL we should replace the distance measures with a proper one
        d = distance(i, obj)
        if d < min_dist:
            min_dist = d
            closest = i
    return closest, min_dist

unowned_dishes = []
dishes_in_frame = []
humans_in_frame = []
timers = []

def object_enter_frame(obj):
    if isinstance(obj, Human):
        humans_in_frame.append(obj)
        attempt_ID(obj)
        for dish in unowned_dishes:
            obj.dishes.append(dish)
            dish.bringer = obj

    if isinstance(obj, Dish):
        dishes_in_frame.append(obj)
        if len(humans_in_frame) == 0:
            unowned_dishes.append(obj)
        else:
            h = get_nearest(obj, humans_in_frame)
            obj.bringer = h
            h.dishes.append(obj)

def object_leave_frame(obj):
    if isinstance(obj, Human):
        for dish in obj.dishes:
            timers.append((dish, 5)) # 5 is the number of frames before we declare the dish abandoned
        humans_in_frame.remove(obj)
    
    if isinstance(obj, Dish):
        if obj.bringer is None:
            unowned_dishes.remove(obj)
        else:
            obj.bringer.dishes.remove(obj)
        dishes_in_frame.remove(obj)


class World():
    def __init__(self):
        self.humans = set()
        self.dishes = set()
        # might have to fill these with the actual values of the first frame
        self.prev_frame_df = pd.DataFrame(columns=["xmin","ymin","xmax","ymax","confidence","class","name"])
        self.prev_frame_img = None

    def new_frame(self, frame_df: pd.DataFrame, frame_img):
        # get the similarity data
        pass
        

def per_frame(frame: pd.DataFrame, prev_state):

    for h in humans_in_frame:
        if h.id is None:
            attempt_ID(h)

    for _ in range(len(timers)):
        d,t = timers.pop(0)
        if t == 0 and d in dishes_in_frame:
            print("Shame! Shame! Shame!", d.bringer.id.name, "left a dirty dish on the counter!")
        else:
            timers.append((d, t-1))

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




