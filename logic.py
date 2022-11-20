# import cv2
# import torch
# import torchvision
import pandas as pd
from ML import objrec
from ML.facerec import face_rec
from skimage.metrics import structural_similarity


# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
# # model.load_weights("weightdirectory")
# # weights = ResNet50_Weights.DEFAULT
# # model = resnet50(weights=weights)

def get_bbox_from_df(df):
    return df["x_min"], df["x_max"], df["y_min"], df["y_max"]

def crop(img, bbox):
    # x1, x2, y1, y2 = bbox[0], bbox[2], bbox[1], bbox[3]
    img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    return img


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

    def enter(self, ):


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


def attempt_ID(image, human):
    local = human.location
    img_crop = crop(image, local)
    out = objrec.rec_objs(img_crop)
    df = out.pandas().xyxy[0]
    for row in df.values:
        if row[-1] == "person" and row[-3] > 0.7:
            face_name = face_rec(img_crop)
            if face_name != "unknown":
                human.name = face_name


def get_similarity(obj, img1, objs, img2):
    #set weights
    w1, w2, w3 = 0.5, 0.3, 0.2

    similarity = [0]*objs.shape[1]
    for i,obj2 in enumerate(objs):
        #calc similarity
        bboxs = [get_bbox_from_df(df) for df in [obj,obj2]]
        pos1, pos2 = [((bb[0] + bb[1]) / 2, (bb[2] + bb[3]) / 2) for bb in bboxs]
        area1, area2 = [(bb[1]-bb[0])*(bb[3]-bb[2]) for bb in bboxs]
        ssim = structural_similarity(crop(img1, bboxs[0]), crop(img2, bboxs[1]))
        max_pos= (img1.shape[1]**2+img1.shape[2])**0.5
        max_area= img1.shape[0]*img1.shape[1]
        max_ssim = 1

        diff = (pos2-pos1)/max_pos*w1+(area2-area1)/max_area*w2+(1-ssim/max_ssim)*w3
        similarity[i] = 1-diff/(w1+w2+w3)
    return similarity


class World():
    def __init__(self):
        self.objects = set()
        # might have to fill these with the actual values of the first frame
        self.prev_frame_df = pd.DataFrame(columns=["xmin", "ymin", "xmax", "ymax", "confidence", "class", "name"])
        self.prev_frame_img = None

    def new_frame(self, frame_df: pd.DataFrame, frame_img):
        pairs = get_similarity(self.prev_frame_df, self.prev_frame_img, frame_df, frame_img)
        for (i, j) in pairs:
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
