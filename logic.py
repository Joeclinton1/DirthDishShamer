import cv2
import pandas as pd
import numpy as np
from skimage.metrics import structural_similarity

TBL_THRESHOLD = 200

def crop(img, bbox):
    # x1, x2, y1, y2 = bbox[0], bbox[2], bbox[1], bbox[3]
    img = img[bbox[2]:bbox[3], bbox[0]:bbox[1]]
    return img


def get_similarity(dish, dishes2):
    # set weights
    w1, w2, w3 = 0.8, 0.2, 0

    similarity = [0] * len(dishes2)
    for i, dish2 in enumerate(dishes2):
        # calc similarity
        pos1, pos2 = [np.array(d.pos) for d in [dish, dish2]]
        area1, area2 = [d.area for d in [dish, dish2]]
        # ssim = structural_similarity(dish.img, dish2.img)
        max_pos = (dish.img.shape[1] ** 2 + dish2.img.shape[2]**2) ** 0.5
        max_area = dish.img.shape[0] * dish2.img.shape[1]
        max_ssim = 1

        diff = np.linalg.norm(pos2 - pos1) / max_pos * w1 + (area2 - area1) / max_area * w2 # + (1 - ssim / max_ssim) * w3
        similarity[i] = 1 - diff / (w1 + w2 + w3)
    return similarity


def obj_in_table_region(obj, table, simple=False):
    if simple:
        return obj.pos[1] > TBL_THRESHOLD
    else:
        return cv2.pointPolygonTest(table, obj.pos, False)


def extract_table_contour(frame):
    img_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = 100
    ret, thresh_img = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours[0]


class Obj():
    def __init__(self, bbox, frame_img):
        self.bbox = bbox
        self.img = crop(frame_img, bbox)
        self.limbo = 0
        self.pos = (bbox[0] + bbox[1]) / 2, (bbox[2] + bbox[3]) / 2
        self.area = (bbox[1] - bbox[0]) * (bbox[3] - bbox[2])


class World():
    def __init__(self, frame):
        self.dishes = []
        self.past_dishes = []
        self.sim_thresh = 0.8
        self.table = extract_table_contour(frame)

    def new_frame(self, objs: pd.DataFrame, frame_img):
        dishes_df = objs[~objs["class"].isin([0, 60])]

        # increment limbo for all dishes
        # remove the dish if it has been in limbo for 3 or more frames
        for dish in self.dishes:
            dish.limbo += 1
            if dish.limbo >= 3:
                del dish

        for _, dish_df in dishes_df.iterrows():
            dish_bbox = list(map(int,[dish_df["xmin"], dish_df["xmax"], dish_df["ymin"], dish_df["ymax"]]))
            dish = Obj(dish_bbox, frame_img)
            if not obj_in_table_region(dish, self.table, simple=True):
                continue

            for dishes2 in self.past_dishes[::-1]:
                if len(dishes2)==0: continue
                sim_vals = get_similarity(dish, dishes2)
                max_sim_idx = np.argmax(sim_vals)

                if sim_vals[max_sim_idx] > self.sim_thresh:
                    del dishes2[max_sim_idx]
                    self.dishes.append(dish)
                    break
            else:
                self.dishes.append(dish)
                return True

        if len(self.past_dishes) >= 2:
            self.past_dishes.pop(0)
        self.past_dishes.append(self.dishes)
        return False
