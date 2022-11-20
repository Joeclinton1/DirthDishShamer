import cv2
import pandas as pd
import numpy as np
from skimage.metrics import structural_similarity

def crop(img, bbox):
    # x1, x2, y1, y2 = bbox[0], bbox[2], bbox[1], bbox[3]
    img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    return img


def get_similarity(dish, dishes2):
    # set weights
    w1, w2, w3 = 0.5, 0.3, 0.2

    similarity = [0] * dishes2.shape[1]
    for i, dish2 in enumerate(dishes2):
        # calc similarity
        pos1, pos2 = [d.pos for d in [dish, dish2]]
        area1, area2 = [d.area for d in [dish, dish2]]
        ssim = structural_similarity(dish.img, dish2.img)
        max_pos = (dish.img.shape[1] ** 2 + dish2.img.shape[2]) ** 0.5
        max_area = dish.img.shape[0] * dish2.img.shape[1]
        max_ssim = 1

        diff = (pos2 - pos1) / max_pos * w1 + (area2 - area1) / max_area * w2 + (1 - ssim / max_ssim) * w3
        similarity[i] = 1 - diff / (w1 + w2 + w3)
    return similarity


def obj_in_table_region(obj, table):
    return cv2.pointPolygonTest(table, obj.pos, False)


def extract_table_contour(frame):
    img_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = 100
    ret, thresh_img = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


class Obj():
    def __init__(self, bbox, frame_img):
        self.bbox = bbox
        self.img = crop(frame_img, bbox)
        self.limbo = 0
        self.pos = (bbox[0] + bbox[1]) / 2, (bbox[1] + bbox[2]) / 2
        self.area = (bbox[1] - bbox[0]) * (bbox[3] - bbox[2])


class World():
    def __init__(self, frame):
        self.dishes = []
        self.past_dishes = []
        self.sim_thresh = 0.8
        self.table = extract_table_contour(frame)

    def new_frame(self, objs: pd.DataFrame, frame_img):
        dishes_df = objs[objs["class"] != 0]

        # increment limbo for all dishes
        # remove the dish if it has been in limbo for 3 or more frames
        for dish in self.dishes:
            dish.limbo += 1
            if dish.limbo >= 3:
                del dish

        for dish_df in dishes_df:
            dish_bbox = [dish_df["x-min"], dish_df["x-max"], dish_df["y_min"], dish_df["y_max"]]
            dish = Obj(dish_bbox, frame_img)
            if not obj_in_table_region(dish, self.table):
                continue

            for dishes2 in self.past_dishes:
                sim_vals, dish_ids = get_similarity(dish, dishes2)
                max_sim_idx = np.argmax(sim_vals)

                if sim_vals[max_sim_idx] > self.sim_thresh:
                    dish.limbo = 0
                    break
                else:
                    self.dishes.append(dish)
                    return True

            self.past_dishes.pop(0)
            self.past_dishes.append(self.dishes)
        return False
