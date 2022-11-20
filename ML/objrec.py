import torch
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def rec_objes(image):
    return model.forward(image)

