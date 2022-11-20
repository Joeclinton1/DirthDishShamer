import torch

class ObjectDetector():
    def __init__(self) -> None:
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    
    def rec_objs(self, image):
        return self.model.forward(image)

