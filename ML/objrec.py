import torch

class ObjectDetector():
    def __init__(self) -> None:
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    
    def recognize(self, image, show=False):
        results = self.model.forward(image)
        if show: results.show()
        return results.pandas().xyxy[0]

