import torch
import einops

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class ObjectDetector():
    def __init__(self) -> None:
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        # self.model = self.model.to(device)
    
    def recognize(self, image, show=False):
        #image = torch.tensor(image, device=device)
        #image = einops.rearrange(image, "(b h) w c -> b c h w", b=1)
        results = self.model.forward(image)#.cpu()
        # print(results.shape)
        if show: results.show()
        return results.pandas().xyxy[0]

