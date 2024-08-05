import torch
import torchvision 
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import sys
sys.path.append('../')
from utils import get_center_bbox, measure_distance

class KeyPointExtraction:
    def __init__(self, model_path):
        self.model = models.resnet50(pretrained= True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 28)
        self.model.load_state_dict(torch.load(model_path, map_location = 'cpu'))

        self.transform =  transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        ])

    def predict(self, image):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_transform = self.transform(img_rgb).unsqueeze(0)

        with torch.no_grad():
            output = self.model(img_transform)

        kps = output.cpu().numpy().squeeze()

        orig_h, orig_w = img_rgb.shape[:2]
        
        kps[::2] *= orig_w/224.0
        kps[1::2] *= orig_h/224.0

        return kps
    
    def draw_keypoints(self, frame, key):
        for i in range(0, len(key), 2): 
            x = int(key[i])
            y = int(key[i+1])

            cv2.putText(frame, str(i//2), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        return frame
    
    def draw_keypoints_video(self, video_frames, keypoints):
        output_frames = []

        for frame in video_frames:
            frame = self.draw_keypoints(frame, keypoints)
            output_frames.append(frame)

        return output_frames

        