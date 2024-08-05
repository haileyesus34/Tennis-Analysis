import cv2 
import numpy as np 
import pickle
from ultralytics import YOLO
import sys 
sys.path.append('../')
from utils import get_center_bbox, measure_distance


class Player_detection:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_frames(self, frames, read_stub = False, stub_path = None):
        
        if read_stub and stub_path is not None: 
            with open(stub_path, 'rb') as f:
                player_detection = pickle.load(f)
            return player_detection
        
        player_detection = []

        for frame in frames:
            results = self.detect_frame(frame)
            player_detection.append(results)

        if stub_path is not None:
            with open(stub_path, 'wb') as f: 
                pickle.dump(player_detection, f)

        return player_detection
    
    def detect_frame(self, frame):    
        results = self.model.track(frame, conf = 0.2, persist=True)[0]
        cls_names = results.names
        player_dict = {}
        
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            cls_id = box.cls.tolist()[0]
            name = cls_names[cls_id]      
            if name == 'person':  
                player_dict[track_id] = result

        return player_dict
    
    def draw_bbox(self, frames, player_detection, player_detect):

        output_video_frames = []

        for frame,  player in zip(frames,  player_detection):
            for track_id, bbox in player.items():
              if track_id in player_detect:
                x1, y1, x2, y2 = bbox               
                cv2.putText(frame, f'id: {str(track_id)}', (int(x1+10), int(y1+10)), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 1)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2) 
            
            output_video_frames.append(frame)

        return output_video_frames
                
    def choose_players(self, keys, player_detections):
        player_dict = player_detections[0]
        distance = 100000
        player_filter = []
        player_detect = []

        for track_id, player in player_dict.items():
            player_center = get_center_bbox(player)
            for i in range(0, len(keys), 2):
                x = int(keys[i])
                y = int(keys[i+1])
                point = (x,y)
                d = measure_distance(player_center, point)
                if d < distance:
                    distance = d
            player_filter.append((track_id, distance))

        player_filter.sort(key= lambda x: x[1])
        player_filter = player_filter[:2]
        player_detect = [player_filter[0][0], player_filter[1][0]]
        
        return player_detect
