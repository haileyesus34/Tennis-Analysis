import cv2
import pickle
from ultralytics import YOLO 
import pandas as pd
import numpy as np 
import sys 
import matplotlib.pyplot as plt
sys.path.append('../')
from utils import get_center_bbox, measure_distance, convert_pixel_to_meter

class Ball_detection:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.width = 250
        self.length = 450


    def interpolate_ball_positions(self, ball_detections):
        bbox = []
        
        bbox = [x.get(1, []) for x  in ball_detections]

        df_bbox = pd.DataFrame(bbox, columns=['x1', 'y1', 'x2', 'y2'])
        df_bbox = df_bbox.interpolate()
        df_bbox = df_bbox.bfill()

        ball_detections = [{1: b} for b in df_bbox.to_numpy().tolist()]
        
        return ball_detections
    
    def get_show_frame(self, ball_detection):
        ball_position = [x.get(1, []) for x in ball_detection]
        
        df_ball_position = pd.DataFrame(ball_position, columns=['x1', 'y1', 'x2', 'y2'])
        df_ball_position = df_ball_position.interpolate()
        df_ball_position = df_ball_position.bfill()

        df_ball_position['mid_y'] = (df_ball_position['y1'] + df_ball_position['y2']) // 2
        df_ball_position['mid_y_rolling'] = df_ball_position['mid_y'].rolling(window=5, min_periods=1, center= False).mean()

        df_ball_position['ball_hit'] = 0
        
        df_ball_position['delta_y'] = df_ball_position['mid_y_rolling'].diff()

        number_of_change_frames = 10

        for i in range(0, df_ball_position.shape[0] - number_of_change_frames):
             negative_change = df_ball_position['delta_y'].iloc[i] > 0 and df_ball_position['delta_y'].iloc[i+1] < 0
             positive_change = df_ball_position['delta_y'].iloc[i] < 0 and df_ball_position['delta_y'].iloc[i+1] > 0

             if negative_change or positive_change:
                change_count = 0
                for frame_num in range(i+1, i+number_of_change_frames):
                    negative_change_next = df_ball_position['delta_y'].iloc[i] > 0 and df_ball_position['delta_y'].iloc[frame_num] < 0
                    positive_change_next = df_ball_position['delta_y'].iloc[i] < 0 and df_ball_position['delta_y'].iloc[frame_num] > 0

                    if negative_change and negative_change_next:
                        change_count = change_count + 1
                    elif positive_change and positive_change_next:
                        change_count = change_count + 1
                if change_count >= int(number_of_change_frames*0.6): 
                   df_ball_position['ball_hit'].iloc[i] = 1
        
        ball_hit_frames = df_ball_position[df_ball_position['ball_hit'] == 1].index.tolist()
        print(ball_hit_frames)
        return ball_hit_frames
        

    def detect_frames(self, frames, read_stub = False, stub_path = None):
        ball_detection = []

        if read_stub and stub_path is not None:
            with open(stub_path, 'rb') as f: 
                ball_detection = pickle.load(f)
            return ball_detection
        
        for i, frame in enumerate(frames):
            bbox = self.detect_frame(frame)
            ball_detection.append(bbox)

        if stub_path is not None:
            with open(stub_path, 'wb') as f: 
                pickle.dump(ball_detection, f)

        return ball_detection
   
    def detect_frame(self, frame):
        ball_dict = {}
        results = self.model.predict(frame, conf = 0.15)[0]

        for  box in results.boxes:
              ball_bbox = box.xyxy.tolist()[0]
              ball_dict[1] =  ball_bbox
        return ball_dict
    
    def draw_bbox(self, video_frames, ball_detections):
        output_video_frames = []
        mini_ball_frames = []

        for frame, ball_dict in zip(video_frames, ball_detections):
                frm = frame.copy()
                x1, y1, x2, y2 = ball_dict[1]
                x_center , y_center = get_center_bbox(ball_dict[1])
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
                cv2.circle(frm, (int(x_center), int(y_center)), 20, (0, 255, 0), cv2.FILLED)
                output_video_frames.append(frame)
                mini_ball_frames.append(frm)
        
        return output_video_frames, mini_ball_frames
    
              
        
