import cv2 
import numpy as np 
import sys 
sys.path.append('../')
from utils import get_center_bbox, measure_distance, convert_pixel_to_meter
import sre_constants
import matplotlib.pyplot as plt

INTERNAL_WIDTH = 8.23
WIDTH = 10.97
LENGTH = 23.76
SERVICE_LENGTH = 6.4
ALLY_DIFFERENCE = 1.37



class MiniCourt:
    def __init__(self, keys):
        self.buffer = 50
        self.length = 450
        self.width = 250
        self.padding = 20
        self.keys = keys

    def draw_court(self, output_frames, ball_detections, player_detection, player_detect, ball_hit_frames):
        key = self.keys
        

        pt1 = [int(key[0]), int(key[1])-100]
        pt2 = [int(key[2]), int(key[3])-100]
        pt3 = [int(key[4]), int(key[5])]
        pt4 = [int(key[6]), int(key[7])]
    
        points = np.float32([pt1, pt2, pt3, pt4])
        landmarks = np.float32([[0,0], [340, 0], [0, 540], [340, 540]])
        matrix = cv2.getPerspectiveTransform(points, landmarks)
        pt = np.dot(matrix, [pt3[0], pt3[1], 1])/np.dot(matrix[2],[pt3[0], pt3[1], 1])
        p = (int(pt[0]), int(pt[1]))


        mini_background = np.zeros((540, 340, 3), np.uint8)
        speed_bg = np.zeros((300, 400, 3), dtype= np.uint8)

        mini_background[:] = 255
        mini_background = self.draw_thresh(mini_background, p)
        
        
        output_video_frames = []
        track_frame = mini_background.copy()

        count_frames = 0
        cur_ball = 0
        prev_ball = (0,0)
        f = 0
        web_location = 500

        player_1_speed = 0
        player_2_speed = 0
        ball_shot_speed = 0

        
        for frame, ball_dict, player, in zip(output_frames, ball_detections, player_detection):          
            track_frame = mini_background.copy()

            alpha = 0.5

            x1, y1, x2, y2 = ball_dict[1]
            x_ball_center , y_ball_center = get_center_bbox(ball_dict[1])
            cur_ball = (x_ball_center , y_ball_center)
            
            if count_frames % 2 == 0:
              time = (2/24)
              mps, kph = self.speed(prev_ball, cur_ball, time)
              ball_shot_speed = kph


            p = np.dot(matrix, [x_ball_center, y_ball_center, 1])/np.dot(matrix[2],[x_ball_center, y_ball_center, 1])
            point = (int(p[0]), int(p[1])-50)
            cv2.circle(track_frame, point, 10, (0, 255, 0), cv2.FILLED)

            if count_frames in ball_hit_frames: 
                f = count_frames
                ball_hit = cur_ball

            if abs(y2 - web_location) < 10:
                print(y2 - web_location)
                time = abs(f-count_frames)/24
                if time == 0:
                    continue
                print(f, count_frames)
                mps, kph = self.speed(ball_hit, cur_ball, time)
                
                if y2 - web_location < 0:
                    player_2_speed = kph
                if y2 - web_location > 0:
                    player_1_speed = kph
                
            for track_id, bbox in player.items():
                if track_id in player_detect:
                  x1, y1, x2, y2 = bbox    
                  x_player_center , y_player_center = get_center_bbox(bbox)  
                  player = (x_player_center, y_player_center)          
                  if track_id == 2:
                     y_center = y2   
                  p = np.dot(matrix, [x_player_center, y_player_center, 1])/np.dot(matrix[2],[x_player_center, y_player_center, 1])
                  point = (int(p[0]), int(p[1]))
                  cv2.circle(track_frame, point, 10, (0, 255, 0), cv2.FILLED)
                  
            tf = frame.copy()
            tf[50:590,1540:1880] = track_frame
            frame = cv2.addWeighted(tf, alpha, frame, 1 - alpha, 0)


            speed_background = speed_bg.copy() 
            speed_background = self.draw_speed(speed_bg, player_1_speed, player_2_speed, ball_shot_speed)

            overlay = frame.copy()
            overlay[620: 920, 1480:1880] = speed_background

            overlay = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            output_video_frames.append(overlay)

            count_frames = count_frames + 1
            prev_ball = cur_ball

        return output_video_frames

    def draw_thresh(self, background, pt1):
        cv2.line(background, (27,33), (312, 33), (0, 0, 255), 2)
        cv2.line(background, (27,503), (312, 503), (0, 0, 255), 2)
        cv2.line(background, (27,33), (27, 503), (0, 0, 255), 2)
        cv2.line(background, (312,33), (312, 503), (0, 0, 255), 2)
        
        cv2.line(background, (62,33), (62, 503), (0, 0, 255), 2)
        cv2.line(background, (277,33), (277, 503), (0, 0, 255), 2)
        cv2.line(background, (62,130), (277, 130), (0, 0, 255), 2)
        cv2.line(background, (62,400), (277, 400), (0, 0, 255), 2)

        cv2.line(background, (27,268), (312, 268), (255, 255, 0), 2)

        return background
    
    def draw_speed(self, window, player_1_speed, player_2_speed, ball_shot_speed):
        win = window.copy()

        cv2.putText(win, 'Ball Speed', (150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(win, 'Speed', (20, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(win, f'{ball_shot_speed} km/hr', (150, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.putText(win, 'player 1', (150, 144), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(win, 'player 2', (300, 144), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.putText(win, 'Player Vav', (20, 196), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(win, f'{player_1_speed} km/hr', (150, 196), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(win, f'{player_2_speed} km/hr', (300, 196), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return win
    
    def speed(self, point1, point2, time):
            d_pixels = measure_distance(point1, point2)
            d_meters = convert_pixel_to_meter(d_pixels)

            speed_meters_per_sec = int(d_meters/time) 
            speed_kilo_per_hour = int(speed_meters_per_sec *3.6)

            return speed_meters_per_sec, speed_kilo_per_hour

