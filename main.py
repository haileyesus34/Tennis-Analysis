from utils import read_video, save_video
import cv2
from tracker import Player_detection, Ball_detection
from courtline_detector import KeyPointExtraction
from mini_court import MiniCourt
import matplotlib.pyplot as plt

def main():

    print('Hello World')

    input_video_path = 'input_videos/tennis.mp4'
    output_video_path = 'output_videos/tennis.mp4'

    video_frames = read_video(input_video_path)

    player_tracker = Player_detection('models\\yolov8l.pt')
    ball_tracker = Ball_detection('models\\best.pt')
    kps = KeyPointExtraction('models\\model_det.pth')


    kps_detect = kps.predict(video_frames[0])
    person_detection = player_tracker.detect_frames(video_frames, read_stub= True, stub_path= 'tracker_stub/player_detection.pk1')
    ball_detections = ball_tracker.detect_frames(video_frames, read_stub = True, stub_path = 'tracker_stub/ball_detection.pk1')
    

    mini_court = MiniCourt(kps_detect)
    player_detect = player_tracker.choose_players(kps_detect, person_detection)
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)
    ball_hit_frames = ball_tracker.get_show_frame(ball_detections)

    output_video_frames, mini_ball_frame = ball_tracker.draw_bbox(video_frames, ball_detections)
    output_video_frames = kps.draw_keypoints_video(output_video_frames, kps_detect)
    output_video_frames = player_tracker.draw_bbox(output_video_frames, person_detection, player_detect)
    output_video_frames = mini_court.draw_court(output_video_frames, ball_detections,person_detection, player_detect, ball_hit_frames)

    for i, frame in enumerate(output_video_frames):   
            cv2.putText(frame, f'Frame: {str(i)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 2)
            
           

    save_video(output_video_frames, output_video_path)


if __name__ == '__main__':
    main()

