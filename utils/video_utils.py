import cv2 

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    video_frames = []

    while True:
       ret, frame = cap.read()
       if ret == False:
           break
       video_frames.append(frame)
    cap.release()
    return video_frames

def save_video(video_frames, video_frame_path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_frame_path, fourcc, 24, (video_frames[0].shape[1], video_frames[0].shape[0]))

    for frame in video_frames:
        out.write(frame)
    out.release()

    print(f'video saved to {video_frame_path}')