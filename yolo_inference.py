from ultralytics import YOLO

model = YOLO('models/yolov8l.pt')

results = model.predict('input_videos\\image.jpg', save = False)

print(results)
print(results[0].boxes[0])

#for box in results[0].boxes:
#    print(box)