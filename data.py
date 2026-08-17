from ultralytics import YOLO
model = YOLO('/Users/mac/Documents/Deployment/Object_Detection/best.pt')
print(model.names)
