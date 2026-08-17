import ultralytics
from ultralytics import YOLO

model = YOLO("/Users/mac/Documents/Object_Detection/best.pt")
model.export(format = 'onnx', opset = 12, simplify = True, dynamic = False)