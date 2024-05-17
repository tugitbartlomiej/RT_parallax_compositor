from ultralytics import YOLO

model = YOLO('yolov8l-seg.pt')

model.export(format='onnx')