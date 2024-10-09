from ultralytics import YOLO

model = YOLO("src/best.pt")
model.predict(source='1', show= True, save = True, conf= 0.8)

model.export(format="onnx" )