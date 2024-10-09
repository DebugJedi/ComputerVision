from ultralytics import YOLO

model = YOLO("src//best.pt")

model.train(data = "src/dataset.yaml", imgsz = 640, batch = 8, 
            epochs = 100, workers = 1, device = 'cpu')