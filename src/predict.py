from ultralytics import YOLO



def predict_stream(run=False):
    while run:
        model = YOLO("src/best.pt")
        model.predict(source='1', show= True, save = False, conf= 0.8)
        
if __name__ =='__main__':
    predict_stream()
# model.export(format="onnx" )