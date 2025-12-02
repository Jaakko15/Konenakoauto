from ultralytics import YOLO

# 1) Vaihda tähän valmiin mallin painot
MODEL_PATH = r"C:\Users\Pelikone\Documents\csvCarNew\runs\detect\train4\weights\best.pt"

# 2) Vaihda tähän data.yaml
DATA_PATH = r"C:\Users\Pelikone\Documents\csvCarNew\Self-driving.v3-x1.yolov8\data.yaml"

def main():
    model = YOLO(MODEL_PATH)
    model.val(
        data=DATA_PATH,
        imgsz=1280,   # sama kuin train_customissa
        device="0",   # sama GPU
        workers=2,
        plots=True    # PR-käyrät, confusion matrix jne.
    )

if __name__ == "__main__":
    main()