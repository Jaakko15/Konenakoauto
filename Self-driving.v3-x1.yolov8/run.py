from ultralytics import YOLO
import cv2

# ----- CONFIG -----
MODEL_PATH = r"C:\Users\Pelikone\Documents\csvCarNew\runs\detect\finetune_bigset3\weights\best.pt"
VIDEO_IN   = r"C:\Users\Pelikone\Documents\csvCarNew\tamkin.mp4"
VIDEO_OUT  = r"C:\Users\Pelikone\Documents\csvCarNew\tamkin_out.mov"

# choose what to draw:
TARGET_CLASSES = {"car", "biker", "pedestrian","truck", }

# optional: colors per class (B,G,R). Missing classes default to green.
COLORS = {
    "car": (0, 255, 0),
    "truck": (0, 200, 255),
    "biker": (255, 0, 0),
    "pedestrian": (255, 255, 0),
}
# ------------------

model = YOLO(MODEL_PATH)
names = {int(k): v for k, v in model.names.items()}

# Build list of class IDs to keep
keep_ids = [i for i,n in names.items() if n in TARGET_CLASSES]

cap = cv2.VideoCapture(VIDEO_IN)
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30
out = cv2.VideoWriter(VIDEO_OUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # run only on the classes we care about (faster and cleaner)
    results = model(frame, conf=0.35, iou=0.6, classes=keep_ids)[0]

    # draw boxes ourselves for control
    for box in results.boxes:
        cid = int(box.cls[0])
        cname = names[cid]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])

        # If user selected the generic 'trafficLight', include all its variants
        if "trafficLight" not in TARGET_CLASSES and cname.startswith("trafficLight"):
            continue
        if cname not in TARGET_CLASSES and not cname.startswith("trafficLight"):
            continue

        color = COLORS.get(cname, (0, 255, 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{cname} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y1_text = max(0, y1 - th - 6)
        cv2.rectangle(frame, (x1, y1_text), (x1 + tw + 4, y1_text + th + 4), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1_text + th + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    out.write(frame)

cap.release()
out.release()
print("Saved:", VIDEO_OUT)
