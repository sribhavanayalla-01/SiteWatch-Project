from ultralytics import YOLO

# Lightweight YOLO model
model = YOLO("yolo11n.pt")

# Construction PPE dataset
model.train(
    data="construction-ppe.yaml",
    epochs=30,
    imgsz=640,
    batch=8,
    project="runs",
    name="sitewatch_ppe"
)