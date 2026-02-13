from ultralytics import YOLO
import os, json, shutil

VUELOS_BASE = "Capturas"
MODEL_PATH = "models/current.pt"
STATE_FILE = "train_state.json"
LOCK_FILE = "training.lock"

if os.path.exists(LOCK_FILE):
    exit()

open(LOCK_FILE, "w").close()

try:
    model = YOLO(MODEL_PATH)

    model.train(
        data="data.yaml",
        epochs=5,
        imgsz=512,
        batch=2,
        device="cpu",
        workers=2,
        lr0=5e-5,
        freeze=15,
        amp=False
    )

    shutil.copy(
        "runs/detect/train/weights/best.pt",
        MODEL_PATH
    )

    vuelos = []
    for fecha in os.listdir(VUELOS_BASE):
        ruta_fecha = os.path.join(VUELOS_BASE, fecha)
        if os.path.isdir(ruta_fecha):
            vuelos.extend(
                os.path.join(fecha, v)
                for v in os.listdir(ruta_fecha)  n    
                if v.startswith("Vuelo_")
            )

    vuelos.sort()

    with open(STATE_FILE, "w") as f:
        json.dump({"ultimo_vuelo": vuelos[-1]}, f)

finally:
    os.remove(LOCK_FILE)
