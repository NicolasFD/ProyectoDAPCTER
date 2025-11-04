import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime
from scipy.signal import convolve2d

# === CONFIGURACIÓN ===
input_dir = "Pruebas"
ruta_base = "C:/DAPCTER/ProyectoDAPCTER"
fecha_hoy = datetime.now().strftime("%Y-%m-%d")
ruta_carpeta = os.path.join(ruta_base, fecha_hoy)

output_dir_HS = os.path.join(ruta_carpeta, "Paneles_HS")
output_dir_OK = os.path.join(ruta_carpeta, "Paneles_Sanos")
os.makedirs(output_dir_HS, exist_ok=True)
os.makedirs(output_dir_OK, exist_ok=True)

INTENSIDAD_UMBRAL = 45
VECINDAD = 5
kernel = np.ones((VECINDAD, VECINDAD), np.float32)

# === Verificación inicial ===
if not os.path.exists(input_dir):
    raise FileNotFoundError(f"No existe la carpeta {input_dir}")

imagenes = [
    os.path.join(input_dir, f)
    for f in os.listdir(input_dir)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

if not imagenes:
    print("No hay imágenes en la carpeta.")
    exit()

# === Cargar el modelo YOLO una sola vez ===
model = YOLO("best.pt")
classNames = ["Panel-Hotspots"]

# === Procesamiento ===
countHS, countOK = 0, 0

for ruta in imagenes:
    frame = cv2.imread(ruta)
    if frame is None:
        print(f"No se pudo leer {os.path.basename(ruta)}")
        continue

    # Procesar con YOLO
    results = model(frame, verbose=False)
    for res in results:
        for box in res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = classNames[cls] if cls < len(classNames) else f"Class {cls}"

            # === Recorte de ROI ===
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # Filtro local rápido usando OpenCV
            resultado = convolve2d(gray_roi, kernel, mode='same', boundary='fill', fillvalue=0)
            mean_val = np.mean(resultado)
            mask = resultado > (mean_val + 1000)
            num_pixels = np.count_nonzero(mask)

            # === Clasificación ===
            if num_pixels >= 200 and (x2 - x1) > 30 and (y2 - y1) > 30:
                filename = os.path.join(output_dir_HS, f"PanelHS_{countHS}.jpg")
                countHS += 1
            else:
                filename = os.path.join(output_dir_OK, f"Panel_{countOK}.jpg")
                countOK += 1

            cv2.imwrite(filename, roi)

print(f"\nProceso terminado.\nPaneles HS: {countHS}\nPaneles sanos: {countOK}")