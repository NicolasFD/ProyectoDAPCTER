import cv2
from ultralytics import YOLO
import numpy as np
import os

# === Cargar imagen de prueba ===
imagen_path = "WIN_20251210_11_56_04_Pro.jpg"   # pon aquí el nombre de tu archivo
frame = cv2.imread(imagen_path)

if frame is None:
    raise FileNotFoundError("No se pudo leer la imagen, revisa la ruta.")

# === Cargar el modelo YOLO ===
model = YOLO("best.pt")

classNames = ["Panel-Hotspots"]

output_dir = "detecciones_guardadas"
os.makedirs(output_dir, exist_ok=True)

# === Procesar la imagen ===
results = model(frame, stream=True)

count=0
for res in results:
    boxes = res.boxes
    if boxes is None or len(boxes) == 0:
        continue

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = classNames[cls] if cls < len(classNames) else f"Class {cls}"

        # Paso 1: Matriz de valores
        roi = frame[y1:y2,x1:x2]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Paso 2: Media y máscara
        mean_val = np.mean(gray_roi)
        num_pixels = np.sum(gray_roi > mean_val+36)
        #mask=gray_roi > mean_val

        # Paso 3: Guardar si hay píxeles mayores a la media
        if num_pixels >= 100 and roi.size>2000:
            filename = f"{output_dir}/{label}_{int(count)}.jpg"
            cv2.imwrite(filename, roi)
            count+=1
        
        #if np.any(mask):
            #filename = f"{output_dir}/{label}_{int(count)}.jpg"
            #cv2.imwrite(filename, roi)
            #count+=1

    
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = classNames[cls] if cls < len(classNames) else f"Class {cls}"

        # Paso 1: Matriz de valores
        roi = frame[y1:y2,x1:x2]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        print(gray_roi)

        # Paso 2: Media y máscara
        mean_val = np.mean(gray_roi)
        mask = gray_roi > mean_val

        #Dibujar bounding box
        text = f"{label} {conf:.2f} Mean:{mean_val:.1f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 0, 0), 2)

# Mostrar la imagen con detecciones
cv2.imshow("Resultado", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
