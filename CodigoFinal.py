import cv2
import numpy as np
from ultralytics import YOLO
import os

# === CONFIGURACIÓN GENERAL ===
input_dir = "ImagenesPruebas"
output_dir = "detecciones_hotspots"
os.makedirs(output_dir, exist_ok=True)

# Verificar carpeta
if not os.path.exists(input_dir):
    raise FileNotFoundError(f"No existe la carpeta {input_dir}")

# Cargar modelo YOLO
model = YOLO("best.pt")
classNames = ["Panel-Hotspots"]

# === PARÁMETROS DE DETECCIÓN DE HOTSPOTS ===
THRESHOLD_RELATIVO = 0.25   # 25% sobre la media de intensidad
AREA_MINIMA = 30            # área mínima en píxeles
BLUR = 5                    # suavizado para eliminar ruido

# === PROCESAMIENTO DE IMÁGENES ===
imagenes = [f for f in os.listdir(input_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))]

if not imagenes:
    print("No hay imágenes en la carpeta.")
else:
    count = 0
    for nombre in imagenes:
        ruta = os.path.join(input_dir, nombre)
        frame = cv2.imread(ruta)
        if frame is None:
            print(f"No se pudo leer {nombre}")
            continue

        # --- Paso 1: detección con YOLO ---
        results = model(frame, stream=True)

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

                # --- Paso 2: análisis de hotspot dentro del ROI detectado ---
                roi = frame[y1:y2, x1:x2]
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (BLUR, BLUR), 0)

                mean_intensity = np.mean(blurred)
                threshold_value = mean_intensity + THRESHOLD_RELATIVO * (255 - mean_intensity)
                _, mask = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)

                # Filtrar ruido pequeño
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

                hotspots = 0
                for i in range(1, num_labels):
                    area = stats[i, cv2.CC_STAT_AREA]
                    if area >= AREA_MINIMA:
                        x, y, w, h = stats[i, :4]
                        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        hotspots += 1

                if hotspots > 0:
                    filename = f"{output_dir}/{label}_{count}_hotspot.jpg"
                    cv2.imwrite(filename, roi)
                    count += 1

        # --- Guardar imagen final con anotaciones ---
        out_path = os.path.join(output_dir, f"resultado_{nombre}")
        cv2.imwrite(out_path, frame)
        print(f"Procesada: {nombre}")

print("\n✅ Proceso terminado. Resultados guardados en:", output_dir)
