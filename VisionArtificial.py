import cv2
import numpy as np
from ultralytics import YOLO
import os

# Carpeta donde están las imágenes guardadas por el script anterior
input_dir = "ImagenesPruebas"

# Verificar carpeta
if not os.path.exists(input_dir):
    raise FileNotFoundError(f"No existe la carpeta {input_dir}")

# Obtener lista de imágenes
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

        # === Cargar el modelo YOLO ===
        model = YOLO("best.pt")

        classNames = ["Panel-Hotspots"]

        output_dir = "detecciones_guardadas"
        os.makedirs(output_dir, exist_ok=True)

        # === Procesar la imagen ===
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

                # === Extraer ROI y convertir a escala de grises ===
                roi = frame[y1:y2, x1:x2]
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

                # === Calcular media, máximo y mínimo ===
                mean_val = np.mean(gray_roi)
                max_val = np.max(gray_roi)
                min_val = np.min(gray_roi)
                diff_max_mean = max_val - mean_val

                # === Detección de “manchas blancas” ===
                mask = gray_roi > (mean_val + 36)
                num_pixels_manchas = np.sum(mask)

                print(f"\n📸 {nombre}")
                print(f"-> ROI {label}: Media={mean_val:.2f}, Máx={max_val}, Mín={min_val}, DifMáx-Media={diff_max_mean:.2f}")
                print(f"   Píxeles por encima de media+50: {num_pixels_manchas}")

                # === Guardar solo si hay manchas válidas ===
                if num_pixels_manchas >= 100 and x2-x1>30 and y2-y1>30:
                    filename = f"{output_dir}/{label}_{count}.jpg"
                    cv2.imwrite(filename, roi)
                    count += 1
                    print(f"   ✅ Mancha detectada — guardado como {filename}")
                else:
                    print("   ⚠️ No cumple con criterio de manchas (no se guarda).")

        print(f"\n🔹 Imagen {nombre} procesada.")

print("\n✅ Proceso terminado.")
