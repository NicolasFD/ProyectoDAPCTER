import cv2
import numpy as np
from ultralytics import YOLO
import os

# === CONFIGURACIÓN ===
input_dir = "ImagenesPruebas"
output_dir = "manchas_detectadas"
os.makedirs(output_dir, exist_ok=True)

# Cargar modelo YOLO
model = YOLO("best.pt")  # puedes cambiar al tuyo entrenado

# === PARÁMETROS ===
INTENSIDAD_UMBRAL = 45
VECINDAD = 5   # Tamaño de la ventana (ej. 3x3 o 5x5)

# Crear kernel de suma local (filtro de vecindad)
kernel = np.ones((VECINDAD, VECINDAD), np.float32)

# === PROCESAMIENTO DE IMÁGENES ===
for filename in os.listdir(input_dir):
    if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    path = os.path.join(input_dir, filename)
    img = cv2.imread(path)
    if img is None:
        print(f"No se pudo cargar {filename}")
        continue

    # Detección YOLO
    results = model(img, verbose=False)
    detections = results[0].boxes.xyxy.cpu().numpy().astype(int)

    # Convertir a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Aplicar convolución local (suma de vecindad)
    suma_local = cv2.filter2D(gray, -1, kernel)

    # Normalizar el resultado a 0-255
    suma_local_norm = cv2.normalize(suma_local, None, 0, 255, cv2.NORM_MINMAX)

    # Generar mapa binario de puntos calientes
    umbral = np.mean(suma_local_norm) + np.std(suma_local_norm)
    puntos_calientes = (suma_local_norm > umbral).astype(np.uint8) * 255

    # Dibujar detecciones YOLO y extraer regiones calientes
    for i, (x1, y1, x2, y2) in enumerate(detections):
        roi = puntos_calientes[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        # Si hay una mancha caliente significativa dentro
        if np.mean(roi) > INTENSIDAD_UMBRAL:
            recorte = img[y1:y2, x1:x2]
            output_path = os.path.join(output_dir, f"{filename}_mancha_{i}.png")
            cv2.imwrite(output_path, recorte)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Guardar imagen con resultados visuales
    resultado_path = os.path.join(output_dir, f"{filename}_resultado.png")
    cv2.imwrite(resultado_path, img)

    print(f"Procesada: {filename}")

print("✅ Procesamiento completado con realce de puntos calientes.")
