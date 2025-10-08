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
    count=0
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

                # Paso 1: Matriz de valores
                roi = frame[y1:y2,x1:x2]
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

                # Paso 2: Media y máscara
                mean_val = np.mean(gray_roi)
                num_pixels = np.sum(gray_roi > mean_val+36)
                #mask=gray_roi > mean_val

                # Paso 3: Guardar si hay píxeles mayores a la media
                if num_pixels >= 200 and x2-x1>30 and y2-y1>30:
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

                # Paso 2: Media y máscara
                mean_val = np.mean(gray_roi)
                mask = gray_roi > mean_val

                #Dibujar bounding box
                text = f"{label} {conf:.2f} Mean:{mean_val:.1f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 0, 0), 2)

        # Mostrar la imagen con detecciones
        
        
        
print("\nProceso terminado.")
