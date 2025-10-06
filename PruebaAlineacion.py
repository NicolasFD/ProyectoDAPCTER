import cv2
import numpy as np
import os

# Carpeta donde están las imágenes guardadas por el script anterior
input_dir = "detecciones_guardadas"

# Verificar carpeta
if not os.path.exists(input_dir):
    raise FileNotFoundError(f"No existe la carpeta {input_dir}")

# Obtener lista de imágenes
imagenes = [f for f in os.listdir(input_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))]

if not imagenes:
    print("No hay imágenes en la carpeta.")
else:
    print("Resultados por imagen (media, mínimo, máximo, diferencia max-media):\n")

    for nombre in imagenes:
        ruta = os.path.join(input_dir, nombre)
        img = cv2.imread(ruta)
        if img is None:
            print(f"No se pudo leer {nombre}")
            continue

        # Convertir a escala de grises
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Calcular estadísticas
        mean_val = np.mean(gray)
        min_val = np.min(gray)
        max_val = np.max(gray)
        diff_max_mean = max_val - mean_val  # diferencia

        print(f"{nombre}  -->  Media: {mean_val:.2f}  "
              f"Mínimo: {min_val}  Máximo: {max_val}  "
              f"Dif(Max-Media): {diff_max_mean:.2f}")

print("\nProceso terminado.")
