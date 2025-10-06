import cv2
import numpy as np
#from pytesseract import image_to_string  # Solo si quieres OCR

# Cargar imagen
imagen = cv2.imread('001R.jpg')

# Convertir a gris y suavizar
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gris, (5,5), 0)

# Detectar bordes
edges = cv2.Canny(blur, 50, 150)

# Buscar contornos
contornos, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for c in contornos:
    perimetro = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02*perimetro, True)

    if len(approx) == 4:  # Posible rectángulo
        x, y, w, h = cv2.boundingRect(approx)
        roi = imagen[y:y+h, x:x+w]

        # --- 1. Matriz de valores del rectángulo ---
        matriz_valores = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)  # ejemplo en HSV
        print("Matriz de valores HSV:", matriz_valores.shape)

        # --- 2. Detectar pixel fuera de rango ---
        mean_color = np.mean(matriz_valores.reshape(-1,3), axis=0)
        lower = np.clip(mean_color - np.array([10,40,40]), 0, 255)
        upper = np.clip(mean_color + np.array([10,40,40]), 0, 255)
        mask = cv2.inRange(matriz_valores, lower, upper)

        # Los píxeles fuera de rango:
        fuera_de_rango = cv2.bitwise_not(mask)
        porcentaje_fuera = np.sum(fuera_de_rango > 0) / (w*h) * 100
        print(f"Rectángulo en ({x},{y}) tiene {porcentaje_fuera:.2f}% píxeles fuera de rango")

        # --- 3. Si quieres leer números dentro de la caja ---
        # texto = image_to_string(roi, config='--psm 7')
        # print("Texto detectado:", texto)

        # Dibujar rectángulo en la imagen
        cv2.drawContours(imagen, [approx], 0, (0,255,0), 2)

cv2.imshow("Rectángulos detectados", imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()
