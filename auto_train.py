import argparse
import os
import cv2
import numpy as np
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--conf", type=float, default=0.45)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--min_area", type=int, default=500)
    parser.add_argument("--output", default="resultados")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    model = YOLO(args.model)

    img = cv2.imread(args.image)
    H, W = img.shape[:2]

    results = model(img, verbose=False, conf=args.conf, iou=args.iou)

    imagen_segmentada = img.copy()
    instancia_id = 0

    for r in results:
        if r.masks is None:
            continue

        masks = r.masks.data.detach().cpu().numpy()

        for m in masks:
            mask = (m > 0.5).astype(np.uint8)

            # 🔥 Ajustar tamaño máscara si es necesario
            if mask.shape[:2] != (H, W):
                mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

            area = int(mask.sum())
            if area < args.min_area:
                continue

            instancia_id += 1

            # Crear carpeta por instancia
            carpeta_instancia = os.path.join(args.output, f"panel_{instancia_id}")
            os.makedirs(carpeta_instancia, exist_ok=True)

            # Guardar máscara
            cv2.imwrite(os.path.join(carpeta_instancia, "mask.png"), mask * 255)

            # Recortar ROI usando máscara
            ys, xs = np.where(mask == 1)
            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()

            roi = img[y1:y2+1, x1:x2+1]
            mask_roi = mask[y1:y2+1, x1:x2+1]

            roi_masked = cv2.bitwise_and(roi, roi, mask=mask_roi.astype(np.uint8))

            cv2.imwrite(os.path.join(carpeta_instancia, "roi_original.png"), roi)
            cv2.imwrite(os.path.join(carpeta_instancia, "roi_masked.png"), roi_masked)

            # Dibujar contorno en imagen global
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(imagen_segmentada, contours, -1, (0,255,0), 2)
            cv2.putText(imagen_segmentada, f"P{instancia_id}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            
            # Obtener coordenadas del panel desde la máscara
            ys, xs = np.where(mask == 1)

            if len(xs) == 0 or len(ys) == 0:
                continue

            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()

            # Recorte exacto
            roi = img[y1:y2+1, x1:x2+1]
            mask_roi = mask[y1:y2+1, x1:x2+1].astype(np.uint8)

            # 🔥 Aplicar máscara para eliminar fondo
            roi_limpio = cv2.bitwise_and(roi, roi, mask=mask_roi)

            # Guardar resultado limpio
            cv2.imwrite(os.path.join(carpeta_instancia, "panel_limpio.png"), roi_limpio)

    # Guardar imagen completa segmentada
    cv2.imwrite(os.path.join(args.output, "imagen_segmentada.png"), imagen_segmentada)

    print(f"✅ Total instancias guardadas: {instancia_id}")
    print(f"📁 Carpeta resultados: {args.output}")


if __name__ == "__main__":
    main()