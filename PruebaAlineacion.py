import streamlit as st
import cv2
import os
from datetime import datetime
from PIL import Image
import numpy as np
import time

# === CONFIGURACIÓN GENERAL ===
st.set_page_config(page_title="DAPCTER", page_icon="🛰️", layout="wide")

st.title("DAPCTER - Sistema de Detección y Procesamiento Aéreo")
st.write("Aplicación para conectar, capturar y procesar imágenes térmicas de paneles solares.")

# === MENÚ DE PESTAÑAS ===
tabs = st.tabs(["Conexión", "Vuelo", "Procesamiento"])

# ===============================================================
# 1️⃣ Pestaña de Conexión
# ===============================================================
with tabs[0]:
    st.header("Conexión del Sistema")
    st.markdown("Conecta el transmisor y verifica la cámara en tiempo real.")

    iniciar = st.button("🔌 Iniciar conexión", key="btn_iniciar_conexion")

    if iniciar:
        st.success("✅ Transmisor conectado correctamente.")
        st.info("La cámara está lista para transmitir video.")

    camara_conexion = st.checkbox("Mostrar cámara en vivo (Conexión)", key="chk_conexion")

    if camara_conexion:
        stframe = st.empty()
        cap = cv2.VideoCapture(0)
        cap.set(3, 640)
        cap.set(4, 480)

        st.warning("Presiona *Detener vista previa* para cerrar la cámara.")
        stop = st.button("Detener vista previa", key="btn_detener_conexion")

        while cap.isOpened() and not stop:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ No se pudo acceder a la cámara.")
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB", caption="Cámara en vivo - Conexión")
            time.sleep(0.03)
            stop = st.session_state.get("btn_detener_conexion", False)
            if stop:
                break

        cap.release()
        st.success("📷 Vista previa detenida.")

# ===============================================================
# 2️⃣ Pestaña de Vuelo
# ===============================================================
with tabs[1]:
    st.header("Vuelo y Captura de Imágenes")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Transmisión en vivo durante vuelo")
        camara_vuelo = st.checkbox("Activar cámara de vuelo", key="chk_vuelo")

        if camara_vuelo:
            stframe2 = st.empty()
            cap2 = cv2.VideoCapture(0)
            cap2.set(3, 640)
            cap2.set(4, 480)

            st.warning("Pulsa *Detener cámara* para finalizar transmisión.")
            stop_vuelo = st.button("Detener cámara", key="btn_detener_vuelo")

            while cap2.isOpened() and not stop_vuelo:
                ret, frame2 = cap2.read()
                if not ret:
                    st.error("❌ No se puede acceder a la cámara de vuelo.")
                    break
                frame_rgb2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                stframe2.image(frame_rgb2, channels="RGB", caption="Cámara en vuelo (en tiempo real)")
                time.sleep(0.03)

                # Verificar si se pulsó el botón
                stop_vuelo = st.session_state.get("btn_detener_vuelo", False)
                if stop_vuelo:
                    break

            cap2.release()
            st.success("📴 Cámara de vuelo detenida.")

        # Botón para tomar captura fuera del bucle
        if st.button("📸 Tomar captura", key="btn_captura_vuelo"):
            cap3 = cv2.VideoCapture(0)
            ret, frame3 = cap3.read()
            if ret:
                fecha = datetime.now().strftime("%Y%m%d_%H%M%S")
                carpeta = "Capturas"
                os.makedirs(carpeta, exist_ok=True)
                filename = os.path.join(carpeta, f"captura_{fecha}.jpg")
                cv2.imwrite(filename, frame3)
                st.success(f"✅ Captura guardada: {filename}")
            else:
                st.error("❌ No se pudo tomar la captura.")
            cap3.release()

    with col2:
        st.subheader("Revisión de capturas previas")
        uploaded_files = st.file_uploader("Importar capturas", type=["jpg", "png"], accept_multiple_files=True, key="upl_vuelo")

        if uploaded_files:
            for img_file in uploaded_files:
                img = Image.open(img_file)
                st.image(img, caption=f"Imagen: {img_file.name}", use_container_width=True)

    st.info("Cuando termines de revisar, pasa a la pestaña de procesamiento.")

# ===============================================================
# 3️⃣ Pestaña de Procesamiento
# ===============================================================
with tabs[2]:
    st.header("Procesamiento de Imágenes")
    st.markdown("Sube tus imágenes o usa las existentes en la carpeta `Pruebas` para procesarlas con el modelo YOLO y clasificar paneles con hotspots.")

    uploaded_files_proc = st.file_uploader(
        "Seleccionar imágenes a procesar", type=["jpg", "png"], accept_multiple_files=True, key="upl_proc"
    )

    if st.button("Iniciar procesamiento", key="btn_iniciar_proc"):
        from ultralytics import YOLO
        from scipy.signal import convolve2d

        st.info("🔄 Iniciando procesamiento, por favor espera...")

        input_dir = "Pruebas"
        ruta_base = "C:/DAPCTER/ProyectoDAPCTER"
        fecha_hoy = datetime.now().strftime("%Y-%m-%d")
        ruta_carpeta = os.path.join(ruta_base, fecha_hoy)

        output_dir_HS = os.path.join(ruta_carpeta, "Paneles_HS")
        output_dir_OK = os.path.join(ruta_carpeta, "Paneles_Sanos")
        os.makedirs(output_dir_HS, exist_ok=True)
        os.makedirs(output_dir_OK, exist_ok=True)

        VECINDAD = 5
        kernel = np.ones((VECINDAD, VECINDAD), np.float32)

        if uploaded_files_proc:
            input_dir = os.path.join(ruta_carpeta, "Subidas")
            os.makedirs(input_dir, exist_ok=True)
            for img_file in uploaded_files_proc:
                img = Image.open(img_file)
                img.save(os.path.join(input_dir, img_file.name))

        if not os.path.exists(input_dir):
            st.error(f"No existe la carpeta {input_dir}")
            st.stop()

        imagenes = [
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if not imagenes:
            st.warning("⚠️ No hay imágenes en la carpeta para procesar.")
            st.stop()

        st.write("📦 Cargando modelo YOLO...")
        model = YOLO("best.pt")
        countHS, countOK = 0, 0
        progress = st.progress(0)

        for i, ruta in enumerate(imagenes):
            frame = cv2.imread(ruta)
            if frame is None:
                st.warning(f"No se pudo leer {os.path.basename(ruta)}")
                continue

            results = model(frame, verbose=False)
            for res in results:
                for box in res.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    roi = frame[y1:y2, x1:x2]
                    if roi.size == 0:
                        continue

                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    resultado = convolve2d(gray_roi, kernel, mode='same', boundary='fill', fillvalue=0)
                    mean_val = np.mean(resultado)
                    mask = resultado > (mean_val + 1000)
                    num_pixels = np.count_nonzero(mask)

                    if num_pixels >= 200 and (x2 - x1) > 30 and (y2 - y1) > 30:
                        filename = os.path.join(output_dir_HS, f"PanelHS_{countHS}.jpg")
                        countHS += 1
                    else:
                        filename = os.path.join(output_dir_OK, f"Panel_{countOK}.jpg")
                        countOK += 1

                    cv2.imwrite(filename, roi)

            progress.progress((i + 1) / len(imagenes))

        st.success(f"✅ Procesamiento finalizado.\nPaneles con HS: {countHS} | Paneles sanos: {countOK}")
