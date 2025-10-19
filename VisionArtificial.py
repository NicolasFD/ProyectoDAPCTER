import streamlit as st
import cv2
import os
from datetime import datetime
from PIL import Image
import numpy as np

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
    st.markdown("""
    En esta sección puedes conectar el transmisor con la cámara y realizar una prueba de video.
    """)

    iniciar = st.button("Iniciar conexión")
    if iniciar:
        st.success("Transmisor conectado correctamente.")
        st.info("La cámara está lista para transmitir video.")

    # Simulación de vista previa
    camara_activa = st.checkbox("Mostrar vista previa de cámara")
    if camara_activa:
        st.image("https://via.placeholder.com/640x360?text=Vista+Previa+de+la+Cámara",
                 caption="Simulación de transmisión")

# ===============================================================
# 2️⃣ Pestaña de Vuelo
# ===============================================================
with tabs[1]:
    st.header("Vuelo y Captura de Imágenes")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Visualización en vivo")
        st.image("https://via.placeholder.com/640x360?text=Cámara+en+vivo")

        if st.button("Tomar captura"):
            fecha = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.success(f"Captura guardada como captura_{fecha}.jpg")

    with col2:
        st.subheader("Revisión de capturas")
        uploaded_files = st.file_uploader("Importar capturas", type=["jpg", "png"], accept_multiple_files=True)

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

    st.markdown("Sube tus imágenes para procesarlas y generar el reporte en PDF.")

    uploaded_files_proc = st.file_uploader("Seleccionar imágenes a procesar", type=["jpg", "png"], accept_multiple_files=True)

    if st.button("Iniciar procesamiento"):
        if uploaded_files_proc:
            progress = st.progress(0)
            for i, img_file in enumerate(uploaded_files_proc):
                # Simulación del procesamiento
                progress.progress((i + 1) / len(uploaded_files_proc))
            st.success("Procesamiento finalizado")
            st.download_button("Descargar reporte PDF", "Reporte_DAPCTER.pdf")
        else:
            st.warning("Por favor sube imágenes antes de procesar.")
