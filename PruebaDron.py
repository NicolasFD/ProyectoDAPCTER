import streamlit as st
import cv2
import os
import time
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

# ===============================================================
# CONFIGURACIÓN GENERAL
# ===============================================================
st.set_page_config(
    page_title="DAPCTER",
    page_icon="🛰️",
    layout="wide"
)

st.title("DAPCTER - Sistema de Detección y Procesamiento Aéreo")
st.write("Aplicación para conectar, capturar y procesar imágenes térmicas.")

# ===============================================================
# VIDEO PROCESSOR
# ===============================================================
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.capturar = False
        self.ultimo_disparo = 0
        self.cooldown = 1.0  # segundos

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        ahora = time.time()

        if self.capturar and (ahora - self.ultimo_disparo) >= self.cooldown:
            os.makedirs("Capturas", exist_ok=True)
            ruta = os.path.join(
                "Capturas",
                f"captura_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            )
            cv2.imwrite(ruta, img)
            self.ultimo_disparo = ahora
            self.capturar = False
            print(f"📸 Captura guardada → {ruta}")

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ===============================================================
# PESTAÑAS
# ===============================================================
tab_conexion, tab_vuelo, tab_proc = st.tabs(
    ["🔌 Conexión", "✈️ Vuelo", "🧠 Procesamiento"]
)

# ===============================================================
# 1️⃣ CONEXIÓN
# ===============================================================
with tab_conexion:
    st.header("Conexión del Sistema")

    if st.button("🔌 Iniciar conexión"):
        st.success("✅ Transmisor conectado")
        st.info("Cámara lista para el vuelo")

# ===============================================================
# 2️⃣ VUELO (CÁMARA + CAPTURA)
# ===============================================================
with tab_vuelo:
    st.header("Vuelo y Captura Manual")

    col_left, col_right = st.columns([1, 2])

    # ---- VISUALIZACIÓN ----
    with col_right:
        st.subheader("📡 Vista en tiempo real")

        ctx = webrtc_streamer(
            key="camara-vuelo",
            video_processor_factory=VideoProcessor,
            media_stream_constraints={
                "video": {"width": 640, "height": 480},
                "audio": False
            },
            async_processing=True
        )

    # ---- CONTROLES ----
    with col_left:
        st.subheader("🎮 Controles de Vuelo")

        if ctx.video_processor:
            if st.button("📸 Tomar captura"):
                ctx.video_processor.capturar = True
                st.success("📸 Imagen capturada")

            st.markdown("### ⏱️ Seguridad")
            st.write(
                f"Tiempo mínimo entre capturas: "
                f"{ctx.video_processor.cooldown:.1f} s"
            )
        else:
            st.info("Esperando inicialización de la cámara...")

        st.divider()
        st.caption("DAPCTER · Captura manual estable")

# ===============================================================
# 3️⃣ PROCESAMIENTO (VACÍO POR AHORA)
# ===============================================================
with tab_proc:
    st.header("Procesamiento de Imágenes")
    st.info("Aquí irá YOLO y análisis térmico (siguiente paso)")
