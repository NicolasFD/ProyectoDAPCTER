import streamlit as st
import cv2
import os
import time
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import numpy as np
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
# VARIABLES DE SESIÓN
# ===============================================================
if "vuelo_id" not in st.session_state:
    st.session_state.vuelo_id = None

if "ruta_vuelo" not in st.session_state:
    st.session_state.ruta_vuelo = None

if "captura_manual" not in st.session_state:
    st.session_state.captura_manual = False

# ===============================================================
# FUNCIÓN: CREAR SESIÓN DE VUELO
# ===============================================================
def crear_sesion_vuelo():
    fecha = datetime.now().strftime("%Y-%m-%d")
    hora = datetime.now().strftime("%H-%M-%S")

    base = "Capturas"
    ruta_dia = os.path.join(base, fecha)
    ruta_vuelo = os.path.join(ruta_dia, f"Vuelo_{hora}")

    os.makedirs(ruta_vuelo, exist_ok=True)

    st.session_state.vuelo_id = f"{fecha}_{hora}"
    st.session_state.ruta_vuelo = ruta_vuelo

def obtener_fechas():
    if not os.path.exists("Capturas"):
        return []
    return sorted(
        [d for d in os.listdir("Capturas")
         if os.path.isdir(os.path.join("Capturas", d))],
        reverse=True
    )

def obtener_vuelos(fecha):
    base = os.path.join("Capturas", fecha)
    if not os.path.exists(base):
        return []
    return sorted(
        [d for d in os.listdir(base)
         if os.path.isdir(os.path.join(base, d)) and d.startswith("Vuelo_")]
    )


# ===============================================================
# VIDEO PROCESSOR
# ===============================================================
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.ruta_vuelo = None
        self.capturar = False
        self.ultima_captura = 0
        self.cooldown = 1.0  # segundos

    def set_ruta_vuelo(self, ruta):
        self.ruta_vuelo = ruta

    def solicitar_captura(self):
        self.capturar = True

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        ahora = time.time()

        if (
            self.capturar
            and self.ruta_vuelo
            and (ahora - self.ultima_captura) >= self.cooldown
        ):
            hora = datetime.now().strftime("%H-%M-%S")
            nombre = f"captura_{hora}.jpg"
            ruta = os.path.join(self.ruta_vuelo, nombre)

            cv2.imwrite(ruta, img)

            self.ultima_captura = ahora
            self.capturar = False
            print(f"📸 Captura guardada → {ruta}")

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ===============================================================
# PESTAÑAS
# ===============================================================
tab_vuelo, tab_proc = st.tabs(
    ["✈️ Vuelo", "🧠 Procesamiento"]
)

# ===============================================================
# ✈️ VUELO
# ===============================================================
with tab_vuelo:
    st.header("✈️ Vuelo y Captura Manual")

    col_left, col_right = st.columns([1, 2])

    # ==========================
    # CONTROLES
    # ==========================
    with col_right:
        st.subheader("📡 Cámara en tiempo real")

        ctx = webrtc_streamer(
            key="camara-vuelo",
            video_processor_factory=VideoProcessor,
            media_stream_constraints={
                "video": {"width": 640, "height": 480},
                "audio": False
            },
            async_processing=True
        )

        if ctx and ctx.video_processor and st.session_state.ruta_vuelo:
            ctx.video_processor.set_ruta_vuelo(
                st.session_state.ruta_vuelo
            )
    
    with col_left:
        st.subheader("🎮 Controles")

        if st.button("🛫 Iniciar nuevo vuelo"):
            crear_sesion_vuelo()
            st.success("Vuelo iniciado correctamente")

        if st.session_state.ruta_vuelo:
            st.info(f"📂 Carpeta activa:\n{st.session_state.ruta_vuelo}")
        else:
            st.warning("No hay vuelo activo")

        st.divider()

        if st.session_state.ruta_vuelo:
            if st.button("📸 Tomar captura"):
                if ctx and ctx.video_processor:
                    ctx.video_processor.solicitar_captura()
                    st.success("📸 Captura solicitada")
                else:
                    st.warning("La cámara no está lista")

    # ==========================
    # VIDEO
    # ==========================
# ===============================================================
# 3️⃣ PROCESAMIENTO
# ===============================================================
with tab_proc:
    st.header("🧠 Procesamiento de Imágenes")

    st.markdown("### 📂 Selección de vuelo")

    fechas = obtener_fechas()

    if not fechas:
        st.warning("No hay vuelos registrados aún")
        st.stop()

    fecha_sel = st.selectbox(
        "📅 Fecha del vuelo",
        fechas
    )

    vuelos = obtener_vuelos(fecha_sel)

    if not vuelos:
        st.warning("No hay vuelos en esta fecha")
        st.stop()

    vuelo_sel = st.selectbox(
        "✈️ Vuelo",
        vuelos
    )

    ruta_vuelo = os.path.join("Capturas", fecha_sel, vuelo_sel)

    imagenes = [
        os.path.join(ruta_vuelo, f)
        for f in os.listdir(ruta_vuelo)
        if f.lower().endswith((".jpg", ".png"))
    ]

    st.info(f"📸 Imágenes encontradas: {len(imagenes)}")

    st.divider()

    # -----------------------------------------------------------
    # PROCESAMIENTO
    # -----------------------------------------------------------
    if st.button("🚀 Iniciar procesamiento"):

        if not imagenes:
            st.error("❌ No hay imágenes para procesar")
            st.stop()

        from ultralytics import YOLO
        from scipy.signal import convolve2d

        st.info("🔄 Procesando imágenes...")

        base_resultados = os.path.join(
            "Resultados",
            fecha_sel,
            vuelo_sel
        )

        hs_dir = os.path.join(base_resultados, "Paneles_HS")
        ok_dir = os.path.join(base_resultados, "Paneles_Sanos")

        os.makedirs(hs_dir, exist_ok=True)
        os.makedirs(ok_dir, exist_ok=True)

        model = YOLO("best.pt")
        kernel = np.ones((5, 5), np.float32)

        countHS, countOK = 0, 0
        progress = st.progress(0)

        for i, ruta in enumerate(imagenes):
            frame = cv2.imread(ruta)
            results = model(frame, verbose=False)

            for res in results:
                for box in res.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    roi = frame[y1:y2, x1:x2]

                    if roi.size == 0:
                        continue

                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    conv = convolve2d(gray, kernel, mode="same")

                    if np.count_nonzero(conv > (np.mean(conv) + 1000)) >= 200:
                        cv2.imwrite(
                            os.path.join(hs_dir, f"HS_{countHS}.jpg"),
                            roi
                        )
                        countHS += 1
                    else:
                        cv2.imwrite(
                            os.path.join(ok_dir, f"OK_{countOK}.jpg"),
                            roi
                        )
                        countOK += 1

            progress.progress((i + 1) / len(imagenes))

        st.success(
            f"✅ Finalizado | HS: {countHS} | Sanos: {countOK}"
        )


