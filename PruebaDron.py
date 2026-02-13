import streamlit as st
import cv2
import os
import time
import json
import subprocess
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import numpy as np
from ultralytics import YOLO
from scipy.signal import convolve2d

# ===============================================================
# CONFIGURACIÓN GENERAL
# ===============================================================
st.set_page_config(
    page_title="DAPCTER",
    page_icon="🛰️",
    layout="wide"
)

st.title("DAPCTER - Sistema de Detección y Procesamiento Aéreo")

# ===============================================================
# CONSTANTES
# ===============================================================
MODEL_PATH = "models/current.pt"
STATE_FILE = "train_state.json"
LOCK_FILE = "training.lock"
VUELOS_BASE = "Capturas"

# ===============================================================
# AUTO-ENTRENAMIENTO AL INICIO
# ===============================================================
def hay_vuelos_nuevos():
    if not os.path.exists(VUELOS_BASE):
        return False

    vuelos = []
    for fecha in os.listdir(VUELOS_BASE):
        ruta = os.path.join(VUELOS_BASE, fecha)
        if os.path.isdir(ruta):
            vuelos.extend(
                os.path.join(fecha, v)
                for v in os.listdir(ruta)
                if v.startswith("Vuelo_")
            )

    if not vuelos:
        return False

    vuelos.sort()

    if not os.path.exists(STATE_FILE):
        return True

    with open(STATE_FILE) as f:
        ultimo = json.load(f).get("ultimo_vuelo")

    return vuelos[-1] != ultimo


def lanzar_entrenamiento():
    if os.path.exists(LOCK_FILE):
        return

    subprocess.Popen(
        ["python", "auto_train.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )


if hay_vuelos_nuevos():
    st.info("🧠 Nuevos vuelos detectados. Reentrenando modelo en segundo plano…")
    lanzar_entrenamiento()
else:
    st.success("✅ Modelo actualizado")

# ===============================================================
# VARIABLES DE SESIÓN
# ===============================================================
if "ruta_vuelo" not in st.session_state:
    st.session_state.ruta_vuelo = None

if "seleccion_imagenes" not in st.session_state:
    st.session_state.seleccion_imagenes = {}

# ===============================================================
# FUNCIONES AUXILIARES
# ===============================================================
def crear_sesion_vuelo():
    fecha = datetime.now().strftime("%Y-%m-%d")
    hora = datetime.now().strftime("%H-%M-%S")
    ruta = os.path.join("Capturas", fecha, f"Vuelo_{hora}")
    os.makedirs(ruta, exist_ok=True)
    st.session_state.ruta_vuelo = ruta


def obtener_fechas(base):
    if not os.path.exists(base):
        return []
    return sorted(os.listdir(base), reverse=True)


def obtener_items(base, prefijo):
    if not os.path.exists(base):
        return []
    return sorted(d for d in os.listdir(base) if d.startswith(prefijo))


def rotar_imagen(img, angulo):
    (h, w) = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angulo, 1.0)
    return cv2.warpAffine(img, M, (w, h))


def obtener_angulo_panel(roi):
    gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gris, 50, 150)
    lineas = cv2.HoughLines(edges, 1, np.pi / 180, 120)
    if lineas is None:
        return 0.0
    angulos = [(l[0][1] - np.pi / 2) * 180 / np.pi for l in lineas[:10]]
    return float(np.median(angulos))

# ===============================================================
# VIDEO PROCESSOR
# ===============================================================
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.ruta = None
        self.capturar = False
        self.ultima = 0
        self.cooldown = 1.0

    def set_ruta(self, ruta):
        self.ruta = ruta

    def solicitar_captura(self):
        self.capturar = True

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        ahora = time.time()

        if self.capturar and self.ruta and (ahora - self.ultima) >= self.cooldown:
            nombre = f"captura_{datetime.now().strftime('%H-%M-%S')}.jpg"
            cv2.imwrite(os.path.join(self.ruta, nombre), img)
            self.ultima = ahora
            self.capturar = False

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ===============================================================
# PESTAÑAS
# ===============================================================
tab_vuelo, tab_proc = st.tabs(["✈️ Vuelo", "🧠 Procesamiento"])

# ===============================================================
# ✈️ VUELO
# ===============================================================
with tab_vuelo:
    col_l, col_r = st.columns([1, 2])

    with col_r:
        ctx = webrtc_streamer(
            key="camara",
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False}
        )

        if ctx and ctx.video_processor and st.session_state.ruta_vuelo:
            ctx.video_processor.set_ruta(st.session_state.ruta_vuelo)

    with col_l:
        if st.button("🛫 Iniciar nuevo vuelo"):
            crear_sesion_vuelo()
            st.success("Vuelo iniciado")

        if st.session_state.ruta_vuelo and st.button("📸 Tomar captura"):
            ctx.video_processor.solicitar_captura()

# ===============================================================
# 🧠 PROCESAMIENTO
# ===============================================================
with tab_proc:
    origen = st.radio("Origen", ["✈️ Vuelo", "⬆️ Subida"], horizontal=True)

    if origen == "✈️ Vuelo":
        fechas = obtener_fechas("Capturas")
        fecha_sel = st.selectbox("Fecha", fechas)
        vuelos = obtener_items(os.path.join("Capturas", fecha_sel), "Vuelo_")
        vuelo_sel = st.selectbox("Vuelo", vuelos)
        ruta_origen = os.path.join("Capturas", fecha_sel, vuelo_sel)
    else:
        st.stop()

    imagenes = [
        os.path.join(ruta_origen, f)
        for f in os.listdir(ruta_origen)
        if f.endswith(".jpg")
    ]

    model = YOLO(MODEL_PATH)
    kernel = np.ones((3, 3), np.float32)

    if st.button("🚀 Iniciar procesamiento"):
        hs, ok = 0, 0
        for ruta in imagenes:
            img = cv2.imread(ruta)
            res = model(img, verbose=False)

            for r in res:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    roi = img[y1:y2, x1:x2]
                    if roi.size == 0:
                        continue

                    roi = rotar_imagen(roi, obtener_angulo_panel(roi))
                    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    c = convolve2d(g, kernel, mode="same")

                    if np.count_nonzero(c > np.mean(c) + 1000) >= 200:
                        hs += 1
                    else:
                        ok += 1

        st.success(f"Procesado | HS: {hs} | OK: {ok}")
