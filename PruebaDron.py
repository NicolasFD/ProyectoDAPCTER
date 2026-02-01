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
    return sorted(
        [d for d in os.listdir(base)
         if os.path.isdir(os.path.join(base, d))],
        reverse=True
    )

def obtener_items(base, prefijo):
    if not os.path.exists(base):
        return []
    return sorted(
        [d for d in os.listdir(base)
         if os.path.isdir(os.path.join(base, d)) and d.startswith(prefijo)]
    )

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
            media_stream_constraints={
                "video": {"width": 640, "height": 480},
                "audio": False
            },
            async_processing=True
        )

        if ctx and ctx.video_processor and st.session_state.ruta_vuelo:
            ctx.video_processor.set_ruta(st.session_state.ruta_vuelo)

    with col_l:
        if st.button("🛫 Iniciar nuevo vuelo"):
            crear_sesion_vuelo()
            st.success("Vuelo iniciado")

        if st.session_state.ruta_vuelo:
            st.info(st.session_state.ruta_vuelo)
            if st.button("📸 Tomar captura"):
                if ctx and ctx.video_processor:
                    ctx.video_processor.solicitar_captura()

# ===============================================================
# 🧠 PROCESAMIENTO
# ===============================================================
with tab_proc:

    # ------------------ SUBIDA ------------------
    st.subheader("⬆️ Subir imágenes")
    archivos = st.file_uploader(
        "Selecciona imágenes",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if archivos:
        fecha = datetime.now().strftime("%Y-%m-%d")
        hora = datetime.now().strftime("%H-%M-%S")
        ruta_subida = os.path.join("Subidas", fecha, f"Subida_{hora}")
        os.makedirs(ruta_subida, exist_ok=True)

        for a in archivos:
            with open(os.path.join(ruta_subida, a.name), "wb") as f:
                f.write(a.getbuffer())

        st.success(f"✅ {len(archivos)} imágenes guardadas")

    st.divider()

    # ------------------ ORIGEN ------------------
    origen = st.radio(
        "📂 Origen de imágenes",
        ["✈️ Vuelo", "⬆️ Subida"],
        horizontal=True
    )

    ruta_origen = None
    fecha_sel = None

    if origen == "✈️ Vuelo":

        fechas = obtener_fechas("Capturas")
        if not fechas:
            st.warning("No hay vuelos registrados")
            st.stop()

        fecha_sel = st.selectbox(
            "📅 Fecha",
            fechas,
            index=None,
            placeholder="Selecciona una fecha"
        )

        if fecha_sel is None:
            st.stop()

        vuelos = obtener_items(
            os.path.join("Capturas", fecha_sel),
            "Vuelo_"
        )

        if not vuelos:
            st.warning("No hay vuelos en esta fecha")
            st.stop()

        vuelo_sel = st.selectbox(
            "✈️ Vuelo",
            vuelos,
            index=None,
            placeholder="Selecciona un vuelo"
        )

        if vuelo_sel is None:
            st.stop()

        ruta_origen = os.path.join("Capturas", fecha_sel, vuelo_sel)

    else:

        fechas = obtener_fechas("Subidas")
        if not fechas:
            st.warning("No hay subidas registradas")
            st.stop()

        fecha_sel = st.selectbox(
            "📅 Fecha de subida",
            fechas,
            index=None,
            placeholder="Selecciona una fecha"
        )

        if fecha_sel is None:
            st.stop()

        subidas = obtener_items(
            os.path.join("Subidas", fecha_sel),
            "Subida_"
        )

        if not subidas:
            st.warning("No hay subidas en esta fecha")
            st.stop()

        subida_sel = st.selectbox(
            "⬆️ Subida",
            subidas,
            index=None,
            placeholder="Selecciona una subida"
        )

        if subida_sel is None:
            st.stop()

        ruta_origen = os.path.join("Subidas", fecha_sel, subida_sel)

    if ruta_origen is None or not os.path.exists(ruta_origen):
        st.stop()

    # ------------------ IMÁGENES ------------------
    imagenes = [
        os.path.join(ruta_origen, f)
        for f in os.listdir(ruta_origen)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    st.info(f"📸 Imágenes encontradas: {len(imagenes)}")

    # ===============================================================
    # SELECTOR MAESTRO
    # ===============================================================
    st.subheader("🖼️ Selección de imágenes")

    modo = st.radio(
        "Selección rápida",
        ["Manual", "Seleccionar todas", "Ninguna"],
        horizontal=True
    )

    for img in imagenes:
        if img not in st.session_state.seleccion_imagenes:
            st.session_state.seleccion_imagenes[img] = True

    if modo == "Seleccionar todas":
        for img in imagenes:
            st.session_state.seleccion_imagenes[img] = True
    elif modo == "Ninguna":
        for img in imagenes:
            st.session_state.seleccion_imagenes[img] = False

    imagenes_seleccionadas = []
    cols = st.columns(4)

    for i, ruta_img in enumerate(imagenes):
        with cols[i % 4]:
            st.image(ruta_img, use_container_width=True)
            marcado = st.checkbox(
                os.path.basename(ruta_img),
                value=st.session_state.seleccion_imagenes[ruta_img],
                key=f"chk_{ruta_img}"
            )
            st.session_state.seleccion_imagenes[ruta_img] = marcado
            if marcado:
                imagenes_seleccionadas.append(ruta_img)

    st.info(f"📊 Seleccionadas: {len(imagenes_seleccionadas)} / {len(imagenes)}")

    # ===============================================================
    # PROCESAMIENTO
    # ===============================================================
    if st.button("🚀 Iniciar procesamiento"):

        if not imagenes_seleccionadas:
            st.error("❌ No hay imágenes seleccionadas")
            st.stop()

        from ultralytics import YOLO
        from scipy.signal import convolve2d

        base_res = os.path.join("Resultados", fecha_sel, os.path.basename(ruta_origen))
        hs_dir = os.path.join(base_res, "Paneles_HS")
        ok_dir = os.path.join(base_res, "Paneles_Sanos")
        os.makedirs(hs_dir, exist_ok=True)
        os.makedirs(ok_dir, exist_ok=True)

        model = YOLO("best.pt")
        kernel = np.ones((5, 5), np.float32)

        hs, ok = 0, 0
        barra = st.progress(0)

        for i, ruta in enumerate(imagenes_seleccionadas):
            img = cv2.imread(ruta)
            res = model(img, verbose=False)

            for r in res:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    roi = img[y1:y2, x1:x2]
                    if roi.size == 0:
                        continue

                    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    c = convolve2d(g, kernel, mode="same")

                    if np.count_nonzero(c > np.mean(c) + 1000) >= 200:
                        cv2.imwrite(os.path.join(hs_dir, f"HS_{hs}.jpg"), roi)
                        hs += 1
                    else:
                        cv2.imwrite(os.path.join(ok_dir, f"OK_{ok}.jpg"), roi)
                        ok += 1

            barra.progress((i + 1) / len(imagenes_seleccionadas))

        st.success(f"✅ Procesado | HS: {hs} | Sanos: {ok}")
