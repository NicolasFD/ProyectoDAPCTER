import streamlit as st
import cv2
import os
import time
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import numpy as np
from fpdf import FPDF


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

def rotar_imagen(img, angulo):
    (h, w) = img.shape[:2]
    centro = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(centro, angulo, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - centro[0]
    M[1, 2] += (nH / 2) - centro[1]

    return cv2.warpAffine(img, M, (nW, nH))


def obtener_angulo_panel(roi):
    gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gris, 50, 150, apertureSize=3)

    lineas = cv2.HoughLines(edges, 1, np.pi / 180, 120)
    if lineas is None:
        return 0.0

    angulos = []
    for linea in lineas[:10]:
        rho, theta = linea[0]
        angulo = (theta - np.pi / 2) * 180 / np.pi
        angulos.append(angulo)

    return float(np.median(angulos))

def extraer_roi_con_mascara(img, box_xyxy, mask_full):
    """
    img: imagen BGR completa (H,W,3)
    box_xyxy: (x1,y1,x2,y2) ints
    mask_full: máscara booleana o 0/1 del tamaño (H,W)
    """
    x1, y1, x2, y2 = box_xyxy
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(img.shape[1], x2); y2 = min(img.shape[0], y2)

    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return None, None

    mask_crop = mask_full[y1:y2, x1:x2].astype(np.uint8)  # 0/1
    if mask_crop.size == 0 or mask_crop.sum() < 50:
        return None, None

    # aplica máscara: deja fondo en negro
    roi_masked = roi.copy()
    roi_masked[mask_crop == 0] = (0, 0, 0)

    return roi_masked, mask_crop
# ===============================================================
# INFORME PDF
# ===============================================================

class InformePDF(FPDF):

    def header(self):
        self.set_font("Arial", "B", 16)
        self.cell(0, 10, "INFORME TECNICO DE INSPECCION TERMOGRAFICA", 0, 1, "C")
        self.set_font("Arial", "", 10)
        self.cell(0, 5, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 0, 1, "C")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Pagina {self.page_no()}", 0, 0, "C")

    def cuadro_tecnico(self, datos):

        margen_inferior = 20
        altura_panel = 45
        altura_tabla = 50

        altura_total = max(altura_panel, altura_tabla) + 10

        # Si no hay espacio suficiente → nueva página
        if self.get_y() + altura_total > self.h - margen_inferior:
            self.add_page()

        y_inicio = self.get_y()

        # -------------------------
        # IMAGEN PANEL (IZQUIERDA)
        # -------------------------

        if os.path.exists(datos["ruta_panel"]):

            ANCHO_PANEL = 40

            self.image(
                datos["ruta_panel"],
                x=10,
                y=y_inicio,
                w=ANCHO_PANEL
            )


        # -------------------------
        # TABLA (DERECHA)
        # -------------------------

        self.set_xy(60, y_inicio)

        self.set_font("Arial", "B", 10)

        filas = [

            ("Imagen origen", datos["imagen"]),
            ("Vuelo", datos["vuelo"]),
            ("Subida", datos["subida"]),
            ("Panel", datos["panel"]),
            ("Punto caliente", datos["punto_caliente"]),
            ("Gravedad", datos["gravedad"]),
            ("Recomendacion", datos["recomendacion"])

        ]

        col1 = 45
        col2 = 65

        for campo, valor in filas:

            # Verificar salto de página dentro de tabla
            if self.get_y() + 8 > self.h - margen_inferior:
                self.add_page()
                self.set_x(80)

            self.cell(col1, 8, campo, 1)
            self.cell(col2, 8, str(valor), 1)
            self.ln()
            self.set_x(80)

        # -------------------------
        # MOVER CURSOR ABAJO
        # -------------------------

        self.set_y(y_inicio + altura_total)




def clasificar_gravedad(es_hotspot, intensidad):

    if not es_hotspot:
        return "Normal"

    if intensidad < 1200:
        return "Bajo"

    elif intensidad < 1800:
        return "Medio"

    elif intensidad < 2500:
        return "Alto"

    else:
        return "Critico"


def obtener_recomendacion(gravedad):

    recomendaciones = {

        "Normal": "Panel en condiciones normales.",

        "Bajo": "Monitorear en siguientes inspecciones.",

        "Medio": "Programar mantenimiento preventivo.",

        "Alto": "Mantenimiento urgente requerido.",

        "Critico": "Apagar panel inmediatamente y reemplazar."
    }

    return recomendaciones.get(gravedad, "Sin informacion")


def generar_informe_pdf(base_res, vuelo, subida, imagenes_info):

    pdf = InformePDF()

    pdf.add_page()

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "INFORMACION DEL PROCESAMIENTO", 0, 1)

    pdf.set_font("Arial", "", 11)

    pdf.cell(0, 8, f"Vuelo: {vuelo}", 0, 1)
    pdf.cell(0, 8, f"Subida: {subida}", 0, 1)
    pdf.cell(0, 8, f"Fecha: {datetime.now().strftime('%Y-%m-%d')}", 0, 1)
    pdf.cell(0, 8, f"Hora: {datetime.now().strftime('%H:%M:%S')}", 0, 1)

    total_imagenes = len(imagenes_info)
    total_paneles = sum(len(img["paneles"]) for img in imagenes_info)

    pdf.cell(0, 8, f"Numero de imagenes: {total_imagenes}", 0, 1)
    pdf.cell(0, 8, f"Numero de paneles: {total_paneles}", 0, 1)

    for img_info in imagenes_info:

        pdf.add_page()

        ruta_img = img_info["ruta"]
        nombre = os.path.basename(ruta_img)

        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, nombre, 0, 1)

        if os.path.exists(ruta_img):

            # verificar espacio antes de insertar
            if pdf.get_y() + 100 > pdf.h - 20:
                pdf.add_page()

            pdf.image(ruta_img, x=15, w=180)

        pdf.ln(10)


        for panel in img_info["paneles"]:

            gravedad = panel["gravedad"]

            datos = {

                "imagen": nombre,
                "vuelo": vuelo,
                "subida": subida,
                "panel": panel["id"],
                "punto_caliente": "SI" if panel["hotspot"] else "NO",
                "gravedad": gravedad,
                "recomendacion": obtener_recomendacion(gravedad),
                "ruta_panel": panel["ruta_panel"]
            }


            pdf.cuadro_tecnico(datos)

            pdf.ln(5)

    ruta_pdf = os.path.join(base_res, "Informe_Termografico.pdf")

    pdf.output(ruta_pdf)

    return ruta_pdf


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

        model = YOLO("best-seg.pt")
        kernel = np.ones((3, 3), np.float32)

        hs, ok = 0, 0
        barra = st.progress(0)

        imagenes_info = []


        for i, ruta in enumerate(imagenes_seleccionadas):

            img = cv2.imread(ruta)
            res = model(img, verbose=False)

            info_imagen = {"ruta": ruta, "paneles": []}
            panel_id = 0

            for r in res:
                if r.masks is None or r.boxes is None:
                    continue

                # masks.data: (N, H, W) tipo torch
                masks = r.masks.data.cpu().numpy()  # 0/1 floats normalmente
                boxes = r.boxes.xyxy.cpu().numpy().astype(int)

                for j in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[j]
                    mask_full = (masks[j] > 0.5)  # boolean (H,W)

                    roi_masked, mask_crop = extraer_roi_con_mascara(img, (x1, y1, x2, y2), mask_full)
                    if roi_masked is None:
                        continue

                    # --- (opcional) rotación basada en bordes: mejor hacerlo con roi_masked ---
                    angulo = obtener_angulo_panel(roi_masked)
                    roi_masked = rotar_imagen(roi_masked, angulo)

                    # Nota: al rotar, la máscara original ya no coincide.
                    # Si quieres ser estricto, deberías rotar también la máscara.
                    # Solución rápida: analiza sobre roi_masked asumiendo fondo negro.

                    g = cv2.cvtColor(roi_masked, cv2.COLOR_BGR2GRAY)

                    # si el fondo negro te sesga la media, calcula métricas SOLO donde g>0
                    valid = (g > 0)
                    if np.count_nonzero(valid) < 200:
                        continue

                    # Convolución
                    c = convolve2d(g, kernel, mode="same")

                    # métricas SOLO en región válida
                    c_valid = c[valid]
                    intensidad = float(np.max(c_valid))
                    media = float(np.mean(c_valid))

                    es_hotspot = np.count_nonzero(c_valid > media + 1000) >= 200

                    gravedad = clasificar_gravedad(es_hotspot, intensidad)

                    if es_hotspot:
                        nombre = f"HS_{hs}.jpg"
                        ruta_panel = os.path.join(hs_dir, nombre)
                        cv2.imwrite(ruta_panel, roi_masked)
                        hs += 1
                    else:
                        nombre = f"OK_{ok}.jpg"
                        ruta_panel = os.path.join(ok_dir, nombre)
                        cv2.imwrite(ruta_panel, roi_masked)
                        ok += 1

                    info_imagen["paneles"].append({
                        "id": f"Panel_{panel_id}",
                        "hotspot": es_hotspot,
                        "gravedad": gravedad,
                        "ruta_panel": ruta_panel
                    })

                    panel_id += 1

            imagenes_info.append(info_imagen)

            barra.progress((i + 1) / len(imagenes_seleccionadas))


        st.success(f"✅ Procesado | HS: {hs} | Sanos: {ok}")

        vuelo_nombre = os.path.basename(ruta_origen)
        subida_nombre = fecha_sel

        ruta_pdf = generar_informe_pdf(
            base_res,
            vuelo_nombre,
            subida_nombre,
            imagenes_info
        )

        st.success("📄 Informe generado correctamente")

        with open(ruta_pdf, "rb") as f:
            st.download_button(
                "⬇️ Descargar informe PDF",
                f,
                file_name="Informe_Termografico.pdf"
            )


