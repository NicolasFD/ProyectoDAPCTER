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

def resize_mask_to_image(mask_u8, img):
    """Asegura que la máscara sea (H,W) igual a la imagen."""
    H, W = img.shape[:2]
    if mask_u8.shape[:2] != (H, W):
        mask_u8 = cv2.resize(mask_u8, (W, H), interpolation=cv2.INTER_NEAREST)
    return mask_u8

def guardar_instancia(base_dir, instancia_id, roi, mask_roi_u8, roi_masked):
    """
    Crea carpeta panel_X y guarda:
    - mask.png (recortada al ROI)
    - roi_original.png
    - roi_masked.png (fondo negro)
    - panel_limpio.png (igual a roi_masked)
    Devuelve ruta_panel_limpio.
    """
    carpeta = os.path.join(base_dir, f"panel_{instancia_id}")
    os.makedirs(carpeta, exist_ok=True)

    cv2.imwrite(os.path.join(carpeta, "mask.png"), mask_roi_u8 * 255)
    cv2.imwrite(os.path.join(carpeta, "roi_original.png"), roi)
    cv2.imwrite(os.path.join(carpeta, "roi_masked.png"), roi_masked)
    cv2.imwrite(os.path.join(carpeta, "panel_limpio.png"), roi_masked)

    return os.path.join(carpeta, "panel_limpio.png")

def dibujar_contorno_y_label(img_bgr, mask_full_u8, label_text, color=(0,255,0)):
    """Dibuja contorno + etiqueta sobre la imagen completa."""
    contours, _ = cv2.findContours(mask_full_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_bgr

    cv2.drawContours(img_bgr, contours, -1, color, 2)

    # ubicar etiqueta en esquina superior del bbox del contorno
    ys, xs = np.where(mask_full_u8 == 1)
    if len(xs) > 0 and len(ys) > 0:
        x1, y1 = int(xs.min()), int(ys.min())
        cv2.putText(img_bgr, label_text, (x1, max(0, y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return img_bgr

def detectar_hotspot_local_robusto(
    roi_bgr,
    valid_mask_bool,
    kernel_size=51,
    blur_mode="gauss",
    blur_ksize=7,
    k_sigma=2.2,
    min_area=40,
    close_ksize=5,
    panel_mask_u8=None
):
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    if blur_mode != "none":
        k = max(1, int(blur_ksize))
        if blur_mode == "gauss":
            if k % 2 == 0:
                k += 1
            gray = cv2.GaussianBlur(gray, (k, k), 0)
        elif blur_mode == "box":
            gray = cv2.blur(gray, (k, k))

    kbg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    fondo = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kbg)
    diff = cv2.subtract(gray, fondo)

    valid_u8 = (valid_mask_bool.astype(np.uint8) * 255)
    vals = diff[valid_u8 > 0]

    if vals.size < 50:
        return np.zeros_like(valid_u8), diff, [], 0.0, 0

    mu = float(np.mean(vals))
    sigma = float(np.std(vals))
    thr = mu + k_sigma * sigma

    mask = np.zeros_like(valid_u8)
    mask[(diff >= thr) & (valid_u8 > 0)] = 255

    if close_ksize > 1:
        kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kc, iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    good = []
    area_total = 0.0

    for lab in range(1, num):
        x, y, w, h, area = stats[lab]

        if area < min_area:
            continue

        comp_mask = (labels == lab).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not cnts:
            continue

        c = max(cnts, key=cv2.contourArea)

        area_cont = cv2.contourArea(c)
        if area_cont < min_area:
            continue

        # convexidad
        hull = cv2.convexHull(c)
        area_hull = cv2.contourArea(hull)
        if area_hull <= 0:
            continue
        solidity = area_cont / float(area_hull)

        # bounding box y extent
        x_c, y_c, w_c, h_c = cv2.boundingRect(c)
        box_area = float(w_c * h_c)
        if box_area <= 0:
            continue
        extent = area_cont / box_area

        # relación de aspecto
        aspect_ratio = max(w_c / float(h_c + 1e-5), h_c / float(w_c + 1e-5))

        # =====================================================
        # FILTRO COMBINADO POR FORMA
        # =====================================================

        # 1) muy convexa -> fuera
        if solidity > 0.88:
            continue

        # 2) demasiado "rellena" dentro de su caja -> fuera
        if extent > 0.65:
            continue

        # 3) demasiado alargada -> fuera
        if aspect_ratio > 4.0:
            continue
        # centroide del componente
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        Hroi, Wroi = roi_bgr.shape[:2]

        # descartar componentes demasiado arriba o demasiado abajo
        if cy < Hroi * 0.15:
            continue
        if cy > Hroi * 0.85:
            continue

        good.append(lab)
        area_total += float(area_cont)

        

    mask_big = np.zeros_like(mask)
    for lab in good:
        mask_big[labels == lab] = 255

    cnts_final, _ = cv2.findContours(mask_big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return mask_big, diff, cnts_final, area_total, len(cnts_final)
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
        # Prioridad: panel anotado (contorno manchas) > panel limpio
        # -------------------------
        ruta_panel_img = datos.get("ruta_panel_ann") or datos.get("ruta_panel")

        if ruta_panel_img and os.path.exists(ruta_panel_img):
            ANCHO_PANEL = 40
            self.image(
                ruta_panel_img,
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
            ("Panel (imagen marcada)", datos["panel"]),
            ("Punto caliente", datos["punto_caliente"]),
            ("Gravedad", datos["gravedad"]),
            ("Area manchas", round(float(datos.get("area_manchas", 0.0)), 1)),
            ("Num manchas", int(datos.get("num_manchas", 0))),
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

        ruta_img = img_info.get("ruta_marcada", img_info["ruta"])
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
                "panel": panel.get("panel_code", panel["id"]),
                "punto_caliente": "SI" if panel["hotspot"] else "NO",
                "gravedad": gravedad,
                "recomendacion": obtener_recomendacion(gravedad),

                "ruta_panel": panel.get("ruta_panel"),
                "ruta_panel_ann": panel.get("ruta_panel_ann"),

                # ✅ agrega esto:
                "area_manchas": panel.get("area_manchas", 0.0),
                "num_manchas": panel.get("num_manchas", 0),
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
        inst_dir = os.path.join(base_res, "Instancias") 
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        marked_dir = os.path.join(base_res, "Marcadas", run_id)
        os.makedirs(marked_dir, exist_ok=True)     # panel_1, panel_2...     # imagen marcada por cada foto
        os.makedirs(inst_dir, exist_ok=True)
        os.makedirs(hs_dir, exist_ok=True)
        os.makedirs(ok_dir, exist_ok=True)

        model = YOLO("best-seg.pt")
        kernel = np.ones((3, 3), np.float32)

        hs, ok = 0, 0
        barra = st.progress(0)

        imagenes_info = []


        for i, ruta in enumerate(imagenes_seleccionadas):

            img = cv2.imread(ruta)
            img_marked = img.copy()
            res = model(img, verbose=False)

            info_imagen = {"ruta": ruta, "paneles": []}
            panel_id = 0

            for r in res:
                if r.masks is None or r.masks.data is None:
                    continue

                masks = r.masks.data.detach().cpu().numpy()  # (N,H,W) floats
                H, W = img.shape[:2]

                # =========================================================
                # ✅ 1) Construir lista de instancias (bbox + máscara)
                # =========================================================
                instances = []
                for j in range(masks.shape[0]):
                    mask_full = (masks[j] > 0.5).astype(np.uint8)

                    # asegurar tamaño máscara = tamaño imagen
                    if mask_full.shape[:2] != (H, W):
                        mask_full = cv2.resize(mask_full, (W, H), interpolation=cv2.INTER_NEAREST)

                    # filtrar ruido
                    if int(mask_full.sum()) < 500:
                        continue

                    ys, xs = np.where(mask_full == 1)
                    if xs.size == 0 or ys.size == 0:
                        continue

                    x1, x2 = int(xs.min()), int(xs.max())
                    y1, y2 = int(ys.min()), int(ys.max())

                    instances.append((x1, x2, y1, y2, mask_full))

                # ✅ 2) Orden izquierda→derecha por x1
                instances.sort(key=lambda t: t[0])

                # ✅ 3) Recorremos ya ordenado
                for (x1, x2, y1, y2, mask_full) in instances:
                    roi = img[y1:y2+1, x1:x2+1]
                    mask_roi = mask_full[y1:y2+1, x1:x2+1].astype(np.uint8)

                    if roi.size == 0 or mask_roi.sum() < 50:
                        continue

                    # panel limpio (fondo negro)
                    roi_masked = cv2.bitwise_and(roi, roi, mask=mask_roi)

                    mask_panel_u8 = (mask_roi > 0).astype(np.uint8)
                    k_in = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # margen ~1px
                    valid = cv2.erode(mask_panel_u8, k_in, iterations=1).astype(bool)

                    if np.count_nonzero(valid) < 200:
                        continue

                    # DETECCIÓN
                    mask_big, resp, cnts, area_total, num_manchas = detectar_hotspot_local_robusto(
                        roi_masked,
                        valid_mask_bool=valid,
                        kernel_size=51,
                        blur_mode="gauss",
                        blur_ksize=7,
                        k_sigma=2.2,
                        min_area=40,
                        close_ksize=5,
                        panel_mask_u8=mask_roi
                    )

                    roi_annotated = roi_masked.copy()
                    if cnts:
                        cv2.drawContours(roi_annotated, cnts, -1, (0,0,255), 2)

                    # dibujar contornos sobre imagen completa (ROI -> imagen)
                    for c in cnts:
                        c2 = c.copy()
                        c2[:, 0, 0] += x1
                        c2[:, 0, 1] += y1
                        cv2.drawContours(img_marked, [c2], -1, (0,0,255), 2)

                    # Intensidad basada en "diff" (resp)
                    # Intensidad basada en respuesta térmica
                    intensidad = float(np.max(resp[valid])) if np.any(valid) else 0.0

                    es_hotspot = (num_manchas >= 1) and (
                        (area_total >= 120) or
                        (intensidad >= 1600 and area_total >= 60)
                    )

                    # clasificación de gravedad
                    gravedad = clasificar_gravedad(es_hotspot, intensidad)

                    # Guardar HS/OK...
                    if es_hotspot:
                        nombre = f"HS_{hs}.jpg"
                        ruta_panel = os.path.join(hs_dir, nombre)
                        nombre_ann = f"HS_{hs}_ann.jpg"
                        ruta_panel_ann = os.path.join(hs_dir, nombre_ann)
                        cv2.imwrite(ruta_panel, roi_masked)
                        cv2.imwrite(ruta_panel_ann, roi_annotated)
                        hs += 1
                    else:
                        nombre = f"OK_{ok}.jpg"
                        ruta_panel = os.path.join(ok_dir, nombre)
                        nombre_ann = f"OK_{ok}_ann.jpg"
                        ruta_panel_ann = os.path.join(ok_dir, nombre_ann)
                        cv2.imwrite(ruta_panel, roi_masked)
                        cv2.imwrite(ruta_panel_ann, roi_annotated)
                        ok += 1

                    # contorno del PANEL en la imagen marcada
                    contours_panel, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(img_marked, contours_panel, -1, (0, 255, 0), 2)

                    # Etiqueta P{panel_id} HS/OK (ahora estable)
                    cv2.putText(img_marked, f"P{panel_id} {'HS' if es_hotspot else 'OK'}",
                                (x1, max(0, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

                    panel_code = f"P{panel_id}"
                    info_imagen["paneles"].append({
                        "panel_code": panel_code,
                        "id": f"Panel_{panel_id}",
                        "hotspot": es_hotspot,
                        "gravedad": gravedad,
                        "ruta_panel": ruta_panel,
                        "ruta_panel_ann": ruta_panel_ann,
                        "area_manchas": area_total,
                        "num_manchas": num_manchas,
                        "intensidad_resp": intensidad
                    })

                    panel_id += 1

            imagenes_info.append(info_imagen)
            nombre_img = os.path.basename(ruta)
            out_marked = os.path.join(marked_dir, f"marked_{os.path.basename(ruta)}")
            cv2.imwrite(out_marked, img_marked)
            info_imagen["ruta_marcada"] = out_marked

            barra.progress((i + 1) / len(imagenes_seleccionadas))


        st.success(f"✅ Procesado | HS: {hs} | Sanos: {ok}")
        st.image(cv2.cvtColor(img_marked, cv2.COLOR_BGR2RGB), caption=f"Marcada: {os.path.basename(ruta)}", use_container_width=True)
        
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


