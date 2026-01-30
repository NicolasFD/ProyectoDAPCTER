# ===============================================================
# 3️⃣ PROCESAMIENTO
# ===============================================================
with tab_proc:
    st.header("🧠 Procesamiento de Imágenes")

    st.markdown("### 📂 Fuente de imágenes")

    fuente = st.radio(
        "¿Qué imágenes deseas procesar?",
        ["📸 Capturas del vuelo", "⬆️ Subir imágenes manualmente"]
    )

    imagenes = []

    # -----------------------------------------------------------
    # OPCIÓN 1: CAPTURAS DEL VUELO
    # -----------------------------------------------------------
    if fuente == "📸 Capturas del vuelo":
        carpeta_capturas = "Capturas"

        if os.path.exists(carpeta_capturas):
            imagenes = [
                os.path.join(carpeta_capturas, f)
                for f in os.listdir(carpeta_capturas)
                if f.lower().endswith((".jpg", ".png"))
            ]

            st.success(f"📸 {len(imagenes)} imágenes encontradas")
        else:
            st.warning("No hay capturas aún")

    # -----------------------------------------------------------
    # OPCIÓN 2: SUBIDA MANUAL
    # -----------------------------------------------------------
    else:
        uploaded_files = st.file_uploader(
            "Seleccionar imágenes",
            type=["jpg", "png"],
            accept_multiple_files=True
        )

        if uploaded_files:
            os.makedirs("Temp_Subidas", exist_ok=True)
            for img in uploaded_files:
                ruta = os.path.join("Temp_Subidas", img.name)
                Image.open(img).save(ruta)
                imagenes.append(ruta)

    st.divider()

    # -----------------------------------------------------------
    # PROCESAMIENTO
    # -----------------------------------------------------------
    if st.button("🚀 Iniciar procesamiento"):

        if not imagenes:
            st.error("❌ No hay imágenes para procesar")
        else:
            from ultralytics import YOLO
            from scipy.signal import convolve2d

            st.info("🔄 Procesando imágenes...")

            ruta_base = "C:/DAPCTER/ProyectoDAPCTER"
            fecha_hoy = datetime.now().strftime("%Y-%m-%d")
            ruta_carpeta = os.path.join(ruta_base, fecha_hoy)

            input_dir = os.path.join(ruta_carpeta, "Subidas")
            hs_dir = os.path.join(ruta_carpeta, "Paneles_HS")
            ok_dir = os.path.join(ruta_carpeta, "Paneles_Sanos")

            os.makedirs(input_dir, exist_ok=True)
            os.makedirs(hs_dir, exist_ok=True)
            os.makedirs(ok_dir, exist_ok=True)

            # Copiar imágenes al input_dir
            for ruta in imagenes:
                nombre = os.path.basename(ruta)
                cv2.imwrite(
                    os.path.join(input_dir, nombre),
                    cv2.imread(ruta)
                )

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
                                f"{hs_dir}/HS_{countHS}.jpg", roi
                            )
                            countHS += 1
                        else:
                            cv2.imwrite(
                                f"{ok_dir}/OK_{countOK}.jpg", roi
                            )
                            countOK += 1

                progress.progress((i + 1) / len(imagenes))

            st.success(
                f"✅ Finalizado | Paneles HS: {countHS} | Paneles sanos: {countOK}"
            )
