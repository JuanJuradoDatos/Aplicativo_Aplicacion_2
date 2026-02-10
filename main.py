# -----------------------------
# Imports
# -----------------------------
import streamlit as st
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import numpy as np

# -----------------------------
# Paths (robustos en cloud)
# -----------------------------
ROOT = Path(__file__).resolve().parent

IMAGES_DIR = ROOT / "images"
MODEL_DIR  = ROOT / "weights"

DEFAULT_IMAGE = IMAGES_DIR / "malignant (94).png"
DEFAULT_DETECT_IMAGE = IMAGES_DIR / "malignant (94)_0.png"

DETECTION_MODEL = MODEL_DIR / "modelo_guardado.pt"

# -----------------------------
# Streamlit config
# -----------------------------
st.set_page_config(page_title="Breast cancer detection (YOLO)", page_icon="𝑓(𝑥)", layout="wide")
st.header("Breast cancer detection app — Juan David Jurado Tapias")

# -----------------------------
# Cache del modelo (CLAVE para RAM)
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_model(model_path: str):
    # Forzamos CPU para estabilidad en cloud free
    model = YOLO(model_path)
    return model

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Configuración")

confidence_value = st.sidebar.slider("Confianza", 0.05, 0.95, 0.40, 0.05)

# (RECOMENDADO) Deja solo Detection en producción
model_path = DETECTION_MODEL

if not model_path.exists():
    st.error(f"No encuentro el modelo en: {model_path}")
    st.stop()

# Carga 1 vez por sesión (gracias al cache)
model = load_model(str(model_path))

st.sidebar.success("Modelo cargado ✅ (cacheado)")

# -----------------------------
# Fuente: Imagen
# -----------------------------
st.sidebar.header("Entrada")
source_image = st.sidebar.file_uploader("Sube una imagen", type=("jpg", "png", "jpeg", "bmp", "webp"))

col1, col2 = st.columns(2)

with col1:
    try:
        if source_image is None:
            img = Image.open(DEFAULT_IMAGE).convert("RGB")
            st.image(img, caption="Imagen por defecto", use_container_width=True)
        else:
            img = Image.open(source_image).convert("RGB")
            st.image(img, caption="Imagen subida", use_container_width=True)
    except Exception as e:
        st.error("Error abriendo la imagen")
        st.exception(e)
        st.stop()

with col2:
    if source_image is None:
        try:
            st.image(Image.open(DEFAULT_DETECT_IMAGE), caption="Ejemplo detectado", use_container_width=True)
        except Exception:
            st.info("No hay ejemplo detectado disponible.")
    else:
        # Botón evita reruns de inferencia por cada cambio
        if st.sidebar.button("Detectar", type="primary"):
            # 1) convertir a numpy y reducir tamaño para ahorrar RAM
            img_np = np.array(img)

            # Reducción simple: limita a 640 px en lado mayor
            h, w = img_np.shape[:2]
            max_side = max(h, w)
            if max_side > 640:
                scale = 640 / max_side
                new_w, new_h = int(w * scale), int(h * scale)
                # PIL resize (más liviano que cv2 aquí)
                img_small = img.resize((new_w, new_h))
                img_np = np.array(img_small)

            with st.spinner("Ejecutando detección..."):
                # 2) predict en CPU y sin streaming extra
                results = model.predict(
                    source=img_np,
                    conf=float(confidence_value),
                    device="cpu",
                    verbose=False
                )

            # 3) Plot (esto crea un array) → convertir a RGB si viene en BGR
            plotted = results[0].plot()
            # Ultralytics suele devolver BGR
            plotted = plotted[..., ::-1]

            st.image(plotted, caption="Resultado", use_container_width=True)

            # Resultados en expander (ligero)
            with st.expander("Resultados (boxes)"):
                boxes = results[0].boxes
                if boxes is None or len(boxes) == 0:
                    st.write("No se detectaron objetos.")
                else:
                    # Mostrar de forma compacta
                    for b in boxes:
                        st.write(b.data.cpu().numpy())
