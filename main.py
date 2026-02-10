# -----------------------------
# Imports (ligeros)
# -----------------------------
import os
import streamlit as st
from pathlib import Path
from PIL import Image
import numpy as np

# Import diferido: evita que la app muera si ultralytics/cv2 falla
try:
    from ultralytics import YOLO
except Exception as e:
    st.error("❌ Error importando Ultralytics/YOLO. Esto casi siempre es OpenCV (cv2) mal instalado.")
    st.exception(e)
    st.info("✅ Solución: en requirements.txt usa 'opencv-python-headless' y elimina 'opencv-python'.")
    st.stop()

# -----------------------------
# Ajustes de entorno (CPU + menos ruido)
# -----------------------------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

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
    # Carga 1 sola vez por sesión
    model = YOLO(model_path)
    return model

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Configuración")
confidence_value = st.sidebar.slider("Confianza", 0.05, 0.95, 0.40, 0.05)

# Solo Detection en producción (más estable)
model_path = DETECTION_MODEL

if not model_path.exists():
    st.error(f"❌ No encuentro el modelo en: {model_path}")
    st.stop()

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
        if st.sidebar.button("Detectar", type="primary"):
            # Convertir a numpy y bajar tamaño para ahorrar RAM
            h, w = img.size[1], img.size[0]
            max_side = max(h, w)
            if max_side > 640:
                scale = 640 / max_side
                new_w, new_h = int(w * scale), int(h * scale)
                img = img.resize((new_w, new_h))

            img_np = np.array(img)

            with st.spinner("Ejecutando detección..."):
                results = model.predict(
                    source=img_np,
                    conf=float(confidence_value),
                    device="cpu",
                    verbose=False
                )

            plotted = results[0].plot()[..., ::-1]  # BGR->RGB
            st.image(plotted, caption="Resultado", use_container_width=True)

            with st.expander("Resultados (boxes)"):
                boxes = results[0].boxes
                if boxes is None or len(boxes) == 0:
                    st.write("No se detectaron objetos.")
                else:
                    for b in boxes:
                        st.write(b.data.cpu().numpy())
