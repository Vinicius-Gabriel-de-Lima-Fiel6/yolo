import streamlit as st
import av
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer

st.set_page_config(page_title="SafeLab AI", layout="wide")

# Inicialização do modelo com tratamento de erro
@st.cache_resource
def load_model():
    try:
        return YOLO("yolo11n.pt") 
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {e}")
        return None

model = load_model()

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    # Processamento com YOLO
    # conf=0.5 evita falsos positivos que pesam no processamento
    results = model(img, conf=0.5, verbose=False)
    
    # Desenha os resultados
    annotated_img = results[0].plot()

    return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

st.title("🛡️ Monitoramento de Segurança v3")

if model:
    webrtc_streamer(
        key="safety-monitor",
        video_frame_callback=video_frame_callback,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
else:
    st.error("O sistema não pôde iniciar porque o modelo YOLO não foi carregado.")
