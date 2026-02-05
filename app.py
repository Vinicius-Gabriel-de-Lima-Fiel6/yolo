import streamlit as st
from ultralytics import YOLO
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

st.set_page_config(page_title="Lab Security Cloud", layout="wide")

# Cache do modelo para não estourar a memória do Cloud
@st.cache_resource
def load_yolo():
    # No Cloud, o modelo 'n' (Nano) é melhor para evitar lentidão
    return YOLO("yolo11n.pt") 

model = load_yolo()

st.title("🛡️ VisionGuard Cloud - Monitoramento Remoto")
st.write("Sistema YOLO rodando via Streamlit Cloud")

class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Processamento YOLO
        # Usamos stream=True para melhor performance em servidores
        results = model.track(img, persist=True, verbose=False)
        
        # Desenha as anotações
        annotated_img = results[0].plot()

        return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

# Componente de vídeo para Web (Navegador)
webrtc_streamer(
    key="yolo-security",
    video_processor_factory=VideoProcessor,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)

st.info("No Cloud, o encerramento é feito parando o streaming no botão 'Stop' acima.")
