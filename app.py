import streamlit as st
from ultralytics import YOLO
import cv2
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# Configuração da página
st.set_page_config(page_title="AI Safety System", layout="wide")
st.title("🛡️ Sistema de Segurança e Contenção")

# Carregamento do modelo (usando Nano para não estourar a RAM do Cloud)
@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt") 

model = load_model()

# Classe que processa o vídeo frame a frame
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        # Transforma o frame do navegador em formato que o OpenCV entende
        img = frame.to_ndarray(format="bgr24")

        # Executa a detecção (Apenas 1 modelo para garantir performance)
        # stream=True ajuda a economizar memória
        results = model(img, stream=True, conf=0.4)
        
        annotated_img = img.copy()
        for r in results:
            annotated_img = r.plot() # Desenha as caixas e classes

        # Retorna o frame processado de volta para o seu navegador
        return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

# Interface do Streamlit
st.sidebar.info("O processamento é feito no servidor e o resultado enviado para sua tela.")

# Componente de vídeo WebRTC
webrtc_streamer(
    key="security-system",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False}, # Não pede microfone
    async_processing=True, # Importante para não travar o vídeo
)

st.write("---")
st.write("Dica: Se o vídeo não aparecer, verifique se você deu permissão de câmera ao seu navegador.")
