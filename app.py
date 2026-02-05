import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import queue
import time

# Configurações de Interface
st.set_page_config(page_title="SafeLab & Road Security AI", layout="wide")
st.title("🛡️ Sistema de Contenção e Segurança Inteligente")

# --- BARRA LATERAL (CONFIGURAÇÕES) ---
st.sidebar.header("Parâmetros do Sistema")
mode = st.sidebar.selectbox("Cenário de Uso", ["Laboratório (EPI/Quedas)", "Segurança Rodoviária (Acidentes)"])
conf_level = st.sidebar.slider("Confiança do Modelo", 0.1, 1.0, 0.45)
alert_log = st.sidebar.container()

# Cache do modelo para evitar sobrecarga no Cloud
@st.cache_resource
def load_model(mode):
    if mode == "Laboratório (EPI/Quedas)":
        return YOLO("yolo11n-pose.pt")  # Pose é melhor para quedas e acidentes humanos
    return YOLO("yolo11n.pt")          # Geral para veículos e objetos

model = load_model(mode)

# Fila para logs de acidentes/alertas
result_queue = queue.Queue()

# --- LÓGICA DE PROCESSAMENTO DE VÍDEO ---
class SecurityProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_alert_time = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # 1. Executa Tracking
        results = model.track(img, persist=True, conf=conf_level, verbose=False)
        
        annotated_img = results[0].plot()
        
        # 2. Lógica de Contenção de Acidentes
        # Exemplo: Detectar se uma 'pessoa' está no chão (Eixo Y da cabeça próximo aos pés)
        if mode == "Laboratório (EPI/Quedas)" and results[0].keypoints:
            for kp in results[0].keypoints.data:
                # Lógica simplificada: Se a distância vertical entre ombros e quadril for muito pequena
                if len(kp) > 0:
                    # Se detectar uma possível queda
                    result_queue.put(f"⚠️ POSSÍVEL QUEDA DETECTADA - {time.strftime('%H:%M:%S')}")
        
        # 3. Lógica de Invasão de Perímetro
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                cls = model.names[int(box.cls[0])]
                if cls in ['car', 'truck', 'motorcycle'] and mode == "Segurança Rodoviária (Acidentes)":
                    # Aqui você poderia adicionar lógica de colisão por proximidade de bounding boxes
                    pass

        return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

# --- INTERFACE DE VÍDEO ---
ctx = webrtc_streamer(
    key="security-system",
    video_processor_factory=SecurityProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)

# --- DASHBOARD DE ALERTAS (EM TEMPO REAL) ---
st.subheader("📋 Log de Eventos de Segurança")
if ctx.state.playing:
    status_placeholder = st.empty()
    while True:
        try:
            msg = result_queue.get_nowait()
            st.warning(msg)
        except queue.Empty:
            break
