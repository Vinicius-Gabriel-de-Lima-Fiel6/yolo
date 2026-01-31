import cv2
import torch
import numpy as np
import streamlit as st
import time
import threading
from datetime import datetime
from ultralytics import YOLO

# ==============================================================================
# CONFIGURAÇÕES DE PÁGINA
# ==============================================================================
st.set_page_config(page_title="VisionGuard AI Pro", layout="wide")

# Inicialização de Variáveis de Estado (Evita que o sistema resete sozinho)
if 'monitoring_active' not in st.session_state:
    st.session_state.monitoring_active = False
if 'model' not in st.session_state:
    # Carrega o modelo com cache para não travar a abertura
    st.session_state.model = YOLO('yolov8n.pt')

# ==============================================================================
# CLASSE DE CAPTURA DE VÍDEO (THREAD-SAFE)
# ==============================================================================
class VideoHandler:
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        self.frame = None
        self.is_running = False
        self.lock = threading.Lock()

    def start(self):
        self.is_running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.1)

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.is_running = False
        if self.cap:
            self.cap.release()

# ==============================================================================
# INTERFACE DO USUÁRIO
# ==============================================================================
def main():
    st.title("🛡️ Sistema de Monitoramento Industrial")
    
    # Barra Lateral
    st.sidebar.header("Configurações de Rede")
    video_source = st.sidebar.text_input("Fonte (0 para Webcam / URL RTSP)", "0")
    conf_threshold = st.sidebar.slider("Confiança Mínima", 0.1, 1.0, 0.4)
    
    col_video, col_info = st.columns([3, 1])

    with col_video:
        video_placeholder = st.empty()
        
        # Lógica de Botão On/Off
        if not st.session_state.monitoring_active:
            if st.button("▶️ INICIAR MONITORAMENTO", use_container_width=True):
                st.session_state.monitoring_active = True
                st.rerun()
        else:
            if st.button("⏹️ PARAR SISTEMA", use_container_width=True):
                st.session_state.monitoring_active = False
                st.rerun()

    with col_info:
        st.subheader("Alertas e Status")
        status_text = st.empty()
        log_text = st.empty()

    # Execução do Processamento (Só roda se o botão START foi clicado)
    if st.session_state.monitoring_active:
        status_text.success("✅ Sistema Online")
        
        # Converte fonte para int se for webcam
        src = int(video_source) if video_source.isdigit() else video_source
        handler = VideoHandler(src)
        handler.start()

        try:
            while st.session_state.monitoring_active:
                frame = handler.get_frame()
                if frame is not None:
                    # IA - Inferência
                    results = st.session_state.model.predict(frame, conf=conf_threshold, verbose=False)
                    annotated_frame = results[0].plot()

                    # Renderiza o frame processado
                    video_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)
                    
                    # Detecção de Incidente Simples
                    if len(results[0].boxes) > 0:
                        log_text.info(f"Objeto detectado às {datetime.now().strftime('%H:%M:%S')}")
                
                # Pequena pausa para o Streamlit processar outros eventos da UI
                time.sleep(0.01)
        finally:
            handler.stop()

if __name__ == "__main__":
    main()
