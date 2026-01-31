import cv2
import torch
import numpy as np
import streamlit as st
import time
from datetime import datetime
from ultralytics import YOLO

# 1. Configuração de Página (DEVE ser a primeira linha de comando Streamlit)
st.set_page_config(page_title="VisionGuard Pro", layout="wide")

# 2. Cache do Modelo (Para a tela abrir instantaneamente)
@st.cache_resource
def load_yolo():
    # Carrega o modelo nano que é mais leve para nuvem
    return YOLO('yolov8n.pt')

# 3. Inicialização do Estado
if 'run_system' not in st.session_state:
    st.session_state.run_system = False

# 4. Interface Visual (Desenhada ANTES de qualquer loop)
st.title("🛡️ VisionGuard AI | Painel Industrial")

with st.sidebar:
    st.header("⚙️ Configurações")
    video_source = st.text_input("Fonte (RTSP URL ou 0 para Webcam)", "0")
    conf_level = st.slider("Confiança", 0.1, 1.0, 0.4)
    st.markdown("---")
    
    # Botões de Controle
    if not st.session_state.run_system:
        if st.button("▶️ LIGAR SISTEMA", use_container_width=True, type="primary"):
            st.session_state.run_system = True
            st.rerun()
    else:
        if st.button("⏹️ DESLIGAR", use_container_width=True):
            st.session_state.run_system = False
            st.rerun()

col_v, col_l = st.columns([3, 1])

with col_v:
    st_frame = st.empty() # Espaço reservado para o vídeo
    if not st.session_state.run_system:
        st.info("Sistema em standby. Clique em 'Ligar' para iniciar o monitoramento.")

with col_l:
    st.subheader("Eventos")
    log_area = st.empty()

# 5. Lógica de Processamento (Só roda se st.session_state.run_system for True)
if st.session_state.run_system:
    model = load_yolo()
    
    # Tratamento da fonte
    src = int(video_source) if video_source.isdigit() else video_source
    cap = cv2.VideoCapture(src)
    
    # Verifica se a câmera abriu
    if not cap.isOpened():
        st.error(f"Não foi possível conectar à fonte: {video_source}")
        st.session_state.run_system = False
    else:
        try:
            while st.session_state.run_system:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Falha ao capturar frame. Tentando reconectar...")
                    break
                
                # Inferência IA
                results = model.predict(frame, conf=conf_level, verbose=False)
                annotated_frame = results[0].plot()
                
                # Renderiza na tela
                st_frame.image(annotated_frame, channels="BGR", use_container_width=True)
                
                # Log de detecção
                if len(results[0].boxes) > 0:
                    log_area.caption(f"Detecção ativa: {datetime.now().strftime('%H:%M:%S')}")
                
                # Pequena pausa para permitir que o Streamlit interaja com a UI
                time.sleep(0.01)
                
        except Exception as e:
            st.error(f"Erro no processamento: {e}")
        finally:
            cap.release()
