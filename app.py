import cv2
import torch
import numpy as np
import streamlit as st
import time
from ultralytics import YOLO

# Configuração Base
st.set_page_config(page_title="VisionGuard Pro", layout="wide")

# 1. Carregar Modelo com Cache (Evita lentidão)
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

# 2. Interface Limpa
st.title("🛡️ VisionGuard AI | Monitoramento Industrial")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Painel de Controle")
    # Se deixar '0', o sistema vai tentar a webcam, se falhar, vai pro vídeo demo
    video_source = st.text_input("Fonte (0=Webcam, ou URL de vídeo)", "0")
    conf_level = st.slider("Sensibilidade da IA", 0.1, 1.0, 0.4)
    
    st.info("Nota: No Streamlit Cloud, use um link de vídeo (URL) ou rode localmente para usar sua webcam.")
    
    run_btn = st.button("▶️ INICIAR SISTEMA", use_container_width=True, type="primary")
    stop_btn = st.button("⏹️ PARAR", use_container_width=True)

# 3. Área de exibição
col_main, col_logs = st.columns([3, 1])
with col_main:
    st_frame = st.empty()
with col_logs:
    st.subheader("📋 Log de Eventos")
    log_output = st.empty()

# 4. Lógica de Execução Blindada
if run_btn:
    model = load_model()
    
    # Tenta abrir a fonte escolhida
    source = 0 if video_source == "0" else video_source
    cap = cv2.VideoCapture(source)
    
    # SE A CAMÊRA FALHAR (Comum no Cloud), usamos um vídeo demo automático
    if not cap.isOpened():
        st.warning("Webcam não detectada no servidor. Carregando vídeo de demonstração...")
        demo_url = "https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/coco_test_video.mp4"
        cap = cv2.VideoCapture(demo_url)

    # Loop de Processamento
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                # Se for vídeo, volta pro início (loop infinito)
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # IA detectando em tempo real
            results = model.predict(frame, conf=conf_level, verbose=False)
            annotated_frame = results[0].plot()
            
            # Mostra na tela
            st_frame.image(annotated_frame, channels="BGR", use_container_width=True)
            
            # Log de detecção
            if len(results[0].boxes) > 0:
                log_output.write(f"✅ {len(results[0].boxes)} objetos detectados.")
            
            # Interrupção pelo botão Stop (não funciona bem dentro do while, 
            # mas o Streamlit vai resetar ao clicar em botões)
            time.sleep(0.01)
            
    except Exception as e:
        st.error(f"Erro: {e}")
    finally:
        cap.release()
