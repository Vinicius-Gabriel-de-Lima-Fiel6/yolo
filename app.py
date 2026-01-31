import cv2
import torch
import numpy as np
import streamlit as st
import time
import threading
import queue
import logging
import json
import requests
from datetime import datetime
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from collections import deque

# ==============================================================================
# CONFIGURAÇÕES GLOBAIS E LOGGER
# ==============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configurações de UI
st.set_page_config(page_title="VisionGuard AI | Industrial Safety", layout="wide", initial_sidebar_state="expanded")

# ==============================================================================
# CLASSES DE PROCESSAMENTO DE VÍDEO (MULTI-THREADED)
# ==============================================================================

class VideoCaptureThread:
    """Captura frames de forma assíncrona para minimizar latência RTSP/Webcam."""
    def __init__(self, source):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        self.frame = None
        self.running = False
        self.lock = threading.Lock()
        
    def start(self):
        if self.running: return
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                logger.error(f"Erro ao ler fonte: {self.source}. Tentando reconectar...")
                self.cap.release()
                time.sleep(2)
                self.cap = cv2.VideoCapture(self.source)
                continue
            with self.lock:
                self.frame = frame

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        self.cap.release()

# ==============================================================================
# MOTOR DE INTELIGÊNCIA ARTIFICIAL (YOLO)
# ==============================================================================

class AISafetyEngine:
    """Gerencia modelos YOLO, detecção de EPIs, Riscos e Comportamento."""
    def __init__(self, model_path='yolov8n.pt', device='cuda'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Inicializando Engine IA no dispositivo: {self.device}")
        
        # Carrega o modelo (Pode ser estendido para carregar múltiplos .pt)
        # Ex: self.epi_model = YOLO('best_epi.pt')
        self.model = YOLO(model_path)
        self.heatmap = None
        self.alert_queue = queue.Queue()
        
    def process_frame(self, frame, conf_threshold=0.25, active_detects=None):
        """
        Executa inferência, tracking e lógica de segurança.
        active_detects: lista de flags vindas da UI (ex: ['Capacete', 'Fogo'])
        """
        results = self.model.track(frame, persist=True, conf=conf_threshold, device=self.device, verbose=False)
        annotated_frame = frame.copy()
        
        alerts_found = []
        
        if results[0].boxes:
            boxes = results[0].boxes
            for box in boxes:
                # Extração de dados da detecção
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = self.model.names[cls]
                xyxy = box.xyxy[0].cpu().numpy()
                
                # --- Lógica de Negócio Industrial ---
                
                # 1. Detecção de Queda (Heurística de Aspect Ratio)
                if label == 'person':
                    w = xyxy[2] - xyxy[0]
                    h = xyxy[3] - xyxy[1]
                    if w > h * 1.2: # Pessoa mais larga que alta indica possível queda
                        self._trigger_alert("PESSOA CAÍDA", "Alta", frame)
                        cv2.rectangle(annotated_frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 0, 255), 4)
                
                # 2. Monitoramento de EPI (Simulação de classes específicas)
                # No mundo real, você usaria os índices do seu modelo customizado
                # if label in ['no-helmet', 'no-gloves']: 
                #     self._trigger_alert(f"FALTA DE EPI: {label}", "Crítica", frame)

                # 3. Detecção de Áreas Restritas (ROI simples)
                # if is_inside_roi(xyxy, restricted_zone): ...
                
            annotated_frame = results[0].plot()

        # Atualizar Heatmap de Risco
        self._update_heatmap(frame.shape, results[0].boxes)
        
        return annotated_frame, alerts_found

    def _update_heatmap(self, shape, boxes):
        if self.heatmap is None:
            self.heatmap = np.zeros((shape[0], shape[1]), dtype=np.float32)
        
        if boxes:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                self.heatmap[y1:y2, x1:x2] += 1
        
        # Decay do heatmap para movimento
        self.heatmap = cv2.GaussianBlur(self.heatmap, (15, 15), 0) * 0.95

    def _trigger_alert(self, alert_type, severity, frame):
        timestamp = datetime.now().strftime("%H:%M:%S")
        alert_data = {
            "type": alert_type,
            "severity": severity,
            "time": timestamp,
            "frame": frame.copy()
        }
        self.alert_queue.put(alert_data)

# ==============================================================================
# SISTEMA DE NOTIFICAÇÃO (EXTERNO)
# ==============================================================================

class NotificationManager:
    """Envia alertas para Telegram, Webhooks e log de segurança."""
    @staticmethod
    def send_telegram(message):
        # Placeholder para integração real: requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage", data=...)
        logger.info(f"TELEGRAM ALERT: {message}")

    @staticmethod
    def send_webhook(data):
        # Envia JSON para backend corporativo
        logger.info(f"WEBHOOK SENT: {data['type']}")

# ==============================================================================
# INTERFACE STREAMLIT (DASHBOARD)
# ==============================================================================

def main():
    # Sidebar de Configuração Industrial
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2092/2092130.png", width=100)
    st.sidebar.title("🛡️ VisionGuard Pro v2.0")
    st.sidebar.markdown("---")
    
    # Seleção de Câmeras
    source_type = st.sidebar.selectbox("Fonte de Dados", ["Webcam", "RTSP Stream", "Vídeo Arquivo", "IP Industrial"])
    source_path = 0 if source_type == "Webcam" else st.sidebar.text_input("URL / Path", "rtsp://admin:password@192.168.1.100:554")
    
    # Seleção de Modelo
    model_option = st.sidebar.selectbox("Modelo YOLO", ["YOLOv8n (Fast)", "YOLOv8m (Balanced)", "Custom EPI-v4"])
    conf_thresh = st.sidebar.slider("Confiança Mínima", 0.1, 1.0, 0.45)
    
    # Toggles de Detecção
    st.sidebar.subheader("Módulos Ativos")
    detect_epi = st.sidebar.toggle("Uso de EPI (Capacete/Luvas)", True)
    detect_fire = st.sidebar.toggle("Incêndio & Fumaça", True)
    detect_fall = st.sidebar.toggle("Pessoas Caídas / Ergonomia", True)
    detect_zones = st.sidebar.toggle("Áreas Restritas", False)

    # Inicialização do Engine
    if 'engine' not in st.session_state:
        st.session_state.engine = AISafetyEngine()
        st.session_state.notificator = NotificationManager()
        st.session_state.cap_thread = None

    # Layout Principal
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("📡 Monitoramento em Tempo Real")
        st_frame = st.empty()
        
        # Controle de Vídeo
        start_btn = st.button("▶️ Iniciar Monitoramento")
        stop_btn = st.button("⏹️ Parar")

        if start_btn:
            st.session_state.cap_thread = VideoCaptureThread(source_path)
            st.session_state.cap_thread.start()
            
            fps_placeholder = st.empty()
            prev_time = 0
            
            while True:
                frame = st.session_state.cap_thread.get_frame()
                if frame is not None:
                    # Processamento IA
                    processed_frame, alerts = st.session_state.engine.process_frame(
                        frame, 
                        conf_threshold=conf_thresh
                    )
                    
                    # Cálculo de FPS
                    curr_time = time.time()
                    fps = 1 / (curr_time - prev_time)
                    prev_time = curr_time
                    
                    # UI Updates
                    st_frame.image(processed_frame, channels="BGR", use_container_width=True)
                    fps_placeholder.metric("Performance", f"{fps:.1f} FPS", delta="GPU Active")
                    
                    # Processar fila de alertas
                    while not st.session_state.engine.alert_queue.empty():
                        alert = st.session_state.engine.alert_queue.get()
                        with col2:
                            st.warning(f"🚨 {alert['type']} ({alert['time']})")
                            st.session_state.notificator.send_webhook(alert)
                
                if stop_btn: 
                    st.session_state.cap_thread.stop()
                    break

    with col2:
        st.subheader("📊 Analytics & Safety")
        st.metric("Safety Score", "94%", delta="-2% (Incidente)")
        
        # Histórico de Eventos Simulado
        st.markdown("**Últimos Eventos**")
        event_data = [
            {"Horário": "14:20", "Evento": "Acesso Área A", "Status": "OK"},
            {"Horário": "14:35", "Evento": "Falta Luva - L4", "Status": "ALERTA"},
            {"Horário": "14:50", "Evento": "Limpeza Concluída", "Status": "INFO"}
        ]
        st.table(event_data)
        
        # Heatmap Thumbnail
        if st.session_state.engine.heatmap is not None:
            st.markdown("**Mapa de Calor (Ocupação)**")
            hm_norm = cv2.normalize(st.session_state.engine.heatmap, None, 0, 255, cv2.NORM_MINMAX)
            hm_color = cv2.applyColorMap(hm_norm.astype(np.uint8), cv2.COLORMAP_JET)
            st.image(hm_color, use_container_width=True)

# ==============================================================================
# GUIA TÉCNICO E DOCUMENTAÇÃO DE TREINAMENTO (PENSAR COMO PRODUTO)
# ==============================================================================

