import cv2
import torch
import numpy as np
import streamlit as st
import time
import threading
import queue
import logging
import requests
from datetime import datetime
from ultralytics import YOLO
from PIL import Image

# ==============================================================================
# CONFIGURAÇÕES TÉCNICAS E LOGGER
# ==============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="VisionGuard AI Pro | Industrial Safety",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# SISTEMA DE CAPTURA ASSÍNCRONA (MULTI-THREADED)
# ==============================================================================
class IndustrialCameraStream:
    """Gerencia conexões RTSP/Webcam de forma resiliente."""
    def __init__(self, source):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        self.frame = None
        self.running = False
        self.lock = threading.Lock()
        
    def start(self):
        if self.running: return
        self.running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()

    def _update_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning(f"Falha na fonte {self.source}. Tentando reconectar...")
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
        if self.cap:
            self.cap.release()

# ==============================================================================
# ENGINE DE IA E LÓGICA DE SEGURANÇA (YOLO + TRACKING)
# ==============================================================================
class SafetyAIProcessor:
    """Processador central de detecção de EPIs, Incêndio e Comportamento."""
    def __init__(self, model_path='yolov8n.pt'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Carregando YOLO em modo: {self.device}")
        self.model = YOLO(model_path)
        self.alert_history = queue.Queue(maxsize=50)
        self.heatmap_data = None
        
    def run_inference(self, frame, conf_thresh, enabled_modules):
        """
        Executa detecção, tracking e regras de segurança industrial.
        """
        # Tracking ativado com ByteTrack para consistência de ID
        results = self.model.track(
            frame, 
            persist=True, 
            conf=conf_thresh, 
            device=self.device, 
            verbose=False
        )
        
        annotated_frame = frame.copy()
        
        if results[0].boxes:
            boxes = results[0].boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]
                xyxy = box.xyxy[0].cpu().numpy()
                
                # --- Lógica 1: Detecção de Queda (Análise de Aspect Ratio) ---
                if 'person' in label and enabled_modules.get('fall'):
                    w = xyxy[2] - xyxy[0]
                    h = xyxy[3] - xyxy[1]
                    if w > h * 1.3: # Geometria horizontal sugere queda
                        self._trigger_alert("PESSOA CAÍDA / ACIDENTE", "ALTA")
                
                # --- Lógica 2: Incêndio e Fumaça ---
                if label in ['fire', 'smoke'] and enabled_modules.get('fire'):
                    self._trigger_alert("DETECÇÃO DE INCÊNDIO", "CRÍTICA")

                # --- Lógica 3: EPIs (Classes Customizadas) ---
                # Nota: Em produção, substitua pelas classes do seu modelo treinado
                if label in ['no-helmet', 'no-mask'] and enabled_modules.get('epi'):
                    self._trigger_alert(f"FALTA DE EPI: {label}", "MÉDIA")

            # Gerar visualização oficial do YOLO (Bounding Boxes + Labels)
            annotated_frame = results[0].plot()
            self._update_heat_map(frame.shape, boxes)

        return annotated_frame

    def _update_heat_map(self, shape, boxes):
        if self.heatmap_data is None:
            self.heatmap_data = np.zeros((shape[0], shape[1]), dtype=np.float32)
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            self.heatmap_data[y1:y2, x1:x2] += 1
        self.heatmap_data *= 0.95 # Decay para movimento

    def _trigger_alert(self, message, severity):
        ts = datetime.now().strftime("%H:%M:%S")
        alert = {"time": ts, "msg": message, "sev": severity}
        if self.alert_history.full():
            self.alert_history.get()
        self.alert_history.put(alert)

# ==============================================================================
# INTERFACE STREAMLIT (DASHBOARD INDUSTRIAL)
# ==============================================================================
def main_app():
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2092/2092130.png", width=80)
    st.sidebar.title("VisionGuard Pro v2.6")
    st.sidebar.markdown("---")
    
    # 1. Configuração de Input
    st.sidebar.subheader("🔌 Conectividade")
    src_type = st.sidebar.selectbox("Tipo de Câmera", ["Webcam", "RTSP Stream", "Arquivo de Vídeo"])
    src_path = 0
    if src_type == "RTSP Stream":
        src_path = st.sidebar.text_input("URL RTSP", "rtsp://admin:12345@192.168.1.50:554")
    elif src_type == "Arquivo de Vídeo":
        src_path = "sample_factory.mp4" # Path local ou URL

    # 2. Parâmetros de IA
    st.sidebar.subheader("🧠 Inteligência Artificial")
    conf_val = st.sidebar.slider("Confiança Mínima", 0.1, 1.0, 0.45)
    
    st.sidebar.subheader("🛡️ Módulos de Segurança")
    mod_epi = st.sidebar.toggle("Monitoramento de EPI", True)
    mod_fire = st.sidebar.toggle("Risco de Incêndio", True)
    mod_fall = st.sidebar.toggle("Detecção de Quedas", True)
    
    active_mods = {"epi": mod_epi, "fire": mod_fire, "fall": mod_fall}

    # Inicialização Persistente
    if 'engine' not in st.session_state:
        st.session_state.engine = SafetyAIProcessor()
        st.session_state.stream = None

    # Layout do Dashboard
    col_vid, col_alert = st.columns([3, 1])

    with col_vid:
        st.subheader("📡 Live Feed - Monitoramento Ativo")
        video_placeholder = st.empty()
        
        c1, c2 = st.columns(2)
        btn_start = c1.button("INICIAR MONITORAMENTO", use_container_width=True, type="primary")
        btn_stop = c2.button("PARAR SISTEMA", use_container_width=True)

        if btn_start:
            st.session_state.stream = IndustrialCameraStream(src_path)
            st.session_state.stream.start()
            
            fps_bench = st.empty()
            last_time = time.time()
            
            while True:
                img = st.session_state.stream.get_frame()
                if img is not None:
                    # Processamento
                    processed_img = st.session_state.engine.run_inference(img, conf_val, active_mods)
                    
                    # UI Updates
                    video_placeholder.image(processed_img, channels="BGR", use_container_width=True)
                    
                    # Performance
                    curr_time = time.time()
                    fps = 1 / (curr_time - last_time)
                    last_time = curr_time
                    fps_bench.caption(f"Hardware: GPU | Performance: {fps:.1f} FPS")

                if btn_stop:
                    st.session_state.stream.stop()
                    st.rerun()

    with col_alert:
        st.subheader("🚨 Alertas de Risco")
        alert_container = st.container(height=400)
        
        # Monitor de Alertas em Tempo Real
        with alert_container:
            temp_list = list(st.session_state.engine.alert_history.queue)
            for a in reversed(temp_list):
                if a['sev'] == "CRÍTICA":
                    st.error(f"**{a['time']}** - {a['msg']}")
                elif a['sev'] == "ALTA":
                    st.warning(f"**{a['time']}** - {a['msg']}")
                else:
                    st.info(f"**{a['time']}** - {a['msg']}")

        st.markdown("---")
        st.subheader("🌡️ Heatmap de Ocupação")
        if st.session_state.engine.heatmap_data is not None:
            hm = st.session_state.engine.heatmap_data
            hm_norm = cv2.normalize(hm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            hm_color = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)
            st.image(hm_color, use_container_width=True)

# ==============================================================================
# GUIA TÉCNICO DE IMPLEMENTAÇÃO INDUSTRIAL
# ==============================================================================
"""
### 🛠️ ESTRATÉGIA DE TREINAMENTO E DEPLOY (SENIOR ADVICE)

1. **Dataset Customizado**:
   Para Laboratórios e Farmácias, o YOLOv8n base não detecta 'Jaleco' ou 'Luvas Nitrílicas'. 
   Você deve coletar ~2000 imagens e utilizar o Roboflow para rotular as classes:
   `[jaleco, luva_azul, mascara_n95, oculos_protecao, queda_detectada]`.

2. **Comando de Treino**:
   ```bash
   yolo task=detect mode=train model=yolov8s.pt data=industrial_safety.yaml epochs=100 imgsz=640 device=0
