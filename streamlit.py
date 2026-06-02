import streamlit as st
import cv2
import threading
import queue
import time
import requests
import asyncio
import tempfile
import os
import subprocess
import sys
import numpy as np
from ultralytics import YOLO
from PIL import Image
from datetime import datetime

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vision AI",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;400;500;600;700&family=Share+Tech+Mono&family=Exo+2:wght@200;300;400;600&display=swap');

:root {
    --bg-primary:    #080c10;
    --bg-secondary:  #0d1117;
    --bg-card:       #0f1923;
    --bg-card2:      #121d2a;
    --accent-cyan:   #00d4ff;
    --accent-green:  #00ff88;
    --accent-red:    #ff3a3a;
    --accent-orange: #ff8800;
    --text-primary:  #e8f4f8;
    --text-secondary:#7a9bb5;
    --text-dim:      #3a5068;
    --border:        #1a2d3f;
}
html, body, .stApp {
    background-color: var(--bg-primary) !important;
    font-family: 'Exo 2', sans-serif;
    color: var(--text-primary);
}
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg-secondary); }
::-webkit-scrollbar-thumb { background: var(--accent-cyan); border-radius: 2px; }
[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.2rem 2rem !important; max-width: 100% !important; }
.stButton > button {
    background: transparent !important;
    border: 1px solid var(--accent-cyan) !important;
    color: var(--accent-cyan) !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    padding: 0.55rem 1.6rem !important;
    border-radius: 2px !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: var(--accent-cyan) !important;
    color: var(--bg-primary) !important;
    box-shadow: 0 0 18px rgba(0,212,255,0.5) !important;
}
.stSelectbox > div > div,
.stSlider > div {
    background: var(--bg-card) !important;
    border-color: var(--border) !important;
    color: var(--text-primary) !important;
}
.stRadio label { color: var(--text-primary) !important; font-family: 'Exo 2', sans-serif !important; }
[data-testid="metric-container"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    padding: 0.8rem 1rem !important;
}
[data-testid="metric-container"] label {
    color: var(--text-secondary) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 1.5px !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--accent-cyan) !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
}
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
}
.streamlit-expanderContent {
    background: var(--bg-card2) !important;
    border: 1px solid var(--border) !important;
}
.stNumberInput input, .stTextInput input {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.8rem !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# HTML COMPONENTS
# ─────────────────────────────────────────────────────────────
def header_html():
    return """
    <div style="display:flex;align-items:center;justify-content:space-between;
                border-bottom:1px solid #1a2d3f;padding-bottom:1rem;margin-bottom:1.5rem;">
      <div style="display:flex;align-items:center;gap:16px;">
        <div style="width:42px;height:42px;border:2px solid #00d4ff;border-radius:50%;
                    display:flex;align-items:center;justify-content:center;
                    box-shadow:0 0 18px rgba(0,212,255,0.4);">
          <span style="font-size:18px;">👁</span>
        </div>
        <div>
          <div style="font-family:'Rajdhani',sans-serif;font-size:1.6rem;font-weight:700;
                      color:#e8f4f8;letter-spacing:3px;line-height:1;">VISION AI</div>
          <div style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;
                      color:#00d4ff;letter-spacing:2px;">INTELLIGENT SURVEILLANCE SYSTEM</div>
        </div>
      </div>
      <div id="clock" style="font-family:'Share Tech Mono',monospace;font-size:0.85rem;
                              color:#7a9bb5;letter-spacing:2px;"></div>
    </div>
    <script>
      function tick(){
        var n=new Date();
        var el=document.getElementById('clock');
        if(el) el.textContent=n.toLocaleTimeString('en-US',{hour12:false});
      }
      tick(); setInterval(tick,1000);
    </script>
    """

def mode_badge(mode):
    cfg = {
        "NAV":  ("#00ff88", "#00ff8822", "NAVIGATION MODE"),
        "SEC":  ("#ff3a3a", "#ff3a3a22", "SECURITY MODE"),
        "IDLE": ("#7a9bb5", "#7a9bb522", "STANDBY"),
    }
    color, bg, label = cfg.get(mode, cfg["IDLE"])
    return f"""
    <div style="display:inline-flex;align-items:center;gap:8px;
                background:{bg};border:1px solid {color};
                padding:6px 14px;border-radius:2px;">
      <div style="width:7px;height:7px;border-radius:50%;background:{color};
                  box-shadow:0 0 8px {color};animation:pulse 1.4s ease-in-out infinite;"></div>
      <span style="font-family:'Rajdhani',sans-serif;font-weight:600;
                   font-size:0.85rem;letter-spacing:2px;color:{color};">{label}</span>
    </div>
    <style>@keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:.4}}}}</style>
    """

def alert_card_html(alerts):
    if not alerts:
        return """
        <div style="background:#0f1923;border:1px solid #1a2d3f;border-radius:4px;
                    padding:1.2rem;text-align:center;">
          <div style="font-family:'Share Tech Mono',monospace;color:#3a5068;
                      font-size:0.8rem;letter-spacing:1px;">NO ACTIVE ALERTS</div>
        </div>"""
    items = ""
    for a in alerts[-8:]:
        clr  = {"critical":"#ff3a3a","warning":"#ff8800","info":"#00d4ff"}.get(a['urgency'],"#7a9bb5")
        icon = {"critical":"⚠","warning":"◈","info":"◉"}.get(a['urgency'],"·")
        ts   = a.get('time','')
        items += f"""
        <div style="display:flex;align-items:center;gap:10px;padding:7px 10px;
                    border-left:3px solid {clr};margin-bottom:5px;
                    background:rgba(0,0,0,0.25);border-radius:0 3px 3px 0;">
          <span style="color:{clr};font-size:0.9rem;">{icon}</span>
          <div style="flex:1;">
            <div style="font-family:'Rajdhani',sans-serif;font-weight:600;
                        color:{clr};font-size:0.82rem;letter-spacing:1px;">{a['message']}</div>
          </div>
          <div style="font-family:'Share Tech Mono',monospace;color:#3a5068;font-size:0.65rem;">{ts}</div>
        </div>"""
    return f'<div style="display:flex;flex-direction:column;">{items}</div>'

def stat_row_html(label, value, color="#00d4ff"):
    return f"""
    <div style="display:flex;justify-content:space-between;align-items:center;
                padding:6px 0;border-bottom:1px solid #1a2d3f;">
      <span style="font-family:'Share Tech Mono',monospace;font-size:0.7rem;
                   color:#7a9bb5;letter-spacing:1px;">{label}</span>
      <span style="font-family:'Rajdhani',sans-serif;font-weight:700;
                   font-size:1rem;color:{color};">{value}</span>
    </div>"""

def section_title(text):
    return f"""
    <div style="font-family:'Rajdhani',sans-serif;font-weight:600;font-size:0.75rem;
                letter-spacing:3px;color:#7a9bb5;text-transform:uppercase;
                margin:1rem 0 0.5rem;padding-bottom:4px;border-bottom:1px solid #1a2d3f;">
      {text}
    </div>"""

def telegram_log_html(logs):
    if not logs:
        return ('<div style="font-family:\'Share Tech Mono\',monospace;color:#3a5068;'
                'font-size:0.72rem;padding:8px;">No messages sent yet.</div>')
    items = ""
    for lg in logs[-6:]:
        items += f"""
        <div style="margin-bottom:8px;padding:7px 10px;background:#0a1520;
                    border-radius:3px;border:1px solid #1a2d3f;">
          <div style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:#3a5068;">{lg['time']}</div>
          <div style="font-family:'Exo 2',sans-serif;font-size:0.78rem;color:#7a9bb5;margin-top:2px;">
            {lg['text'][:80]}{'…' if len(lg['text'])>80 else ''}
          </div>
        </div>"""
    return items

# ─────────────────────────────────────────────────────────────
# TTS ENGINE — Fixed version
#
# ROOT CAUSE of original silence:
#   1. pyttsx3 called from a daemon thread inside Streamlit's
#      server process has no GUI event loop → init() hangs or
#      runAndWait() returns immediately without audio.
#   2. edge_tts + pygame.mixer also fails in a headless server
#      context (no audio device accessible from threads).
#
# FIX: Speak via a fresh subprocess each time.
#   • On Windows → PowerShell Add-Type SpeechSynthesizer
#     (built-in, zero deps, plays through the real user session)
#   • On Linux/macOS → espeak / espeak-ng (install once:
#     sudo apt install espeak-ng  OR  brew install espeak)
#   • Optional better voice: pyttsx3 subprocess helper script
#
# The subprocess runs in the USER's session, so Windows audio
# routing works. Critical alerts kill any running subprocess
# before starting the new one.
# ─────────────────────────────────────────────────────────────

_tts_lock      = threading.Lock()
_current_proc  = [None]   # list so closure can mutate it

def _detect_tts_backend():
    """Return 'powershell', 'espeak', 'pyttsx3_sub', or 'none'."""
    if sys.platform == "win32":
        return "powershell"
    # Linux / macOS — try espeak-ng, then espeak
    for cmd in ("espeak-ng", "espeak"):
        try:
            subprocess.run([cmd, "--version"],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL,
                           timeout=2)
            return cmd          # returns "espeak-ng" or "espeak"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    # Last resort: pyttsx3 in a subprocess (avoids the thread issue)
    try:
        import pyttsx3          # noqa: just checking it exists
        return "pyttsx3_sub"
    except ImportError:
        pass
    return "none"

_TTS_BACKEND = _detect_tts_backend()

# Small helper script written to a temp file for pyttsx3_sub mode
_PYTTSX3_HELPER = None
if _TTS_BACKEND == "pyttsx3_sub":
    _fd, _PYTTSX3_HELPER = tempfile.mkstemp(suffix=".py")
    os.close(_fd)
    with open(_PYTTSX3_HELPER, "w") as _f:
        _f.write("""
import sys, pyttsx3
engine = pyttsx3.init()
engine.setProperty('rate', 165)
engine.setProperty('volume', 1.0)
voices = engine.getProperty('voices')
for v in voices:
    if any(k in v.name.lower() for k in ('english','david','zira','hazel')):
        engine.setProperty('voice', v.id)
        break
engine.say(sys.argv[1])
engine.runAndWait()
""")

def _speak_subprocess(text: str, interrupt: bool = False):
    """
    Fire-and-forget: launch speech in a subprocess so audio
    runs in the real user session, not Streamlit's server thread.
    """
    if _TTS_BACKEND == "none":
        return

    def _run():
        with _tts_lock:
            # Kill previous if interrupt requested
            if interrupt and _current_proc[0] and _current_proc[0].poll() is None:
                try:
                    _current_proc[0].kill()
                    _current_proc[0].wait(timeout=1)
                except Exception:
                    pass

            try:
                if _TTS_BACKEND == "powershell":
                    # Escape single quotes in text
                    safe = text.replace("'", " ").replace('"', ' ')
                    ps_cmd = (
                        f"Add-Type -AssemblyName System.Speech; "
                        f"$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                        f"$s.Rate = 2; "
                        f"$s.Speak('{safe}');"
                    )
                    proc = subprocess.Popen(
                        ["powershell", "-NoProfile", "-NonInteractive",
                         "-Command", ps_cmd],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        creationflags=subprocess.CREATE_NO_WINDOW
                        if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
                    )

                elif _TTS_BACKEND in ("espeak-ng", "espeak"):
                    proc = subprocess.Popen(
                        [_TTS_BACKEND, "-s", "160", "-v", "en", text],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )

                elif _TTS_BACKEND == "pyttsx3_sub":
                    proc = subprocess.Popen(
                        [sys.executable, _PYTTSX3_HELPER, text],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                else:
                    return

                _current_proc[0] = proc
                proc.wait(timeout=15)   # max 15 s per phrase

            except Exception:
                pass

    threading.Thread(target=_run, daemon=True).start()


def speak(text: str, interrupt: bool = False):
    """Public API — drop-in replacement for original speak()."""
    _speak_subprocess(text, interrupt=interrupt)

# ─────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────
defaults = {
    "running":         False,
    "mode":            "IDLE",
    "chosen_mode":     "IDLE",
    "frame_count":     0,
    "alert_log":       [],
    "telegram_log":    [],
    "people_count":    0,
    "intruder_count":  0,
    "fps":             0.0,
    "conf_threshold":  0.55,
    "telegram_token":  "",
    "telegram_chat":   "",
    "crowd_threshold": 5,
    "alert_interval":  60,
    "depth_critical":  0.20,
    "depth_warning":   0.07,
    "tts_backend_info": _TTS_BACKEND,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────
# NAV CONSTANTS
# ─────────────────────────────────────────────────────────────
NAV_RELEVANT = {
    'person','bicycle','car','motorcycle','bus','truck',
    'chair','couch','dining table','bed','toilet',
    'dog','cat','door','bottle','laptop',
    'backpack','suitcase','umbrella','handbag',
    'traffic light','stop sign','bench','potted plant',
    'tv','refrigerator','sink','oven','microwave'
}
NAV_COOLDOWN     = {'critical': 3, 'warning': 6, 'info': 12}
URGENCY_COLOR_CV = {'critical': (0,0,220), 'warning': (0,165,255), 'info': (0,200,80)}

# ─────────────────────────────────────────────────────────────
# TELEGRAM
# ─────────────────────────────────────────────────────────────
def send_telegram(text, frame=None, token="", chat_id=""):
    def _send():
        try:
            base = f"https://api.telegram.org/bot{token}"
            if frame is not None:
                _, buf = cv2.imencode(".jpg", frame)
                requests.post(
                    f"{base}/sendPhoto",
                    data={"chat_id": chat_id, "caption": text},
                    files={"photo": ("img.jpg", buf.tobytes(), "image/jpeg")},
                    timeout=5
                )
            else:
                requests.post(
                    f"{base}/sendMessage",
                    data={"chat_id": chat_id, "text": text},
                    timeout=5
                )
            st.session_state.telegram_log.append({
                'time': datetime.now().strftime("%H:%M:%S"),
                'text': text,
            })
        except Exception:
            pass
    threading.Thread(target=_send, daemon=True).start()

# ─────────────────────────────────────────────────────────────
# PERSON TRACKER (Security)
# ─────────────────────────────────────────────────────────────
class PersonTracker:
    def __init__(self):
        self.tracks  = {}
        self.next_id = 0

    def update(self, detections, frame_h, move_thresh=30, close_ratio=0.6):
        assigned, results = set(), []
        for cx, cy, bh in detections:
            cx, cy, bh = float(cx), float(cy), float(bh)
            best_id, best_d = None, float('inf')
            for tid, t in self.tracks.items():
                if t['positions']:
                    px, py = t['positions'][-1]
                    d = np.hypot(cx-px, cy-py)
                    if d < best_d:
                        best_d, best_id = d, tid
            if best_id is None or best_d > 100:
                best_id = self.next_id
                self.next_id += 1
                self.tracks[best_id] = {'positions': [], 'first_seen': time.time()}
            assigned.add(best_id)
            t = self.tracks[best_id]
            t['positions'].append((cx, cy))
            if len(t['positions']) > 20:
                t['positions'].pop(0)
            move_dist = 0.0
            if len(t['positions']) >= 2:
                p1, p2    = t['positions'][-2], t['positions'][-1]
                move_dist = float(np.hypot(p2[0]-p1[0], p2[1]-p1[1]))
            is_close  = bh > (frame_h * close_ratio)
            is_moving = move_dist > move_thresh
            results.append({
                'id':             best_id,
                'classification': "INTRUDER" if (is_close or is_moving) else "Visitor",
                'is_moving':      bool(is_moving),
                'move_dist':      round(move_dist, 1),
            })
        for tid in list(self.tracks.keys()):
            if tid not in assigned:
                del self.tracks[tid]
        return results

# ─────────────────────────────────────────────────────────────
# NAV TRACKER
# ─────────────────────────────────────────────────────────────
class NavTracker:
    def __init__(self):
        self.state = {}

    def get_zone(self, cx, fw):
        if cx < fw / 3:    return "left"
        if cx > 2*fw / 3:  return "right"
        return "ahead"

    def get_urgency(self, area_ratio, approaching, dc, dw):
        if area_ratio > dc or (area_ratio > dw and approaching):
            return "critical"
        if area_ratio > dw:
            return "warning"
        return "info"

    def process(self, detections, fw, fh, dc, dw):
        now, frame_area = time.time(), fw * fh
        alerts, seen_keys = [], set()
        for name, x1, y1, x2, y2 in detections:
            if name not in NAV_RELEVANT:
                continue
            cx          = (x1 + x2) / 2
            zone        = self.get_zone(cx, fw)
            key         = (name, zone)
            seen_keys.add(key)
            obj_area    = (x2-x1) * (y2-y1)
            area_ratio  = obj_area / frame_area
            prev        = self.state.get(key, {})
            prev_area   = prev.get('last_area', area_ratio)
            approaching = area_ratio > prev_area * 1.08
            urgency     = self.get_urgency(area_ratio, approaching, dc, dw)
            last_spoken = prev.get('last_spoken', 0)
            self.state[key] = {
                'last_area':   area_ratio,
                'last_spoken': last_spoken,
                'urgency':     urgency,
                'approaching': approaching,
            }
            if now - last_spoken < NAV_COOLDOWN[urgency]:
                continue
            alerts.append({
                'key': key, 'name': name, 'zone': zone,
                'urgency': urgency, 'area_ratio': area_ratio,
                'approaching': approaching,
            })
        for key in list(self.state.keys()):
            if key not in seen_keys:
                del self.state[key]
        priority = {'critical': 0, 'warning': 1, 'info': 2}
        alerts.sort(key=lambda a: priority[a['urgency']])
        return alerts

    def mark_spoken(self, key):
        if key in self.state:
            self.state[key]['last_spoken'] = time.time()

def build_nav_message(alert):
    name, zone = alert['name'], alert['zone']
    urgency, approaching = alert['urgency'], alert['approaching']
    loc = "ahead" if zone == "ahead" else f"on your {zone}"
    if urgency == 'critical':
        prefix = "STOP! " if zone == "ahead" else "CAUTION! "
        suffix = " moving toward you!" if approaching else " very close!"
        return f"{prefix}{name} {loc}{suffix}"
    elif urgency == 'warning':
        suffix = " and approaching" if approaching else ""
        return f"{name} {loc}{suffix}"
    else:
        return f"{name} {loc}"

# ─────────────────────────────────────────────────────────────
# FRAME OVERLAY HELPERS
# ─────────────────────────────────────────────────────────────
def draw_scanline_overlay(frame):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    for y in range(0, h, 4):
        cv2.line(overlay, (0,y), (w,y), (0,0,0), 1)
    return cv2.addWeighted(frame, 0.88, overlay, 0.12, 0)

def draw_corner_brackets(frame, color=(0,212,255), size=28, thick=2):
    h, w = frame.shape[:2]
    for (x, y), (dx, dy) in zip(
        [(0,0),(w-1,0),(0,h-1),(w-1,h-1)],
        [(1,1),(-1,1),(1,-1),(-1,-1)]
    ):
        cv2.line(frame, (x,y), (x+dx*size, y), color, thick)
        cv2.line(frame, (x,y), (x, y+dy*size), color, thick)
    return frame

def draw_hud_bar(frame, text, bg=(10,25,38)):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0,0), (w,42), bg, -1)
    cv2.putText(frame, text, (12,28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,212,255), 1)
    return frame

def annotate_nav(frame, boxes, mdl, nav_tracker, fw, fh):
    for box in boxes:
        name = mdl.names[int(box.cls)]
        if name not in NAV_RELEVANT:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cx   = (x1+x2) / 2
        zone = nav_tracker.get_zone(cx, fw)
        key  = (name, zone)
        st_  = nav_tracker.state.get(key, {})
        urg  = st_.get('urgency', 'info')
        col  = URGENCY_COLOR_CV[urg]
        area = ((x2-x1)*(y2-y1)) / (fw*fh)
        cv2.rectangle(frame, (x1,y1), (x2,y2), col, 2)
        cv2.putText(frame, f"{name} [{urg}] {area:.2f}",
                    (x1, max(y1-8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1)
    return frame

def annotate_sec(frame, people, tracked):
    for p, t in zip(people, tracked):
        x1, y1, x2, y2 = map(int, p.xyxy[0].tolist())
        is_int = t['classification'] == "INTRUDER"
        col    = (0,0,220) if is_int else (0,200,80)
        cv2.rectangle(frame, (x1,y1), (x2,y2), col, 2)
        cv2.putText(frame, f"{t['classification']} #{t['id']}",
                    (x1, max(y1-8,12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
    return frame

# ─────────────────────────────────────────────────────────────
# LOAD MODEL (cached across reruns)
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Rajdhani',sans-serif;font-weight:700;font-size:1.1rem;
                letter-spacing:3px;color:#00d4ff;padding:0.5rem 0 1rem;">
      ⚙ SYSTEM CONFIG
    </div>""", unsafe_allow_html=True)

    # TTS status badge
    tts_color = {"powershell":"#00ff88","espeak-ng":"#00ff88",
                 "espeak":"#00ff88","pyttsx3_sub":"#ff8800","none":"#ff3a3a"}.get(_TTS_BACKEND,"#ff3a3a")
    tts_label = {"powershell":"Windows SAPI (built-in)","espeak-ng":"espeak-ng",
                 "espeak":"espeak","pyttsx3_sub":"pyttsx3 subprocess","none":"NO TTS — install espeak"}.get(_TTS_BACKEND,"none")
    st.markdown(f"""
    <div style="background:#0f1923;border:1px solid {tts_color}33;border-radius:3px;
                padding:6px 10px;margin-bottom:10px;">
      <div style="font-family:'Share Tech Mono',monospace;font-size:0.6rem;
                  color:#7a9bb5;letter-spacing:1px;">🔊 TTS ENGINE</div>
      <div style="font-family:'Rajdhani',sans-serif;font-weight:600;font-size:0.8rem;
                  color:{tts_color};margin-top:2px;">{tts_label}</div>
    </div>""", unsafe_allow_html=True)

    with st.expander("📷  CAMERA & MODEL", expanded=True):
        cam_index = st.number_input("Camera Index", 0, 10, 0, key="cam_idx")
        st.session_state.conf_threshold = st.slider(
            "Detection Confidence", 0.1, 1.0,
            float(st.session_state.conf_threshold), 0.05)

    with st.expander("🔐  TELEGRAM", expanded=False):
        st.session_state.telegram_token = st.text_input(
            "Bot Token", st.session_state.telegram_token, type="password")
        st.session_state.telegram_chat = st.text_input(
            "Chat ID", st.session_state.telegram_chat)

    with st.expander("🛡  SECURITY PARAMS", expanded=False):
        st.session_state.crowd_threshold = st.slider(
            "Crowd Threshold (people)", 2, 20, int(st.session_state.crowd_threshold))
        st.session_state.alert_interval = st.slider(
            "Alert Interval (sec)", 10, 300, int(st.session_state.alert_interval))

    with st.expander("🧭  NAVIGATION PARAMS", expanded=False):
        st.session_state.depth_critical = st.slider(
            "Critical Depth (bbox %)", 0.05, 0.50,
            float(st.session_state.depth_critical), 0.01)
        st.session_state.depth_warning = st.slider(
            "Warning Depth (bbox %)", 0.01, 0.30,
            float(st.session_state.depth_warning), 0.01)

    st.markdown("---")

    # Quick TTS test button
    if st.button("🔊 Test Voice", use_container_width=True):
        speak("Vision AI voice test. Navigation mode ready.", interrupt=True)
        st.success("Voice test triggered!")

    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;
                color:#3a5068;line-height:1.9;">
      🔊 Voice guide active in NAV mode<br>
      📡 Telegram alerts active in SEC mode
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown(header_html(), unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# MODE SELECTOR
# ─────────────────────────────────────────────────────────────
top_left, top_mid, top_right = st.columns([3, 2, 2])

with top_left:
    mode_options = ["IDLE", "NAV", "SEC"]
    current_idx  = mode_options.index(st.session_state.chosen_mode)

    chosen_mode = st.radio(
        "Select Mode",
        mode_options,
        index=current_idx,
        horizontal=True,
        key="_mode_radio",
    )
    st.session_state.chosen_mode = chosen_mode
    st.session_state.mode        = chosen_mode

with top_mid:
    if not st.session_state.running:
        if st.button("▶  START SYSTEM", use_container_width=True):
            st.session_state.running     = True
            st.session_state.frame_count = 0
            st.session_state.alert_log   = []
    else:
        if st.button("⏹  STOP SYSTEM", use_container_width=True):
            st.session_state.running     = False
            st.session_state.chosen_mode = "IDLE"
            st.session_state.mode        = "IDLE"

with top_right:
    st.markdown(mode_badge(st.session_state.chosen_mode), unsafe_allow_html=True)

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────────────────────
col_feed, col_panel = st.columns([3, 1.1])

with col_feed:
    st.markdown(section_title("LIVE FEED"), unsafe_allow_html=True)
    feed_placeholder = st.empty()
    feed_placeholder.markdown("""
    <div style="width:100%;aspect-ratio:16/9;background:#0d1117;
                border:1px solid #1a2d3f;border-radius:4px;
                display:flex;flex-direction:column;align-items:center;
                justify-content:center;gap:12px;">
      <div style="font-size:3rem;opacity:0.2;">📷</div>
      <div style="font-family:'Share Tech Mono',monospace;font-size:0.75rem;
                  color:#3a5068;letter-spacing:2px;">AWAITING FEED — PRESS START</div>
    </div>""", unsafe_allow_html=True)

    st.markdown(section_title("DETECTION LOG"), unsafe_allow_html=True)
    alert_placeholder = st.empty()
    alert_placeholder.markdown(alert_card_html([]), unsafe_allow_html=True)

with col_panel:
    st.markdown(section_title("SYSTEM STATS"), unsafe_allow_html=True)

    m1, m2 = st.columns(2)
    fps_metric    = m1.empty()
    frames_metric = m2.empty()
    fps_metric.metric("FPS", "—")
    frames_metric.metric("FRAMES", "—")

    m3, m4 = st.columns(2)
    people_metric   = m3.empty()
    intruder_metric = m4.empty()
    people_metric.metric("PEOPLE", "—")
    intruder_metric.metric("INTRUDERS", "—")

    st.markdown(section_title("OBJECT STATE"), unsafe_allow_html=True)
    obj_state_ph = st.empty()

    st.markdown(section_title("TELEGRAM LOG"), unsafe_allow_html=True)
    tg_log_ph = st.empty()
    tg_log_ph.markdown(telegram_log_html([]), unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# MAIN PROCESSING LOOP
# ─────────────────────────────────────────────────────────────
if st.session_state.running:
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Failed to load YOLO model: {e}")
        st.stop()

    cap = cv2.VideoCapture(int(st.session_state.get("cam_idx", 0)))
    if not cap.isOpened():
        st.error("Cannot open camera. Check the Camera Index in Config.")
        st.session_state.running = False
        st.stop()

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    sec_tracker     = PersonTracker()
    nav_tracker     = NavTracker()
    last_alert_time = 0
    last_crowd_time = 0
    empty_start     = time.time()
    is_stable_sent  = False
    person_start    = None
    fps_t           = time.time()

    dc = float(st.session_state.depth_critical)
    dw = float(st.session_state.depth_warning)

    # Announce mode start via voice
    if st.session_state.chosen_mode == "NAV":
        speak("Navigation mode activated. Voice guidance is ready.", interrupt=True)

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.warning("Camera feed lost.")
            break

        st.session_state.frame_count += 1
        now = time.time()

        elapsed_fps = now - fps_t
        if elapsed_fps > 0:
            st.session_state.fps = round(1.0 / elapsed_fps, 1)
        fps_t = now

        mode = st.session_state.chosen_mode

        results = model(
            frame,
            conf=float(st.session_state.conf_threshold),
            verbose=False
        )[0]

        # ══════════════════════════════════════════════════════
        # NAVIGATION MODE
        # ══════════════════════════════════════════════════════
        if mode == "NAV":
            detections = []
            for box in results.boxes:
                name = model.names[int(box.cls)]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append((name, float(x1), float(y1), float(x2), float(y2)))

            alerts = nav_tracker.process(detections, fw, fh, dc, dw)

            for alert in alerts:
                msg = build_nav_message(alert)
                # ← subprocess-based speak: works from any thread
                speak(msg, interrupt=(alert['urgency'] == 'critical'))
                nav_tracker.mark_spoken(alert['key'])
                st.session_state.alert_log.append({
                    'message': msg,
                    'urgency': alert['urgency'],
                    'time':    datetime.now().strftime("%H:%M:%S"),
                })

            frame = annotate_nav(frame, results.boxes, model, nav_tracker, fw, fh)

            obj_lines = ""
            for (nm, zn), st_ in nav_tracker.state.items():
                urg = st_.get('urgency', 'info')
                clr = {"critical":"#ff3a3a","warning":"#ff8800","info":"#00d4ff"}[urg]
                obj_lines += stat_row_html(f"{nm} / {zn}", urg.upper(), clr)
            obj_state_ph.markdown(
                obj_lines or
                '<div style="font-family:\'Share Tech Mono\',monospace;color:#3a5068;'
                'font-size:0.7rem;padding:4px;">Nothing tracked yet.</div>',
                unsafe_allow_html=True
            )
            st.session_state.people_count   = sum(1 for (n,_) in nav_tracker.state if n == 'person')
            st.session_state.intruder_count = 0

        # ══════════════════════════════════════════════════════
        # SECURITY MODE
        # ══════════════════════════════════════════════════════
        elif mode == "SEC":
            people       = [b for b in results.boxes if model.names[int(b.cls)] == 'person']
            person_count = len(people)

            detections = [(
                float((b.xyxy[0][0] + b.xyxy[0][2]) / 2),
                float((b.xyxy[0][1] + b.xyxy[0][3]) / 2),
                float( b.xyxy[0][3] - b.xyxy[0][1])
            ) for b in people]

            tracked     = sec_tracker.update(detections, fh)
            intruder_ct = sum(1 for t in tracked if t['classification'] == "INTRUDER")
            visitor_ct  = sum(1 for t in tracked if t['classification'] == "Visitor")
            any_moving  = any(t['is_moving'] for t in tracked)

            frame = annotate_sec(frame, people, tracked)
            st.session_state.people_count   = person_count
            st.session_state.intruder_count = intruder_ct

            if person_count >= st.session_state.crowd_threshold:
                cv2.rectangle(frame, (0,fh-40), (fw,fh), (180,0,0), -1)
                cv2.putText(frame, f"!! CROWD ALERT — {person_count} PEOPLE !!",
                            (10,fh-14), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
                if now - last_crowd_time > st.session_state.alert_interval:
                    last_crowd_time = now
                    msg = (f"🚨 CROWD ALERT\nPeople:{person_count} "
                           f"Intruders:{intruder_ct} Visitors:{visitor_ct}")
                    send_telegram(msg, frame,
                                  token=st.session_state.telegram_token,
                                  chat_id=st.session_state.telegram_chat)
                    st.session_state.alert_log.append({
                        'message': f"Crowd alert — {person_count} people",
                        'urgency': 'critical',
                        'time':    datetime.now().strftime("%H:%M:%S"),
                    })

            elif person_count > 0:
                empty_start    = now
                is_stable_sent = False
                if person_start is None:
                    person_start = now
                if now - last_alert_time > st.session_state.alert_interval:
                    last_alert_time = now
                    elapsed = int(now - person_start)
                    hdr = "🚨 PERSON LINGERING 10+ MIN" if elapsed > 600 else "🔔 SECURITY ALERT"
                    msg = (f"{hdr}\nPeople:{person_count} Intruders:{intruder_ct} "
                           f"Visitors:{visitor_ct}\nMovement:{'YES' if any_moving else 'NO'}\n"
                           f"Time in frame:{elapsed}s")
                    send_telegram(msg, frame,
                                  token=st.session_state.telegram_token,
                                  chat_id=st.session_state.telegram_chat)
                    st.session_state.alert_log.append({
                        'message': f"Alert: {person_count} person(s), {intruder_ct} intruder(s)",
                        'urgency': 'warning' if intruder_ct == 0 else 'critical',
                        'time':    datetime.now().strftime("%H:%M:%S"),
                    })
            else:
                person_start  = None
                sec_tracker   = PersonTracker()
                silence_dur   = now - empty_start
                if silence_dur > 10 and not is_stable_sent:
                    is_stable_sent = True
                    send_telegram("✅ All Clear — No people detected.",
                                  token=st.session_state.telegram_token,
                                  chat_id=st.session_state.telegram_chat)
                    st.session_state.alert_log.append({
                        'message': "All clear — room empty",
                        'urgency': 'info',
                        'time':    datetime.now().strftime("%H:%M:%S"),
                    })

            obj_lines  = stat_row_html("PEOPLE",    str(person_count), "#00d4ff")
            obj_lines += stat_row_html("INTRUDERS", str(intruder_ct),
                                       "#ff3a3a" if intruder_ct > 0 else "#3a5068")
            obj_lines += stat_row_html("VISITORS",  str(visitor_ct),   "#00ff88")
            obj_lines += stat_row_html("MOVEMENT",  "YES" if any_moving else "NO",
                                       "#ff8800" if any_moving else "#3a5068")
            obj_state_ph.markdown(obj_lines, unsafe_allow_html=True)

        # ── HUD overlay ───────────────────────────────────────
        mode_label = {"NAV":"NAVIGATION","SEC":"SECURITY","IDLE":"STANDBY"}.get(mode, mode)
        draw_hud_bar(frame,
            f"MODE:{mode_label}  CONF:{st.session_state.conf_threshold:.2f}"
            f"  FPS:{st.session_state.fps}  FRAME:{st.session_state.frame_count}")
        draw_corner_brackets(frame)
        frame = draw_scanline_overlay(frame)

        # ── Display ───────────────────────────────────────────
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        feed_placeholder.image(Image.fromarray(rgb), use_container_width=True)

        # ── Metrics ───────────────────────────────────────────
        fps_metric.metric("FPS",          str(st.session_state.fps))
        frames_metric.metric("FRAMES",    str(st.session_state.frame_count))
        people_metric.metric("PEOPLE",    str(st.session_state.people_count))
        intruder_metric.metric("INTRUDERS", str(st.session_state.intruder_count))

        alert_placeholder.markdown(
            alert_card_html(st.session_state.alert_log[-8:]),
            unsafe_allow_html=True)
        tg_log_ph.markdown(
            telegram_log_html(st.session_state.telegram_log),
            unsafe_allow_html=True)

    # ── Cleanup ───────────────────────────────────────────────
    cap.release()
    st.session_state.running     = False
    st.session_state.chosen_mode = "IDLE"
    st.session_state.mode        = "IDLE"
    feed_placeholder.markdown("""
    <div style="width:100%;aspect-ratio:16/9;background:#0d1117;
                border:1px solid #1a2d3f;border-radius:4px;
                display:flex;align-items:center;justify-content:center;">
      <div style="font-family:'Share Tech Mono',monospace;font-size:0.75rem;
                  color:#3a5068;letter-spacing:2px;">FEED STOPPED</div>
    </div>""", unsafe_allow_html=True)