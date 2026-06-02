import cv2
import threading
import queue
import time
import requests
import asyncio
import tempfile
import os
import numpy as np
from ultralytics import YOLO
from dotenv import load_dotenv
load_dotenv()

# --- CONFIG ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")

CAMERA_INDEX       = 0
YOLO_CONF          = 0.55
STABLE_TIMEOUT     = 120

# --- SECURITY THRESHOLDS ---
MOVEMENT_THRESHOLD = 30    # pixels between frames — above = moving
CLOSE_HEIGHT_RATIO = 0.6   # bbox height > 60% of frame height = close/intruder
CROWD_THRESHOLD    = 5     # 5+ people triggers crowd alert
ALERT_INTERVAL     = 60    # seconds between repeated alerts

# --- NAVIGATION CONFIG ---
NAV_RELEVANT = {
    'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck',
    'chair', 'couch', 'dining table', 'bed', 'toilet',
    'dog', 'cat', 'door', 'bottle', 'laptop',
    'backpack', 'suitcase', 'umbrella', 'handbag',
    'traffic light', 'stop sign', 'bench', 'potted plant',
    'tv', 'refrigerator', 'sink', 'oven', 'microwave'
}

# Cooldown in seconds before re-announcing same object+zone
NAV_COOLDOWN = {
    'critical': 3,
    'warning':  6,
    'info':    12,
}

# Fraction of total frame area thresholds
DEPTH_CRITICAL = 0.20   # > 20% of frame = very close
DEPTH_WARNING  = 0.07   # 7–20% = nearby
# < 7% = far (info)

print("Initializing Intelligent Vision Engine...")
model = YOLO("yolov8n.pt")

# =============================================================
# AUDIO ENGINE (Navigation mode only)
# =============================================================
_tts_q = queue.Queue()

def _tts_worker():
    import edge_tts
    import pygame
    pygame.mixer.init()
    while True:
        text = _tts_q.get()
        if text is None:
            break
        if text == "STOP":
            pygame.mixer.music.stop()
            continue
        try:
            fd, tmp_path = tempfile.mkstemp(suffix=".mp3")
            os.close(fd)
            asyncio.run(edge_tts.Communicate(text, "en-US-GuyNeural").save(tmp_path))
            pygame.mixer.music.load(tmp_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            pygame.mixer.music.unload()
            os.remove(tmp_path)
        except Exception:
            pass
        _tts_q.task_done()

threading.Thread(target=_tts_worker, daemon=True).start()

def speak(text, interrupt=False):
    if interrupt:
        while not _tts_q.empty():
            try:
                _tts_q.get_nowait()
            except Exception:
                break
        _tts_q.put("STOP")
    _tts_q.put(text)

# =============================================================
# ALARM SOUND (Security crowd alert only)
# =============================================================
def play_alarm():
    def _beep():
        try:
            import winsound
            for _ in range(6):
                winsound.Beep(1000, 400)
                time.sleep(0.2)
        except Exception:
            try:
                import pygame
                pygame.mixer.init()
                sample_rate = 44100
                t   = np.linspace(0, 0.4, int(sample_rate * 0.4))
                wav = (np.sin(2 * np.pi * 1000 * t) * 32767).astype(np.int16)
                wav = np.column_stack([wav, wav])
                pygame.sndarray.make_sound(wav).play()
                time.sleep(2.5)
            except Exception:
                pass
    threading.Thread(target=_beep, daemon=True).start()

# =============================================================
# TELEGRAM
# =============================================================
def send_msg(text, frame=None):
    def _send():
        try:
            base = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
            if frame is not None:
                _, buf = cv2.imencode(".jpg", frame)
                requests.post(
                    f"{base}/sendPhoto",
                    data={"chat_id": TELEGRAM_CHAT_ID, "caption": text},
                    files={"photo": ("img.jpg", buf.tobytes(), "image/jpeg")},
                    timeout=5
                )
            else:
                requests.post(
                    f"{base}/sendMessage",
                    data={"chat_id": TELEGRAM_CHAT_ID, "text": text},
                    timeout=5
                )
        except Exception:
            pass
    threading.Thread(target=_send, daemon=True).start()

# =============================================================
# SECURITY — PERSON TRACKER
# =============================================================
class PersonTracker:
    """
    Tracks individuals across frames by centroid proximity.
    Classifies each as INTRUDER or Visitor based on movement + distance.
    """
    def __init__(self):
        self.tracks  = {}
        self.next_id = 0

    def update(self, detections, frame_h):
        """
        detections : list of (cx, cy, bbox_height)
        Returns    : list of result dicts per person
        """
        assigned = set()
        results  = []

        for cx, cy, bh in detections:
            best_id, best_d = None, float('inf')
            for tid, t in self.tracks.items():
                if t['positions']:
                    px, py = t['positions'][-1]
                    d = np.hypot(cx - px, cy - py)
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

            if len(t['positions']) >= 2:
                p1, p2    = t['positions'][-2], t['positions'][-1]
                move_dist = np.hypot(p2[0]-p1[0], p2[1]-p1[1])
            else:
                move_dist = 0.0

            is_close  = bh > (frame_h * CLOSE_HEIGHT_RATIO)
            is_moving = move_dist > MOVEMENT_THRESHOLD
            classification = "INTRUDER" if (is_close or is_moving) else "Visitor"

            results.append({
                'id':             best_id,
                'classification': classification,
                'is_moving':      is_moving,
                'move_dist':      round(move_dist, 1),
            })

        for tid in list(self.tracks.keys()):
            if tid not in assigned:
                del self.tracks[tid]

        return results

# =============================================================
# NAVIGATION — SMART NAV TRACKER
# =============================================================
class NavTracker:
    """
    Tracks objects across frames for:
    - Depth estimation via bounding box area
    - Approach detection via bbox growth
    - Per-object cooldown based on urgency tier
    """
    def __init__(self):
        # key: (class_name, zone) -> {last_area, last_spoken, urgency}
        self.state = {}

    def get_zone(self, cx, fw):
        if cx < fw / 3:
            return "left"
        elif cx > 2 * fw / 3:
            return "right"
        else:
            return "ahead"

    def get_urgency(self, area_ratio, approaching):
        if area_ratio > DEPTH_CRITICAL or (area_ratio > DEPTH_WARNING and approaching):
            return "critical"
        elif area_ratio > DEPTH_WARNING:
            return "warning"
        else:
            return "info"

    def process(self, detections, fw, fh):
        """
        detections : list of (class_name, x1, y1, x2, y2)
        Returns    : list of alert dicts sorted by urgency (critical first)
        """
        now        = time.time()
        frame_area = fw * fh
        alerts     = []
        seen_keys  = set()

        for name, x1, y1, x2, y2 in detections:
            if name not in NAV_RELEVANT:
                continue

            cx         = (x1 + x2) / 2
            zone       = self.get_zone(cx, fw)
            key        = (name, zone)
            seen_keys.add(key)

            obj_area   = (x2 - x1) * (y2 - y1)
            area_ratio = obj_area / frame_area

            prev       = self.state.get(key, {})
            prev_area  = prev.get('last_area', area_ratio)

            # Object is approaching if its bbox grew by more than 8%
            approaching = area_ratio > prev_area * 1.08

            urgency    = self.get_urgency(area_ratio, approaching)
            cooldown   = NAV_COOLDOWN[urgency]
            last_spoken = prev.get('last_spoken', 0)

            # Update state
            self.state[key] = {
                'last_area':    area_ratio,
                'last_spoken':  last_spoken,
                'urgency':      urgency,
                'approaching':  approaching,
            }

            if now - last_spoken < cooldown:
                continue  # Still in cooldown — skip

            alerts.append({
                'key':        key,
                'name':       name,
                'zone':       zone,
                'urgency':    urgency,
                'area_ratio': area_ratio,
                'approaching': approaching,
            })

        # Clean up stale tracks (object left frame)
        for key in list(self.state.keys()):
            if key not in seen_keys:
                del self.state[key]

        # Sort: critical first, then warning, then info
        priority = {'critical': 0, 'warning': 1, 'info': 2}
        alerts.sort(key=lambda a: priority[a['urgency']])
        return alerts

    def mark_spoken(self, key):
        if key in self.state:
            self.state[key]['last_spoken'] = time.time()


def build_nav_message(alert):
    """Builds a natural spoken phrase tailored to urgency and context."""
    name       = alert['name']
    zone       = alert['zone']
    urgency    = alert['urgency']
    approaching = alert['approaching']

    if urgency == 'critical':
        if zone == "ahead":
            prefix = "Stop! "
            suffix = " moving toward you!" if approaching else " directly ahead, very close!"
        else:
            prefix = "Caution! "
            suffix = f" approaching from your {zone}!" if approaching else f" very close on your {zone}!"
        return f"{prefix}{name}{suffix}"

    elif urgency == 'warning':
        suffix = " and approaching" if approaching else ""
        if zone == "ahead":
            return f"{name} ahead{suffix}"
        return f"{name} on your {zone}{suffix}"

    else:  # info
        if zone == "ahead":
            return f"{name} in the distance ahead"
        return f"{name} on your {zone}"


# Urgency → BGR color for cv2 drawing
URGENCY_COLOR = {
    'critical': (0,   0,   220),   # Red
    'warning':  (0,   165, 255),   # Orange
    'info':     (0,   200,  80),   # Green
}

# =============================================================
# MAIN CONTROLLER
# =============================================================
def run_intelligent_system():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    mode = "START"

    # Navigation state
    nav_tracker = NavTracker()

    # Security state
    sec_tracker       = PersonTracker()
    last_alert_time   = 0
    last_crowd_time   = 0
    empty_start_time  = time.time()
    is_stable_sent    = False
    person_start_time = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        key = cv2.waitKey(1) & 0xFF

        if key == ord('n'):
            mode = "NAV"
            nav_tracker = NavTracker()  # reset on mode switch
            speak("Navigation Mode. I will guide your path.", True)
        elif key == ord('s'):
            mode = "SEC"
            speak("Security Mode. Monitoring silently.", True)
        elif key == ord('q'):
            break

        results = model(frame, conf=YOLO_CONF, verbose=False)[0]

        # =====================================================
        # NAVIGATION MODE — Smart, depth-aware, prioritised
        # =====================================================
        if mode == "NAV":
            # Collect all detections
            detections = []
            for box in results.boxes:
                name = model.names[int(box.cls)]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append((name, x1, y1, x2, y2))

            # Get prioritised alerts this frame
            alerts = nav_tracker.process(detections, fw, fh)

            # Speak alerts — critical ones interrupt the queue
            for alert in alerts:
                msg = build_nav_message(alert)
                interrupt = (alert['urgency'] == 'critical')
                speak(msg, interrupt=interrupt)
                nav_tracker.mark_spoken(alert['key'])

            # Draw all nav-relevant bounding boxes
            for box in results.boxes:
                name = model.names[int(box.cls)]
                if name not in NAV_RELEVANT:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cx   = (x1 + x2) / 2
                zone = nav_tracker.get_zone(cx, fw)
                key_nav  = (name, zone)
                st   = nav_tracker.state.get(key_nav, {})
                urgency  = st.get('urgency', 'info')
                color    = URGENCY_COLOR[urgency]

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Show name + urgency + zone
                label = f"{name} | {urgency} | {zone}"
                cv2.putText(frame, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)

                # Show area ratio as depth indicator
                obj_area   = (x2 - x1) * (y2 - y1)
                area_ratio = obj_area / (fw * fh)
                depth_label = f"depth:{area_ratio:.2f}"
                cv2.putText(frame, depth_label, (x1, y2 + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # HUD legend for nav
            cv2.rectangle(frame, (0, 38), (380, 68), (30, 30, 30), -1)
            cv2.putText(frame, "RED=critical  ORANGE=warning  GREEN=info",
                        (6, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

        # =====================================================
        # SECURITY MODE — Telegram only, no voice
        # =====================================================
        elif mode == "SEC":
            people       = [b for b in results.boxes if model.names[int(b.cls)] == 'person']
            person_count = len(people)
            now          = time.time()

            # Build detections for tracker
            detections = []
            for p in people:
                x1, y1, x2, y2 = p.xyxy[0].tolist()
                detections.append(((x1+x2)/2, (y1+y2)/2, y2-y1))

            tracked = sec_tracker.update(detections, fh)

            if person_count > 0:
                empty_start_time = now
                is_stable_sent   = False
                if person_start_time is None:
                    person_start_time = now

                intruder_ct = sum(1 for t in tracked if t['classification'] == "INTRUDER")
                visitor_ct  = sum(1 for t in tracked if t['classification'] == "Visitor")
                any_moving  = any(t['is_moving'] for t in tracked)

                # Draw boxes
                for p, t in zip(people, tracked):
                    x1, y1, x2, y2 = map(int, p.xyxy[0].tolist())
                    is_intruder = t['classification'] == "INTRUDER"
                    color       = (0, 0, 220) if is_intruder else (0, 200, 80)
                    label       = f"{t['classification']} #{t['id']}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

                # Count banner
                cv2.rectangle(frame, (0, 40), (380, 68), (30, 30, 30), -1)
                cv2.putText(frame,
                            f"People: {person_count}  |  Intruders: {intruder_ct}  Visitors: {visitor_ct}",
                            (6, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

                # ── CROWD ALERT (5+ people) ──────────────────────
                if person_count >= CROWD_THRESHOLD:
                    cv2.rectangle(frame, (0, 70), (fw, 98), (0, 0, 200), -1)
                    cv2.putText(frame, f"!! CROWD DANGER — {person_count} PEOPLE !!",
                                (8, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    if now - last_crowd_time > ALERT_INTERVAL:
                        last_crowd_time = now
                        play_alarm()
                        send_msg(
                            f"🚨 CROWD DANGER ALERT\n"
                            f"━━━━━━━━━━━━━━━━━━\n"
                            f"Total people : {person_count}\n"
                            f"Intruders    : {intruder_ct}\n"
                            f"Visitors     : {visitor_ct}\n"
                            f"Movement     : {'YES — DANGER' if any_moving else 'None detected'}\n"
                            f"━━━━━━━━━━━━━━━━━━\n"
                            f"Multiple people detected. Immediate action required.",
                            frame
                        )

                # ── REGULAR ALERT (every 60s) ────────────────────
                elif now - last_alert_time > ALERT_INTERVAL:
                    last_alert_time = now
                    elapsed         = int(now - person_start_time)
                    movement_status = "Movement detected — DANGER" if any_moving else "No movement — Normal"
                    header = "🚨 CRITICAL: Person lingering 10+ min" if elapsed > 600 else "🔔 SECURITY ALERT"

                    send_msg(
                        f"{header}\n"
                        f"━━━━━━━━━━━━━━━━━━\n"
                        f"Total people : {person_count}\n"
                        f"Intruders    : {intruder_ct}\n"
                        f"Visitors     : {visitor_ct}\n"
                        f"Movement     : {movement_status}\n"
                        f"Time in frame: {elapsed}s\n"
                        f"━━━━━━━━━━━━━━━━━━",
                        frame
                    )

            else:
                # No people
                person_start_time = None
                sec_tracker       = PersonTracker()
                silence_duration  = now - empty_start_time

                if silence_duration > 10 and not is_stable_sent:
                    is_stable_sent = True
                    send_msg(
                        "✅ Room Status: All Clear\n"
                        "━━━━━━━━━━━━━━━━━━\n"
                        "No people detected.\n"
                        "Everything is normal and stable."
                    )

        # -- HUD --
        cv2.putText(frame, f"MODE: {mode} | N=NAV  S=SEC  Q=QUIT",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
        cv2.imshow("Vision Assistant", frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_intelligent_system()