import cv2
import json
import math
import os
import time
import threading
import queue
import numpy as np
from collections import deque
from urllib.parse import urlparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from ultralytics import YOLO
import pyttsx3

# ─────────────────── CONFIG ───────────────────
ESP32_STREAM_URL = "http://10.160.66.124:81/stream"

FRAME_W, FRAME_H     = 416, 416
DISPLAY_W, DISPLAY_H = 900, 600

CENTER_BLOCK_TH   = 0.18
CENTER_CAUTION_TH = 0.36
AREA_BLOCK_TH     = 0.08
AREA_CAUTION_TH   = 0.025
AREA_APPROACH_TH  = 0.005
APPROACH_FRAMES   = 3

DARK_TH   = 35
BRIGHT_TH = 220

WALK_Y_FRAC         = 0.50
SMOOTHING_WINDOW    = 3
SAMPLE_INTERVAL      = 0.35
MIN_DISPLAY_THREAT   = 0.70
MIN_BLOCK_THRESHOLD  = 0.85
CENTER_OCCLUSION_TH   = 0.10
CENTER_OCCLUSION_HIT  = 0.16
CENTER_OCCLUSION_WARN = 0.06
ZONES      = ["FAR LEFT", "LEFT", "CENTER", "RIGHT", "FAR RIGHT"]
ZONE_COUNT = len(ZONES)

# ── FULL COCO DETECTION LIBRARY (all 80 classes) ──
OBSTACLE_CLASSES = {
    # People
    0:  "person",
    # Vehicles
    1:  "bicycle",
    2:  "car",
    3:  "motorcycle",
    4:  "airplane",
    5:  "bus",
    6:  "train",
    7:  "truck",
    8:  "boat",
    # Outdoor / street
    9:  "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    # Animals
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    # Bags & personal
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    # Sports / recreational
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    # Bottles & drinkware
    39: "bottle",
    40: "wine glass",
    41: "cup",
    # Cutlery
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    # Food
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    # Furniture
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    # Electronics
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    # Appliances
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    # Misc indoor
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush",
}
# ──────────────────────────────────────────────

model = YOLO("yolov8n.pt")

# ─────────── WEB STATUS API ───────────
SERVER_PORT = 8000
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

STATUS_LOCK = threading.Lock()
LAST_FRAME_BYTES = None
STATUS_PAYLOAD = {
    "state": "Waiting for stream",
    "detail": "The detector will show the latest obstacle guidance here.",
    "closest_object": "—",
    "closest_direction": "—",
    "free_direction": "—",
    "stream_connected": False,
    "zones": [
        {"name": name, "threat": 0.0, "blocked": False}
        for name in ZONES
    ]
}


def encode_frame(frame):
    success, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
    if not success:
        return None
    return jpeg.tobytes()


class StatusHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def _send_json(self, payload, code=200):
        body = json.dumps(payload).encode("utf-8")
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError, OSError):
            return

    def _send_file(self, relative_path, content_type):
        file_path = os.path.join(PROJECT_ROOT, relative_path.lstrip("/"))
        if not os.path.isfile(file_path):
            self.send_error(404)
            return
        with open(file_path, "rb") as f:
            body = f.read()
        try:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError, OSError):
            return

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/status":
            with STATUS_LOCK:
                payload = STATUS_PAYLOAD.copy()
            self._send_json(payload)
            return
        if path == "/api/frame":
            with STATUS_LOCK:
                frame_bytes = LAST_FRAME_BYTES
            if not frame_bytes:
                self.send_error(503)
                return
            try:
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(frame_bytes)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(frame_bytes)
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError, OSError):
                return
            return
        if path in ("/", "/index.html"):
            self._send_file("index.html", "text/html; charset=utf-8")
            return
        if path == "/styles.css":
            self._send_file("styles.css", "text/css; charset=utf-8")
            return
        if path == "/script.js":
            self._send_file("script.js", "application/javascript; charset=utf-8")
            return
        self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def start_status_server():
    server = ThreadingHTTPServer(("0.0.0.0", SERVER_PORT), StatusHandler)
    print(f"[HTTP] Serving dashboard at http://localhost:{SERVER_PORT}")
    server.serve_forever()

server_thread = threading.Thread(target=start_status_server, daemon=True)
server_thread.start()

# ─────────── TTS CONFIG ───────────
ALERT_COOLDOWN = 2.5   # seconds between voice alerts

# ─────────── TTS ANNOUNCER ───────────
class Announcer(threading.Thread):
    """Dedicated TTS thread — never blocks the vision loop."""
    def __init__(self):
        super().__init__(daemon=True)
        self.q = queue.Queue(maxsize=2)

    def say(self, text: str):
        if self.q.full():
            try: self.q.get_nowait()
            except queue.Empty: pass
        self.q.put(text)

    def run(self):
        engine = pyttsx3.init()
        engine.setProperty("rate", 170)
        engine.setProperty("volume", 1.0)
        voices = engine.getProperty("voices")
        if voices:
            engine.setProperty("voice", voices[0].id)
        while True:
            text = self.q.get()
            engine.say(text)
            engine.runAndWait()

announcer = Announcer()
announcer.start()

# ─────────── STREAM READER ───────────
class StreamReader:
    def __init__(self, url):
        self.url   = url
        self.frame = None
        self.lock  = threading.Lock()
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not self._stop.is_set():
            cap = cv2.VideoCapture(self.url)
            if not cap.isOpened():
                print(f"[STREAM] Retrying {self.url} ...")
                time.sleep(2); continue
            print("[STREAM] Connected.")
            while not self._stop.is_set():
                ret, frame = cap.read()
                if not ret: print("[STREAM] Reconnecting..."); break
                with self.lock: self.frame = frame
            cap.release()

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self): self._stop.set()

# ─────────── TRACKER ───────────
class SimpleTracker:
    def __init__(self, max_lost=5, iou_thresh=0.3):
        self.tracks={};self.next_id=0
        self.max_lost=max_lost;self.iou_th=iou_thresh

    @staticmethod
    def _iou(a,b):
        ax1,ay1,ax2,ay2=a;bx1,by1,bx2,by2=b
        ix1,iy1=max(ax1,bx1),max(ay1,by1)
        ix2,iy2=min(ax2,bx2),min(ay2,by2)
        inter=max(0,ix2-ix1)*max(0,iy2-iy1)
        if not inter: return 0.0
        return inter/((ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter)

    def update(self, dets):
        matched=set()
        for (x1,y1,x2,y2,cls_id) in dets:
            best_id,best_iou=None,self.iou_th
            for tid,t in self.tracks.items():
                if tid in matched: continue
                iou=self._iou((x1,y1,x2,y2),t["box"])
                if iou>best_iou: best_iou,best_id=iou,tid
            cx=(x1+x2)/2;cy=(y1+y2)/2
            area=(x2-x1)*(y2-y1)/(FRAME_W*FRAME_H)
            if best_id is not None:
                t=self.tracks[best_id];delta=area-t["area"]
                t.update({"box":(x1,y1,x2,y2),"area":area,"center":(cx,cy),
                          "lost":0,"cls_id":cls_id,
                          "approach_count":t["approach_count"]+1 if delta>AREA_APPROACH_TH else 0})
                matched.add(best_id)
            else:
                self.tracks[self.next_id]={"box":(x1,y1,x2,y2),"area":area,
                    "center":(cx,cy),"lost":0,"cls_id":cls_id,"approach_count":0}
                matched.add(self.next_id);self.next_id+=1
        dead=[tid for tid,t in self.tracks.items()
              if tid not in matched and t["lost"]>=self.max_lost]
        for tid in dead: del self.tracks[tid]
        for tid in self.tracks:
            if tid not in matched: self.tracks[tid]["lost"]+=1
        return self.tracks

# ═══════════════════════════════════════════════════════
#   BRAIN-LIKE ZONE PERCEPTION
# ═══════════════════════════════════════════════════════

prev_zones = None

def perceive_zones(gray, prev_gray=None):
    h, w   = gray.shape
    seg_w  = w // ZONE_COUNT
    roi_h  = int(h * 0.70)
    scores = []

    for i in range(ZONE_COUNT):
        x0  = i * seg_w
        x1  = min(x0 + seg_w, w)
        roi = gray[:roi_h, x0:x1]

        # Use edge strength and motion, not low-texture uniformity.
        # Empty wall/floor areas often look "smooth" but are not hazards.
        edges = cv2.Canny(roi, 50, 150)
        edge_density = np.count_nonzero(edges) / float(edges.size)
        texture_score = float(np.clip(edge_density * 3.5, 0.0, 1.0))

        motion_score = 0.0
        if prev_gray is not None:
            prev_roi = prev_gray[:roi_h, x0:x1]
            if prev_roi.shape == roi.shape:
                diff = np.abs(roi.astype(np.int16) - prev_roi.astype(np.int16)).mean()
                motion_score = float(np.clip(diff / 28.0, 0.0, 1.0))

        # Motion matters most; texture only adds support.
        threat = float(np.clip(0.75 * motion_score + 0.25 * texture_score, 0.0, 1.0))
        scores.append(threat)

    return scores

def scores_to_blocked(scores, threshold=MIN_BLOCK_THRESHOLD):
    return [s >= threshold for s in scores]

# ─────────── HELPERS ───────────
def fix_rotation(frame):
    h, w = frame.shape[:2]
    if h > w: frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    return frame

def forward_intent(cx, cy):
    dx = abs(cx - FRAME_W/2)/(FRAME_W/2)
    dy = abs(cy - FRAME_H/2)/(FRAME_H/2)
    return math.sqrt(dx*dx + dy*dy)

def zone_of(cx):
    idx = min(int(cx/FRAME_W*ZONE_COUNT), ZONE_COUNT-1)
    return idx, ZONES[idx]

def distance_label(area):
    if area > 0.12: return "VERY CLOSE"
    if area > 0.06: return "CLOSE"
    if area > 0.025: return "APPROACHING"
    return "FAR"

def check_vision(gray):
    mean = gray.mean()
    if mean < DARK_TH:   return "DARK"
    if mean > BRIGHT_TH: return "BRIGHT"
    return None

def should_speak(state):
    return bool(state and (state.startswith("DANGER") or state.startswith("BLOCKED")))

def center_occlusion_score(gray, prev_gray):
    # Focus on central area where the user is most likely blocked.
    y0 = int(FRAME_H * 0.18)
    y1 = int(FRAME_H * 0.82)
    x0 = int(FRAME_W * 0.18)
    x1 = int(FRAME_W * 0.82)

    center = gray[y0:y1, x0:x1]
    if center.size == 0:
        return 0.0

    center = cv2.GaussianBlur(center, (9, 9), 0)
    center_mean = float(center.mean())
    center_std = float(center.std())
    edge_map = cv2.Canny(center, 40, 120)
    edge_ratio = np.count_nonzero(edge_map) / float(edge_map.size)

    # Static score: a hand/object covering the middle often makes the region
    # look smoother and more uniform, while also changing its mean brightness.
    global_mean = float(gray.mean())
    mean_shift = abs(center_mean - global_mean) / 255.0
    smoothness = float(np.clip(1.0 - edge_ratio * 2.0, 0.0, 1.0))
    uniformity = float(np.clip(center_std / 65.0, 0.0, 1.0))
    static_score = float(np.clip(mean_shift * 0.35 + smoothness * 0.45 + uniformity * 0.20, 0.0, 1.0))

    motion_score = 0.0
    if prev_gray is not None:
        prev = prev_gray[y0:y1, x0:x1]
        if prev.shape == center.shape:
            prev = cv2.GaussianBlur(prev, (9, 9), 0)
            diff = cv2.absdiff(center, prev)
            _, mask = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
            motion_score = float(np.clip(np.count_nonzero(mask) / float(mask.size), 0.0, 1.0))

    return float(np.clip(max(static_score, motion_score * 0.85), 0.0, 1.0))

def best_free_direction(zone_occ, threat_scores, zone_blocked):
    center_bonus = [0.0, -0.02, -0.04, -0.02, 0.0]
    scored = []
    for i in range(ZONE_COUNT):
        # Keep threat influence small so empty areas don't dominate.
        s = zone_occ[i] * 2.5 + threat_scores[i] * 0.15 + center_bonus[i]
        if zone_blocked[i]:
            s += 12.0
        scored.append(s)
    return ZONES[int(np.argmin(scored))]

# ─────────── DRAWING ───────────
def draw_scene(frame, threat_scores, zone_blocked, zone_occ, state, alert_color):
    h, w  = frame.shape[:2]
    seg_w = w // ZONE_COUNT

    for i in range(ZONE_COUNT):
        score = threat_scores[i]
        is_strong_block = zone_blocked[i] or (zone_occ[i] > AREA_BLOCK_TH)
        if is_strong_block or (score > MIN_DISPLAY_THREAT and (zone_occ[i] > 0.002 or score > 0.78)):
            overlay = frame.copy()
            x0 = i * seg_w
            r = int(255 * min(score * 1.5, 1.0))
            g = int(255 * max(1.0 - score, 0.0))
            cv2.rectangle(overlay, (x0,0), (x0+seg_w,h), (0,g,r), -1)
            alpha = min(score * 0.5, 0.35)
            cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)

    for i in range(1, ZONE_COUNT):
        x = int(w * i / ZONE_COUNT)
        cv2.line(frame, (x,0), (x,h), (80,80,80), 1, cv2.LINE_AA)
    for frac in [0.2, 0.4]:
        y = int(h * frac)
        cv2.line(frame, (0,y), (w,y), (50,50,50), 1, cv2.LINE_AA)

    wy = int(h * WALK_Y_FRAC)
    cv2.line(frame, (0,wy), (w,wy), (0,200,200), 1, cv2.LINE_AA)
    cv2.putText(frame, "-- walk zone --", (6, wy-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0,200,200), 1)

    for i, name in enumerate(ZONES):
        lc = (60,60,255) if zone_blocked[i] else (160,160,160)
        cv2.putText(frame, name, (int(w*i/ZONE_COUNT)+3, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, lc, 1)

    bar_y = h - 22
    for i in range(ZONE_COUNT):
        x0    = i * seg_w
        score = min(max(threat_scores[i], zone_occ[i]/0.15*0.5), 1.0)
        pct   = int(score * 100)
        if pct >= 80:
            c = (0, 0, 255)
        elif pct >= 60:
            c = (0, 165, 255)
        else:
            c = (0, 200, 0)
        cv2.rectangle(frame, (x0, bar_y), (x0+seg_w-2, h-2), c, -1)
        cv2.putText(frame, f"{pct}%", (x0+3, h-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (255,255,255), 1)

    if state:
        font  = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.1
        thick = 3
        (tw,th),_ = cv2.getTextSize(state, font, scale, thick)
        x = (w-tw)//2; y = th+28
        cv2.rectangle(frame, (x-16,y-th-14), (x+tw+16,y+8), (0,0,0), -1)
        cv2.putText(frame, state, (x,y), font, scale, alert_color, thick)

# ─────────── MAIN ───────────
stream  = StreamReader(ESP32_STREAM_URL)
tracker = SimpleTracker()

cv2.namedWindow("Blind Assist - Enhanced", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Blind Assist - Enhanced", DISPLAY_W, DISPLAY_H)

last_sample_time = 0
last_state       = ""
last_print_time  = 0
PRINT_COOLDOWN   = 3.0
state_history    = deque(maxlen=SMOOTHING_WINDOW)
last_display     = np.zeros((DISPLAY_H, DISPLAY_W, 3), np.uint8)
prev_gray        = None

print("[READY] System live. Press Q to quit.")

while True:
    raw = stream.read()

    if raw is None:
        with STATUS_LOCK:
            STATUS_PAYLOAD.update({
                "state": "Waiting for stream",
                "detail": "The camera feed is not available yet.",
                "stream_connected": False,
                "closest_object": "—",
                "closest_direction": "—",
                "free_direction": "—",
                "zones": [
                    {"name": name, "threat": 0.0, "blocked": False}
                    for name in ZONES
                ]
            })
        waiting = last_display.copy()
        cv2.putText(waiting, "Waiting for ESP32...",
                    (DISPLAY_W//2-150, DISPLAY_H//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,200,200), 2)
        cv2.imshow("Blind Assist - Enhanced", waiting)
        if cv2.waitKey(30) & 0xFF == ord("q"): break
        continue

    raw = fix_rotation(raw)

    disp     = cv2.resize(raw, (DISPLAY_W, DISPLAY_H))
    disp_g   = cv2.cvtColor(disp, cv2.COLOR_BGR2GRAY)
    pg_disp  = cv2.resize(prev_gray, (DISPLAY_W, DISPLAY_H)) if prev_gray is not None else None
    t_scores_disp = perceive_zones(disp_g, pg_disp)
    zb_disp       = scores_to_blocked(t_scores_disp)

    now = time.time()
    if now - last_sample_time < SAMPLE_INTERVAL:
        draw_scene(disp, t_scores_disp, zb_disp, [0.0]*ZONE_COUNT,
                   last_state,
                   (0,0,255) if "DANGER" in last_state or "BLOCKED" in last_state
                   else (0,255,255) if last_state else (200,200,200))
        cv2.imshow("Blind Assist - Enhanced", disp)
        if cv2.waitKey(1) & 0xFF == ord("q"): break
        continue

    last_sample_time = now

    frame = cv2.resize(raw, (FRAME_W, FRAME_H))
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    vision_issue  = check_vision(gray)
    threat_scores = perceive_zones(gray, prev_gray)
    zone_blocked  = scores_to_blocked(threat_scores)
    any_blocked   = any(zone_blocked)
    center_cover  = center_occlusion_score(gray, prev_gray)
    prev_gray     = gray.copy()

    zone_occ      = [0.0] * ZONE_COUNT
    danger        = False
    caution       = False
    closest_area  = 0.0
    closest_label = ""
    closest_dir   = ""
    tracks        = {}
    WALK_Y        = int(FRAME_H * WALK_Y_FRAC)

    # Only keep the center-occlusion fallback as a last resort for cases where
    # the detector misses a truly blocking object.
    strong_center_cover = (center_cover >= CENTER_OCCLUSION_HIT) or \
                          (vision_issue is not None and center_cover >= CENTER_OCCLUSION_WARN)

    if strong_center_cover:
        z_idx = 2
        forced_score = min(1.0, center_cover * 1.8)
        zone_occ[z_idx] = max(zone_occ[z_idx], forced_score)
        threat_scores[z_idx] = max(threat_scores[z_idx], forced_score)
        zone_blocked[z_idx] = True
        danger = True
        closest_area = max(closest_area, forced_score)
        closest_label = "OBSTRUCTION - VERY CLOSE"
        closest_dir = ZONES[z_idx]

    if not vision_issue:
        results = model(frame, imgsz=FRAME_W, conf=0.20,
                        classes=list(OBSTACLE_CLASSES.keys()), verbose=False)
        dets = []
        for box in results[0].boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            area   = (x2-x1)*(y2-y1)/(FRAME_W*FRAME_H)
            if y2 < WALK_Y and area < 0.08: continue
            dets.append((x1,y1,x2,y2,cls_id))

        tracks = tracker.update(dets)

        for tid, t in tracks.items():
            if t["lost"] > 0: continue
            x1,y1,x2,y2 = map(int, t["box"])
            cx,cy  = t["center"]
            area   = t["area"]
            cls_id = t["cls_id"]
            approaching = t["approach_count"] >= APPROACH_FRAMES

            vert_w = max(0.0,(y2-WALK_Y)/(FRAME_H-WALK_Y))
            z_idx,z_name = zone_of(cx)
            zone_occ[z_idx] += area * vert_w

            if area > closest_area:
                closest_area  = area
                closest_label = distance_label(area)
                closest_dir   = z_name
                if cls_id in OBSTACLE_CLASSES:
                    closest_label = f"{OBSTACLE_CLASSES[cls_id].upper()} - {closest_label}"

            fi = forward_intent(cx,cy)
            # Collision logic:
            # - Yellow only when the object is in the path and steadily approaching.
            # - Red only when it is close enough and large enough to block the path.
            if fi < CENTER_BLOCK_TH and area > AREA_BLOCK_TH and approaching:
                danger = True
            elif fi < CENTER_CAUTION_TH and area > AREA_CAUTION_TH and approaching and area > 0.03:
                caution = True

            label     = OBSTACLE_CLASSES.get(cls_id,"obj")
            dist      = distance_label(area)
            box_color = (0,0,255) if area>AREA_BLOCK_TH else \
                        (0,165,255) if area>AREA_CAUTION_TH else (0,220,0)
            cv2.rectangle(frame,(x1,y1),(x2,y2),box_color,1)
            tag = f"{label} | {dist} | {ZONES[zone_of(cx)[0]]}"
            (tw,th),_=cv2.getTextSize(tag,cv2.FONT_HERSHEY_SIMPLEX,0.38,1)
            ty=max(y1-4,th+2)
            cv2.rectangle(frame,(x1,ty-th-2),(x1+tw+4,ty+2),(0,0,0),-1)
            cv2.putText(frame,tag,(x1+2,ty),cv2.FONT_HERSHEY_SIMPLEX,0.38,box_color,1)

    for i,occ in enumerate(zone_occ):
        if occ > AREA_BLOCK_TH:
            zone_blocked[i] = True
        elif occ < AREA_CAUTION_TH:
            zone_blocked[i] = False

    free_dir = best_free_direction(zone_occ, threat_scores, zone_blocked)

    if danger:
        raw_state = f"DANGER - {closest_label} - MOVE {free_dir}"
        color = (0,0,255)
    elif any_blocked:
        bnames = [ZONES[i] for i in range(ZONE_COUNT) if zone_blocked[i]]
        raw_state = f"BLOCKED - {' & '.join(bnames)} - MOVE {free_dir}"
        color = (0,0,255)
    elif caution:
        raw_state = f"CAUTION - {closest_dir} - MOVE {free_dir}"
        color = (0,255,255)
    elif vision_issue:
        raw_state = f"CAUTION - VISION {vision_issue}"
        color = (0,165,255)
    else:
        raw_state = ""
        color = (200,200,200)

    state_history.append(raw_state)
    counts = {}
    for s in state_history: counts[s]=counts.get(s,0)+1
    dominant = max(counts,key=counts.get)
    smooth_state = dominant if counts[dominant]>=max(1,SMOOTHING_WINDOW//2+1) else last_state

    now_t = time.time()
    if smooth_state != last_state:
        if smooth_state:
            print(f"[{time.strftime('%H:%M:%S')}] {smooth_state}")
            if should_speak(smooth_state):
                announcer.say(smooth_state.replace(" - ", ". ").replace("DANGER", "Danger").replace("CAUTION", "Caution").replace("BLOCKED", "Blocked").replace("MOVE", "Move"))
        last_state      = smooth_state
        last_print_time = now_t
    elif smooth_state and should_speak(smooth_state) and (now_t - last_print_time) > max(PRINT_COOLDOWN, ALERT_COOLDOWN):
        print(f"[{time.strftime('%H:%M:%S')}] {smooth_state}")
        announcer.say(smooth_state.replace(" - ", ". ").replace("DANGER", "Danger").replace("CAUTION", "Caution").replace("BLOCKED", "Blocked").replace("MOVE", "Move"))
        last_print_time = now_t

    display = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
    draw_scene(display, threat_scores, zone_blocked, zone_occ, smooth_state, color)
    last_display = display

    frame_bytes = encode_frame(display)
    with STATUS_LOCK:
        LAST_FRAME_BYTES = frame_bytes
        STATUS_PAYLOAD.update({
            "state": smooth_state or "Monitoring",
            "detail": smooth_state or "The detector is running normally.",
            "closest_object": closest_label or "No obstacle detected",
            "closest_direction": closest_dir or "—",
            "free_direction": free_dir or "—",
            "stream_connected": True,
            "zones": [
                {
                    "name": ZONES[i],
                    "threat": float(round(threat_scores[i], 3)),
                    "blocked": bool(zone_blocked[i])
                }
                for i in range(ZONE_COUNT)
            ]
        })

    cv2.imshow("Blind Assist - Enhanced", display)

    if cv2.waitKey(1) & 0xFF == ord("q"): break

stream.stop()
cv2.destroyAllWindows()
print("[EXIT] Done.")