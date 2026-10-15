# Voice commands for pad.py: "lighter, bulldoze", "mouse, follow", "stop".
# Offline speech recognition with faster-whisper on the Mac mic (background thread).
import re, threading, queue, time, json, base64
import numpy as np

# spoken object -> detector.  ("color", <COLOR_RANGES key>) or ("black", None).  Add more here.
OBJECTS = {
    "lighter": ("color", "red"),   "light": ("color", "red"),  "lighters": ("color", "red"), "red": ("color", "red"),
    "can": ("color", "orange"),    "cans": ("color", "orange"), "orange": ("color", "orange"), "soda": ("color", "orange"),
    "mouse": ("dark", "mouse"),    "mice": ("dark", "mouse"),   "mouses": ("dark", "mouse"),
    "cube": ("dark", "cube"),      "cubes": ("dark", "cube"),   "block": ("dark", "cube"), "blocks": ("dark", "cube"),
    "box": ("dark", "cube"),       "boxes": ("dark", "cube"),   "dice": ("dark", "cube"),  "die": ("dark", "cube"),
}
ACTIONS = {
    "bulldoze": "bulldoze", "bulldozer": "bulldoze", "bulldozed": "bulldoze", "bulldozing": "bulldoze",
    "ram": "bulldoze", "charge": "bulldoze", "attack": "bulldoze", "push": "bulldoze", "smash": "bulldoze", "crash": "bulldoze",
    "face": "face", "follow": "face", "track": "face", "look": "face", "turn": "face", "aim": "face", "find": "face",
}
STOPS = {"stop", "stopped", "stopping", "halt", "freeze", "whoa", "woah", "abort", "enough"}
FILLER = {"the", "a", "an", "to", "at", "and", "go", "please", "car", "now", "it", "into", "on", "that", "this", "that's",
          "toward", "towards", "my", "hey", "ok", "okay", "um", "uh", "so", "then", "you", "of", "let's", "lets", "into"}

PRAISE = ["good job", "nice", "good boy", "good girl", "good robot", "well done", "awesome", "great job", "you did it", "yay",
          "nice work", "amazing", "love you", "smart", "good car", "great work", "excellent", "perfect"]
MEAN = ["bad", "bad robot", "stupid", "dumb", "idiot", "useless", "hate you", "shut up", "you suck", "terrible", "loser", "trash"]
def _has(text, phrases):
    t = " " + " ".join(re.findall(r"[a-z']+", text.lower())) + " "
    return any(" " + p + " " in t for p in phrases)

def parse(text):
    """Return dict(action=stop|bulldoze|face|target|None, obj=str|None, spec=detector tuple|None)."""
    words = re.findall(r"[a-z']+", text.lower())
    if any(w in STOPS for w in words):
        return {"action": "stop", "obj": None, "spec": None}
    t = " " + " ".join(words) + " "
    if any(p in t for p in (" go home ", " go back", " corner ", " return ", " come back", " come home ", " home ")):
        return {"action": "home", "obj": None, "spec": None}
    if not any(w in ACTIONS for w in words):
        if _has(text, MEAN): return {"action": "sad", "obj": None, "spec": None}
        if _has(text, PRAISE): return {"action": "happy", "obj": None, "spec": None}
    if "scan" in words or "rescan" in words or ("look" in words and "around" in words) or "map" in words:
        return {"action": "scan", "obj": None, "spec": None}
    action = next((ACTIONS[w] for w in words if w in ACTIONS), None)
    rest = [w for w in words if w not in ACTIONS and w not in FILLER]
    obj = next((w for w in rest if w in OBJECTS), None)
    spec = OBJECTS.get(obj) if obj else None
    if obj is None and rest and action:            # unknown object: ask the VLM
        obj = " ".join(rest[-2:]); spec = ("vlm", obj)
    if action is None and spec is None: return {"action": None, "obj": None, "spec": None}
    return {"action": action or "target", "obj": obj, "spec": spec}

# ---------- microphone listener ----------
PROMPT = "Commands: lighter, bulldoze. mouse, bulldoze. cube, bulldoze. can, follow. scan. look around. stop."
class Listener:
    def __init__(self, on_text, model="base.en"):
        self.on_text, self.model_name = on_text, model
        self.status = "voice: loading..."
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            import sounddevice as sd
            from faster_whisper import WhisperModel
            model = WhisperModel(self.model_name, device="cpu", compute_type="int8")
        except Exception as e:
            self.status = f"voice off: {e}"[:60]; return
        q = queue.Queue(); SR, BLK = 16000, 480        # 30 ms blocks
        def cb(indata, frames, t, st): q.put(indata[:, 0].copy())
        try:
            stream = sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=BLK, callback=cb); stream.start()
        except Exception as e:
            self.status = f"mic error: {e}"[:60]; return
        self.status = "voice: listening"
        noise, buf, speaking, silent = 0.005, [], False, 0
        while True:
            b = q.get(); e = float(np.sqrt(np.mean(b * b)))
            thr = max(0.006, noise * 4)
            if not speaking:
                noise = 0.95 * noise + 0.05 * e
                buf = (buf + [b])[-10:]                  # keep 300 ms pre-roll
                if e > thr: speaking, silent = True, 0; self.status = "voice: hearing..."
                continue
            buf.append(b); silent = silent + 1 if e < thr else 0
            if silent > 15 or len(buf) > 230:            # 450 ms pause or ~7 s max
                audio = np.concatenate(buf); buf, speaking = [], False
                if len(audio) < SR * 0.3: self.status = "voice: listening"; continue
                try:
                    segs, _ = model.transcribe(audio, language="en", beam_size=1, initial_prompt=PROMPT,
                                               vad_filter=False, condition_on_previous_text=False)
                    text = " ".join(s.text for s in segs if s.no_speech_prob < 0.7).strip()
                except Exception as ex: text = ""; print("whisper:", ex)
                self.status = "voice: listening"
                if text and text.strip(" .") and text.strip() != PROMPT: self.on_text(text)

# ---------- unknown objects: ask local Ollama ----------
def vlm_bbox(frame, name, model="gemma3:4b", url="http://localhost:11434/api/generate"):
    """Ask the VLM where <name> is. Returns (x, y, w, h) in frame pixels or None."""
    import cv2, urllib.request
    H, W = frame.shape[:2]
    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    prompt = (f"This is an overhead photo, {W}x{H} pixels. Find the {name}. Reply ONLY with JSON "
              '{"found": true, "box": [x1, y1, x2, y2]} giving its bounding box in pixels, or {"found": false}.')
    body = json.dumps({"model": model, "prompt": prompt, "images": [base64.b64encode(jpg.tobytes()).decode()],
                       "stream": False, "format": "json", "options": {"temperature": 0}}).encode()
    r = urllib.request.urlopen(urllib.request.Request(url, body, {"Content-Type": "application/json"}), timeout=90)
    ans = json.loads(json.loads(r.read())["response"])
    box = ans.get("box") or ans.get("bbox")
    if not ans.get("found", True) or not box or len(box) != 4: return None
    x1, y1, x2, y2 = map(float, box)
    if max(x1, y1, x2, y2) <= 1.0: x1, x2, y1, y2 = x1 * W, x2 * W, y1 * H, y2 * H          # normalized 0-1
    elif max(x2, y2) > max(W, H) * 1.05 and max(x2, y2) <= 1000: x1, x2, y1, y2 = x1 * W / 1000, x2 * W / 1000, y1 * H / 1000, y2 * H / 1000
    x1, x2 = sorted((max(0, min(W - 1, x1)), max(0, min(W - 1, x2))))
    y1, y2 = sorted((max(0, min(H - 1, y1)), max(0, min(H - 1, y2))))
    if x2 - x1 < 4 or y2 - y1 < 4: return None
    return int(x1), int(y1), int(x2 - x1), int(y2 - y1)
