# Object map: class-agnostic region proposals (non-table blobs) labeled by Gemma (Ollama) from crops.
import cv2, numpy as np, json, base64, urllib.request, re, difflib

def color_proposals(img, car=None, max_n=12):
    """Boxes (x,y,w,h) of things sitting on the table: pixels that don't look like the wood table."""
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    # table color model = median of wood-ish pixels
    wood = cv2.inRange(hsv, (5, 30, 90), (30, 200, 255)) > 0
    ref = np.median(lab[wood], axis=0) if wood.sum() > 1000 else np.median(lab.reshape(-1, 3), axis=0)
    d = np.linalg.norm((lab - ref) * np.array([0.6, 1.0, 1.0]), axis=2)
    m = (d > 38).astype(np.uint8) * 255
    edges = cv2.dilate(cv2.Canny(cv2.GaussianBlur(img, (5, 5), 0), 60, 160), np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m | (edges & (d > 22).astype(np.uint8) * 255), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lb, st, cen = cv2.connectedComponentsWithStats(m)
    out = []
    for j in range(1, n):
        x, y, w, h, a = st[j]
        if a < 250 or a > 0.12 * W * H: continue
        if x <= 2 or y <= 2 or x + w >= W - 2 or y + h >= H - 2: continue
        if car is not None:
            x0, y0, x1, y1 = car["bbox"]; side = car["size"] / 1.6
            cx, cy = x + w / 2, y + h / 2
            if np.hypot(cx - car["center"][0], cy - car["center"][1]) < 2.8 * side: continue
            if not (x + w < x0 or x > x1 or y + h < y0 or y > y1): continue
        out.append((int(x), int(y), int(w), int(h), int(a)))
    out.sort(key=lambda b: -b[4])
    return [b[:4] for b in out[:max_n]]

def edge_proposals(img, car=None, max_n=14):
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV); lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    wood = cv2.inRange(hsv, (5, 30, 90), (30, 200, 255)) > 0
    ref = np.median(lab[wood], axis=0) if wood.sum() > 1000 else np.median(lab.reshape(-1, 3), axis=0)
    d = np.linalg.norm((lab - ref) * np.array([0.6, 1.0, 1.0]), axis=2)
    col = (d > 34).astype(np.uint8) * 255
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    ed = cv2.Canny(g, 50, 130)
    ed = cv2.morphologyEx(ed, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    # fill closed outlines
    cs = cv2.findContours(ed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    fill = np.zeros_like(ed); cv2.drawContours(fill, cs, -1, 255, -1)
    m = cv2.morphologyEx(col | fill, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    core = cv2.erode(m, np.ones((13, 13), np.uint8))          # break thin bridges to people / frame edge
    n, lb, st, cen = cv2.connectedComponentsWithStats(core)
    out = []
    for j in range(1, n):
        x, y, w, h, a = st[j]
        if a < 80: continue
        blob = cv2.dilate((lb == j).astype(np.uint8) * 255, np.ones((15, 15), np.uint8)) & m
        x, y, w, h = cv2.boundingRect(blob); a = int(cv2.countNonZero(blob))
        if a > 0.10 * W * H or w > 0.5 * W or h > 0.6 * H: continue
        if x <= 2 or y <= 2 or x + w >= W - 2 or y + h >= H - 2: continue
        if car is not None:
            side = car["size"] / 1.6; cx, cy = x + w / 2, y + h / 2
            if np.hypot(cx - car["center"][0], cy - car["center"][1]) < 2.8 * side: continue
            x0, y0, x1, y1 = car["bbox"]
            if not (x + w < x0 or x > x1 or y + h < y0 or y > y1): continue
        out.append((x, y, w, h, a))
    out.sort(key=lambda b: -b[4])
    return [b[:4] for b in out[:max_n]]

def _merge(boxes):
    """Union boxes that overlap a lot (same object found by several methods)."""
    boxes = [list(b) for b in boxes]; changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])); iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
                inter = ix * iy; small = min(a[2] * a[3], b[2] * b[3])
                if small and inter / small > 0.5:
                    x0, y0 = min(a[0], b[0]), min(a[1], b[1]); x1, y1 = max(a[0] + a[2], b[0] + b[2]), max(a[1] + a[3], b[1] + b[3])
                    boxes[i] = [x0, y0, x1 - x0, y1 - y0]; del boxes[j]; changed = True; break
            if changed: break
    return [tuple(int(v) for v in b) for b in boxes]

def proposals(img, car=None, extra=()):
    """Class-agnostic candidates: color-vs-table blobs + edge-closed blobs + optional extra boxes."""
    H, W = img.shape[:2]; ex = []
    for (x, y, w, h) in extra:
        if w * h > 0.15 * W * H: continue
        if car is not None and np.hypot(x + w / 2 - car["center"][0], y + h / 2 - car["center"][1]) < 2.8 * car["size"] / 1.6: continue
        ex.append((x, y, w, h))
    return _merge(list(color_proposals(img, car)) + list(edge_proposals(img, car)) + ex)[:16]

def ask(images, prompt, model="gemma3:4b", fmt=None, timeout=120):
    jp = [base64.b64encode(cv2.imencode(".jpg", i, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()).decode() for i in images]
    body = {"model": model, "prompt": prompt, "images": jp, "stream": False, "options": {"temperature": 0}}
    if fmt: body["format"] = fmt
    req = urllib.request.Request("http://localhost:11434/api/generate", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())["response"]

def label_crop(img, box):
    x, y, w, h = box; H, W = img.shape[:2]; p = int(0.35 * max(w, h)) + 8
    crop = img[max(0, y - p):min(H, y + h + p), max(0, x - p):min(W, x + w + p)]
    s = 224 / max(crop.shape[:2]); crop = cv2.resize(crop, None, fx=max(1, s), fy=max(1, s))
    r = ask([crop], "This is a top-down photo crop of an object on a wooden table. What is the object in the center? "
                    "Reply with JSON {\"name\": \"<1-3 word common name>\", \"aliases\": [\"other names\"]}. "
                    "If it is just table, wood grain, a shadow or a stain, use name \"none\".", fmt="json")
    j = json.loads(r); name = str(j.get("name", "none")).lower().strip()
    return name, [str(a).lower() for a in j.get("aliases", [])][:4]

_CACHE = []   # [(box, thumb, name, aliases)] labels from earlier scans, reused when the region looks the same

def _thumb(img, box):
    x, y, w, h = box
    return cv2.resize(img[y:y + h, x:x + w], (16, 16)).astype(np.float32)

def _cached(img, box):
    t = _thumb(img, box)
    for b, tb, name, al in _CACHE:
        ix = max(0, min(b[0] + b[2], box[0] + box[2]) - max(b[0], box[0])); iy = max(0, min(b[1] + b[3], box[1] + box[3]) - max(b[1], box[1]))
        iou = ix * iy / float(b[2] * b[3] + box[2] * box[3] - ix * iy)
        if iou > 0.6 and np.abs(tb - t).mean() < 20: return name, al
    return None

def scan(img, car=None, status=None, extra=(), on_item=None, cancel=lambda: False):
    """Label every candidate region with Gemma (cached by position+appearance). on_item(item) is called as each is found."""
    items = []
    boxes = proposals(img, car, extra)
    for i, b in enumerate(boxes):
        if cancel(): break
        if status: status(f"labeling {i + 1}/{len(boxes)}...")
        hit = _cached(img, b)
        if hit: name, al = hit
        else:
            try: name, al = label_crop(img, b)
            except Exception as e: print("scan:", e); continue
            _CACHE.append((b, _thumb(img, b), name, al)); del _CACHE[:-60]
        if name in ("none", "", "table", "wood", "shadow", "stain", "wooden table", "wood grain"): continue
        it = {"name": name, "aliases": al, "box": b}; items.append(it)
        if on_item: on_item(it)
    return items

SYN = {"watch": ["clock", "wristwatch", "smartwatch"], "screwdriver": ["tool", "driver"], "phone": ["cell phone", "smartphone", "mobile"],
       "mouse": ["computer mouse"], "cup": ["mug", "glass"], "bottle": ["water bottle"], "pen": ["pencil", "marker"], "keys": ["key", "keychain"]}
def match(word, items):
    """Fuzzy-match a spoken name to a map item. Returns item or None."""
    word = word.lower().strip()
    if not word or not items: return None
    cands = {word} | set(SYN.get(word, [])) | {k for k, v in SYN.items() if word in v}
    best, score = None, 0.0
    for it in items:
        for nm in [it["name"]] + it["aliases"]:
            for c in cands:
                s = 1.0 if c == nm else 0.9 if (c in nm.split() or nm in c.split() or c in nm) else difflib.SequenceMatcher(None, c, nm).ratio()
                if s > score: best, score = it, s
    return best if score >= 0.75 else None
