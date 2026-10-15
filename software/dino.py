# Grounding DINO (open-vocabulary detector) wrapper: text prompt -> boxes. Runs on MPS if available.
import threading, time
import cv2, numpy as np

MODEL_ID = "IDEA-Research/grounding-dino-tiny"
COMMON = ["banana", "scissors", "wrist watch", "computer mouse", "lighter", "phone", "black cube", "cup", "bottle", "pen",
          "screwdriver", "glasses", "can", "remote", "apple", "orange", "ball", "book", "wallet", "headphones"]
QUERY = {"watch": "wrist watch", "smartwatch": "wrist watch", "mouse": "computer mouse", "mice": "computer mouse",
         "cube": "black cube", "block": "black cube", "box": "black cube", "dice": "black cube", "die": "black cube",
         "phone": "cell phone", "cellphone": "cell phone", "can": "soda can", "glasses": "eyeglasses"}
def query_for(word): return QUERY.get(word.lower().strip(), word)
_m = {"model": None, "proc": None, "dev": "cpu", "err": "", "load_s": None}
_lock = threading.Lock()

def load():
    if _m["model"] is not None: return True
    with _lock:
        if _m["model"] is not None: return True
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            t = time.time()
            _m["proc"] = AutoProcessor.from_pretrained(MODEL_ID)
            m = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID)
            dev = "mps" if torch.backends.mps.is_available() else "cpu"
            try: m = m.to(dev)
            except Exception: dev = "cpu"
            m.eval(); _m["model"], _m["dev"], _m["load_s"] = m, dev, time.time() - t
            print(f"dino: loaded on {dev} in {_m['load_s']:.1f}s")
        except Exception as e:
            _m["err"] = str(e)[:60]; print("dino load:", e); return False
    return True

def load_async(): threading.Thread(target=load, daemon=True).start()
def ready(): return _m["model"] is not None

def _near_car(box, car):
    if car is None: return False
    x, y, w, h = box; cx, cy = x + w / 2, y + h / 2
    side = car["size"] / 1.6
    if np.hypot(cx - car["center"][0], cy - car["center"][1]) < 2.2 * side: return True
    return x <= car["center"][0] <= x + w and y <= car["center"][1] <= y + h        # box contains the car

def detect(img, names, car=None, thr=0.3, text_thr=0.25):
    """img: BGR. names: list of phrases. Returns [(label, score, (x,y,w,h))] best first, car region excluded."""
    if not load(): return []
    import torch
    from PIL import Image
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    text = " . ".join(n.lower().strip() for n in names if n.strip()) + " ."
    with _lock:
        inp = _m["proc"](images=pil, text=text, return_tensors="pt").to(_m["dev"])
        with torch.no_grad(): out = _m["model"](**inp)
        r = _m["proc"].post_process_grounded_object_detection(out, inp.input_ids, threshold=thr, text_threshold=text_thr,
                                                              target_sizes=[pil.size[::-1]])[0]
    labels = r.get("text_labels", r["labels"])
    H, W = img.shape[:2]; res = []
    for s, l, b in zip(r["scores"].tolist(), labels, r["boxes"].tolist()):
        x1, y1, x2, y2 = [int(v) for v in b]
        x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(W - 1, x2), min(H - 1, y2)
        box = (x1, y1, x2 - x1, y2 - y1)
        if box[2] < 5 or box[3] < 5 or box[2] * box[3] > 0.3 * W * H or _near_car(box, car): continue
        l = " ".join(w for w in str(l).split() if not w.startswith("##")) or names[0]
        res.append((l, float(s), box))
    res.sort(key=lambda d: -d[1])
    return res
