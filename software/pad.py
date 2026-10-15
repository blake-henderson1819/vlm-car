# Car control pad: overhead camera view + clickable D-pad. Also WASD (hold to keep driving), space = stop.
# + / - change speed.  q or Esc = quit
import glob, time, threading, random
import math
import cv2, numpy as np, serial
import json, os, vision
from vision import find_car, find_color_target, find_black_target, angle_error
import voice, scan, dino, homing
TARGET_COLORS = ['red', 'orange']
TRIMF = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'heading_trim.json')
if os.path.exists(TRIMF): vision.HEADING_TRIM[0] = json.load(open(TRIMF))['deg']

ports = glob.glob("/dev/cu.usbmodem*") + glob.glob("/dev/cu.usbserial*")
car = serial.Serial(ports[0], 115200, timeout=0) if ports else None
if car: time.sleep(2.5)
speed = 180
EDGE_FRAC = 0.08       # "afraid of the edge": keep the marker this far (fraction of frame) from every edge
def edge_guard(s):
    """If an M command would move the marker toward a nearby screen edge, replace it with a short move the other way."""
    try:
        if not s.startswith("M "): return s
        l, r = [int(v) for v in s.split()[1:3]]
        c = seen.get("car")
        if c is None or l + r == 0: return s
        x, y = c["center"]; mx, my = EDGE_FRAC * CW, EDGE_FRAC * CH
        d = 1 if l + r > 0 else -1
        vx, vy = d * math.cos(c["heading"]), d * math.sin(c["heading"])
        bad = (x < mx and vx < 0) or (x > CW - mx and vx > 0) or (y < my and vy < 0) or (y > CH - my and vy > 0)
        if not bad: return s
        seen["edge"] = time.time()
        return f"M {-200 * d} {-200 * d} 400"          # straight away from the edge
    except Exception:
        return s
TRAIL = []              # recent motor moves (l, r, ms) so "go home" can retrace them backwards when the marker is lost
def send(s):
    s = edge_guard(s)
    if s.startswith("M ") and not seen.get("retracing"):
        try:
            l, r, ms = [int(v) for v in s.split()[1:4]]
            TRAIL.append((l, r, ms)); del TRAIL[:-300]
        except Exception: pass
    if car: car.write((s + "\n").encode())
def retrace_step():
    """Undo the most recent move (same wheels, opposite direction). Returns False when there's nothing left."""
    if not TRAIL: return False
    l, r, ms = TRAIL.pop()
    seen["retracing"] = True
    try: send(f"M {-l} {-r} {ms}")
    finally: seen["retracing"] = False
    time.sleep(ms / 1000 + 0.05)
    return True
ARC_SLOW = 130          # slow wheel in an arc turn: both wheels always same direction, never below ~100 (stall/over-torque)
def arc(cw_raw, direction=1, ms=300, slow=ARC_SLOW):
    """Arc turn: fast wheel 255, slow wheel `slow`, both forward (direction=1) or both reverse (-1).
    cw_raw=+1 is the old 'right' spin sense (left wheel faster); multiply by TURN_SIGN for screen-clockwise."""
    slow = max(100, min(255, int(slow)))
    l, r = (255, slow) if cw_raw > 0 else (slow, 255)
    send(f"M {l * direction} {r * direction} {int(ms)}")
def drive(kind, ms=300):
    if kind == "up": send(f"M {speed} {speed} {ms}")
    elif kind == "down": send(f"M -255 -255 {ms}")
    else: arc(1 if kind == "right" else -1, 1, ms)                  # A/D and auto turns: forward arcs

dist = [None]
def reader():
    buf = b""
    while car:
        buf += car.read(256)
        while b"\n" in buf:
            ln, buf = buf.split(b"\n", 1); ln = ln.strip()
            if ln.startswith(b"D "):
                try: dist[0] = int(ln[2:])
                except ValueError: pass
        time.sleep(0.02)
threading.Thread(target=reader, daemon=True).start()


# ---------- face-the-target ----------
seen = {"car": None, "target": None, "click": None, "scale": 1.0}
# On this car a "right" command (M +255 -255) turns COUNTER-clockwise on screen (found in simple_car: needed a sign flip).
TURN_SIGN = -1
HOME_TURN_SIGN = 1     # go-home reverse arcs: flip if it turns the long way
facing = {"on": False, "msg": "", "sign": TURN_SIGN}
def face_target():
    # follow mode: keep turning to face the target until F or space is pressed again
    facing["on"], facing["msg"] = True, "Following target"
    last_err = None
    while facing["on"]:
        time.sleep(0.4)                                   # let the car settle and the camera catch up
        car, tgt = seen["car"], seen["click"] if seen["click"] is not None else seen["target"]
        if car is None or tgt is None:
            facing["msg"] = "Following - can't see the " + ("car" if car is None else "target"); last_err = None; continue
        err = math.degrees(angle_error(car, tgt))
        if abs(err) < 10:
            facing["msg"] = "Following - locked on"; last_err = None; continue
        facing["msg"] = f"Following - off by {err:+.0f} deg"
        if last_err is not None and abs(err) > abs(last_err) + 8 and abs(last_err) < 150:
            facing["sign"] *= -1                          # turned the wrong way: flip
        last_err = err
        ms = int(max(70, min(300, abs(err) * 3.5)))
        drive("right" if err * facing["sign"] > 0 else "left", ms)
        time.sleep(ms / 1000)
    send("S"); facing["msg"] = "Follow off"

COMMIT_L, COMMIT_DEG = 2.3, 15     # lock the target when within 2.3 car-lengths and aimed within 15 deg
def bulldoze():
    """Aim at the target and approach; once close and aimed, LOCK its position and ram straight through it
    (the car covers the target, so detection is ignored in the ram phase)."""
    facing["on"], facing["msg"] = True, "Bulldoze: aiming"
    last_err, lost, lock, last_tgt = None, 0, None, None
    for i in range(80):
        if not facing["on"]: break
        time.sleep(0.3)
        car, tgt = seen["car"], seen["click"] if seen["click"] is not None else seen["target"]
        if car is not None and tgt is not None and np.hypot(*(tgt - car["center"])) < 0.9 * car["size"]:
            tgt = None                                       # detection on the car body = the car covering it
        if car is not None and tgt is None and last_tgt is not None:
            # target vanished (usually under the car): ram its last known spot instead of giving up
            lock = np.array(last_tgt, dtype=float); break
        if car is None or tgt is None:
            lost += 1
            facing["msg"] = "Bulldoze: can't see the " + ("car" if car is None else "target")
            if lost > 10: facing["msg"] = "Bulldoze: lost it"; send("S"); facing["on"] = False; return
            continue
        lost = 0; last_tgt = np.array(tgt, dtype=float)
        err = math.degrees(angle_error(car, tgt))
        L = car["size"]; dist_c = float(np.hypot(*(tgt - car["center"])))
        if dist_c < COMMIT_L * L and abs(err) < COMMIT_DEG:
            lock = np.array(tgt, dtype=float); break
        if abs(err) > 25:                                   # way off: turn in place first
            if last_err is not None and abs(err) > abs(last_err) + 8 and abs(last_err) < 150: facing["sign"] *= -1
            last_err = err
            ms = int(max(70, min(300, abs(err) * 3.5)))
            facing["msg"] = f"Bulldoze: turning ({err:+.0f} deg)"
            drive("right" if err * facing["sign"] > 0 else "left", ms); time.sleep(ms / 1000)
            continue
        last_err = None
        steer = max(-60, min(60, err * 2.5)) * facing["sign"]  # + = steer clockwise (speed up left wheel)
        base = 210                                          # slowest wheel 150: both wheels always drive
        facing["msg"] = f"Bulldoze: approaching ({dist_c / L:.1f} car-lengths, {err:+.0f} deg)"
        send(f"M {int(min(255, base + steer))} {int(min(255, base - steer))} 350")
    if facing["on"] and lock is not None:
        # ---- ram phase: full power at the locked point until the marker is ~1 car-length past it ----
        seen["lock"] = lock; facing["msg"] = "BULLDOZE! (locked)"
        t0, lost_t = time.time(), None
        while facing["on"] and time.time() - t0 < 2.5:
            car = seen["car"]
            if car is None:
                lost_t = lost_t or time.time()
                if time.time() - lost_t > 1.0: break
                send("M 255 255 200"); time.sleep(0.1); continue
            lost_t = None
            h = np.array([math.cos(car["heading"]), math.sin(car["heading"])])
            past = float(np.dot(car["center"] - lock, h))   # >0: marker is beyond the locked point
            if past > 1.0 * car["size"]: break
            steer = 0
            if past < -0.5 * car["size"]:                   # still ahead of us: small heading correction
                steer = max(-50, min(50, math.degrees(angle_error(car, lock)) * 2.0)) * facing["sign"]
            l, r = 255 - max(0, -steer), 255 - max(0, steer)  # full power, slow one side to steer
            send(f"M {int(l)} {int(r)} 250"); time.sleep(0.1)
        send("S"); seen["lock"] = None
        if facing["on"]: facing["msg"] = "Bulldozed!"; send(f"H {random.randint(0, 9)}")
    send("S"); facing["on"] = False

# ---------- voice commands ----------
seen["spec"], seen["obj"], seen["heard"] = ("color", "red"), "red", ""
HOME_FRAC = (0.78, 0.75)                      # home = 22% in from the right, 25% up from the bottom
def home_point(): return np.array([HOME_FRAC[0] * CW, HOME_FRAC[1] * CH])
def go_home():
    """Simple go-home: point the car's REAR at home with short reverse arc turns, then back straight in.
    If the marker is off screen, retrace recent moves backwards until it's visible again."""
    facing["on"], facing["mode"], facing["msg"] = True, "home", "Going home (reversing)"
    seen["cmd_id"] = seen.get("cmd_id", 0) + 1
    H = home_point(); lost_t = None
    for i in range(250):
        if not facing["on"]: send("S"); return
        car = seen["car"]
        if car is None or car.get("aruco") is None:
            lost_t = lost_t or time.time()
            if time.time() - lost_t > 0.3:
                facing["msg"] = f"Home: can't see marker, retracing ({len(TRAIL)} moves left)"
                if not retrace_step():
                    send("S"); facing["msg"] = "Home: lost the marker"; facing["on"] = False; return
                continue
            time.sleep(0.1); continue
        lost_t = None
        to_home = H - car["center"]; dist = float(np.hypot(*to_home))
        if dist < 0.6 * car["size"]: break                                   # close enough: home
        rear = car["heading"] + math.pi
        err = math.degrees((math.atan2(to_home[1], to_home[0]) - rear + math.pi) % (2 * math.pi) - math.pi)
        if abs(err) > 20:                                                      # rear not aimed: reverse arc turn
            ms = int(max(120, min(350, abs(err) * 3)))
            facing["msg"] = f"Home: turning (rear off {err:+.0f} deg)"
            arc((1 if err > 0 else -1) * HOME_TURN_SIGN, -1, ms)
            time.sleep(ms / 1000 + 0.3)                                        # pause so the camera sees the real angle
        else:                                                                  # aimed: back straight in
            ms = int(max(150, min(450, dist / car["size"] * 250)))
            facing["msg"] = f"Home: backing in ({dist / car['size']:.1f} car-lengths)"
            send(f"M -230 -230 {ms}"); time.sleep(ms / 1000 + 0.15)
    send("S"); time.sleep(0.2); send(f"H {random.randint(0, 9)}")
    TRAIL.clear(); facing["msg"] = "Home!"; facing["on"] = False

def start_home():
    if facing["on"]: facing["on"] = False; time.sleep(0.5)
    threading.Thread(target=go_home, daemon=True).start()

def start_action(action):
    if action == "face" and facing["on"] and facing.get("mode") == "face": return
    if facing["on"]:
        facing["on"] = False; time.sleep(0.5)            # let the running action exit first
    facing["mode"] = action
    threading.Thread(target=bulldoze if action == "bulldoze" else face_target, daemon=True).start()

def set_target(obj, spec, then=None):
    seen["click"] = None; seen["obj"] = obj
    if spec[0] == "color": seen["color"] = spec[1]
    if spec[0] == "vlm":
        it = scan.match(obj, objmap["items"]) or scan.match(obj.split()[-1], objmap["items"])
        if it is not None:                                   # on the Gemma map: track it from its map box
            seen["spec"] = ("map", it["name"]); seen["obj"] = it["name"]; seen["track_init"] = it["box"]; seen["map_item"] = it
            if then: start_action(then)
            return
    if spec[0] != "vlm":
        seen["spec"] = spec
        if then: start_action(then)
        return
    seen["spec"] = ("vlm", obj); seen["tracker"] = None; seen["track_box"] = None
    def ask():
        facing["msg"] = f"Asking gemma3 where the {obj} is..."
        try: box = voice.vlm_bbox(seen["frame"].copy(), obj)
        except Exception as e: box = None; print("ollama:", e)
        if seen.get("spec") != ("vlm", obj): return     # target changed meanwhile
        if box is None: facing["msg"] = f"Couldn't find the {obj}"; return
        seen["track_init"] = box; facing["msg"] = f"Found the {obj}"
        if then: start_action(then)
    threading.Thread(target=ask, daemon=True).start()

USE_GEMMA_MAP = False         # True = old Gemma crop-labeling map (slow); False = Grounding DINO map
said_names = []               # object names the user has said this session (added to the scan prompt)
def go_command(obj, spec, act):
    """Stop, fresh frame, Grounding DINO for <obj>, track + go. Fallbacks: built-in detector, Gemma lookup."""
    seen["cmd_id"] = cid = seen.get("cmd_id", 0) + 1
    t0 = time.time(); live = lambda: seen.get("cmd_id") == cid
    facing["on"] = False; held[0] = None; send("S")
    facing["msg"] = f"looking for {obj}..."; seen["obj"] = obj
    if obj not in said_names: said_names.append(obj)
    time.sleep(0.3)                                          # car settles, fresh frame
    if not live(): return
    img, car_ = seen["frame"].copy(), seen.get("car")
    query = dino.query_for(obj)
    try: dets = dino.detect(img, [query], car_)
    except Exception as e: dets = []; print("dino:", e)
    if not live(): return
    def go(box, name, how):
        seen["click"] = None; seen["obj"] = name; seen["spec"] = ("map", name); seen["tracker"] = None; seen["track_init"] = box
        seen["map_item"] = None; seen["dino_q"] = query if how == "dino" else None; seen["dino_t"] = time.time()
        print(f"command->moving: {time.time() - t0:.1f}s ({name} via {how})")
        facing["msg"] = f"{name} via {how} in {time.time() - t0:.1f}s"
        start_action(act)
    if dets:
        lab, sc, box = dets[0]
        objmap["items"] = [it for it in objmap["items"] if it["name"] != obj] + [{"name": obj, "aliases": [], "box": box}]
        go(box, obj, "dino"); return
    if spec and spec[0] in ("color", "dark"):                    # built-in detector fallback
        seen["click"] = None; seen["spec"] = spec; seen["obj"] = obj; seen["dino_q"] = None
        if spec[0] == "color": seen["color"] = spec[1]
        print(f"command->moving: {time.time() - t0:.1f}s (built-in {obj})"); start_action(act); return
    facing["msg"] = f"asking gemma where the {obj} is..."
    try: box = voice.vlm_bbox(img, obj)
    except Exception as e: box = None; print("ollama:", e)
    if not live(): return
    if box is not None: go(box, obj, "gemma"); return
    facing["msg"] = f"couldn't find {obj}"; print("couldn't find", obj); send(f"A {random.randint(0, 9)}")

def dino_refresh(q, cid):
    """Re-detect the current target to fix tracker drift; keep the detection nearest the tracked spot."""
    try:
        img, car_ = seen["frame"].copy(), seen.get("car")
        dets = dino.detect(img, [q], car_)
        if dets and seen.get("cmd_id") == cid and seen.get("dino_q") == q:
            cur = seen.get("target")
            if cur is not None:
                dets.sort(key=lambda d: np.hypot(d[2][0] + d[2][2] / 2 - cur[0], d[2][1] + d[2][3] / 2 - cur[1]) - 200 * d[1])
            seen["track_init"] = dets[0][2]; seen["dino_hits"] = seen.get("dino_hits", 0) + 1
    except Exception as e: print("dino refresh:", e)
    seen["dino_busy"] = False

def react(mood):
    send(f"{'H' if mood == 'happy' else 'A'} {random.randint(0, 9)}")
    facing["msg"] = ":D  happy chirp!" if mood == "happy" else ":(  sad beeps..."
    seen["mood"] = (mood, time.time())

def on_voice(text):
    seen["heard"] = text; print("heard:", text)
    cmd = voice.parse(text)
    if cmd["action"] == "stop":
        seen["cmd_id"] = seen.get("cmd_id", 0) + 1
        facing["on"] = False; held[0] = None; send("S"); facing["msg"] = "Stopped (voice)"; return
    if cmd["action"] == "scan": start_scan(); return
    if cmd["action"] == "home": threading.Thread(target=start_home, daemon=True).start(); return
    if cmd["action"] in ("happy", "sad"): react(cmd["action"]); return
    if cmd["action"] is None: return
    act = cmd["action"] if cmd["action"] in ("bulldoze", "face") else None
    if cmd["spec"] and act: threading.Thread(target=go_command, args=(cmd["obj"], cmd["spec"], act), daemon=True).start()
    elif cmd["spec"]: set_target(cmd["obj"], cmd["spec"], act)
    elif act: start_action(act)

# ---------- Gemma object map (G / voice "scan") ----------
objmap = {"items": [], "status": "", "busy": False}
def do_scan():
    if objmap["busy"] or seen.get("frame") is None: return
    objmap["busy"] = True; objmap["status"] = "scanning..."
    try:
        img, car_ = seen["frame"].copy(), seen.get("car")
        if USE_GEMMA_MAP: items = scan.scan(img, car_, status=lambda m: objmap.__setitem__("status", m))
        else:
            t = time.time()
            names = dino.COMMON + [n for n in said_names if n not in dino.COMMON]
            dets = dino.detect(img, names, car_)
            items, used = [], []
            for lab, sc, box in dets:                          # drop duplicate boxes of the same object
                if any(abs(box[0] - u[0]) < 15 and abs(box[1] - u[1]) < 15 for u in used): continue
                used.append(box); items.append({"name": lab, "aliases": [], "box": box, "score": sc})
            print(f"dino scan {time.time() - t:.2f}s")
        objmap["items"] = items
        objmap["status"] = f"map: {len(items)} objects"; print("map:", [(i["name"], i["box"]) for i in items])
    except Exception as e: objmap["status"] = f"scan failed: {e}"[:40]; print("scan:", e)
    objmap["busy"] = False
def start_scan(): threading.Thread(target=do_scan, daemon=True).start()
def rescan():
    if objmap["busy"]: return
    scan._CACHE.clear(); start_scan()

CW, CH, PW = 960, 540, 300                 # camera view size, pad width
cx, cy, B = CW + PW // 2, 230, 80          # pad center, button size
BTN = {"up": (cx - B // 2, cy - B * 3 // 2), "down": (cx - B // 2, cy + B // 2),
       "left": (cx - B * 3 // 2, cy - B // 2), "right": (cx + B // 2, cy - B // 2), "stop": (cx - B // 2, cy - B // 2)}
RESCAN = (cx + B // 2, cy + B // 2, B, B)          # bottom-right cell next to the D-pad
held = [None]
def hit(x, y):
    for k, (bx, by) in BTN.items():
        if bx <= x <= bx + B and by <= y <= by + B: return k
def on_mouse(ev, x, y, flags, _):
    if ev == cv2.EVENT_LBUTTONDOWN and x < CW:
        seen["click"] = np.array([x * seen["scale"], y * seen["scale"]]); return
    if ev == cv2.EVENT_RBUTTONDOWN and x < CW:
        seen["click"] = None; return
    rx, ry, rw, rh = RESCAN
    if ev == cv2.EVENT_LBUTTONDOWN and rx <= x <= rx + rw and ry <= y <= ry + rh:
        rescan(); return
    if ev == cv2.EVENT_LBUTTONDOWN:
        k = hit(x, y)
        if k == "stop": held[0] = None; send("S")
        else: held[0] = k
    elif ev == cv2.EVENT_LBUTTONUP:
        if held[0]: send("S")
        held[0] = None

dino.load_async()
cam = cv2.VideoCapture(0)
listener = voice.Listener(on_voice)
win = "Car Control"
cv2.namedWindow(win); cv2.setMouseCallback(win, on_mouse)
last_send, active = 0, None
ARROW = {"up": "^", "down": "v", "left": "<", "right": ">", "stop": "STOP"}
while True:
    ok, f = cam.read()
    view = cv2.resize(f, (CW, CH)) if ok else np.zeros((CH, CW, 3), np.uint8)
    if ok:
        small = view; seen["scale"] = 1.0
        det = find_car(small); seen["car"] = det; seen["frame"] = small
        objmap["frames"] = objmap.get("frames", 0) + 1
        for it in objmap["items"]:                                   # persistent Gemma map labels
            x, y, w, h = it["box"]
            cv2.rectangle(view, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.putText(view, it["name"], (x, y + h + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        spec = seen["spec"]
        if (facing["on"] and seen.get("dino_q") and not seen.get("dino_busy") and dino.ready()
                and time.time() - seen.get("dino_t", 0) > 1.0):
            seen["dino_busy"] = True; seen["dino_t"] = time.time()
            threading.Thread(target=dino_refresh, args=(seen["dino_q"], seen.get("cmd_id")), daemon=True).start()
        if spec[0] == "map": spec = ("vlm",)                      # map targets use the same live tracker
        if spec[0] == "color": seen["target"] = find_color_target(small, det, spec[1])
        elif spec[0] == "dark": seen["target"] = vision.find_dark_target(small, det, spec[1])
        else:
            if seen.get("track_init") is not None:
                seen["tracker"] = cv2.TrackerCSRT_create(); seen["tracker"].init(small, tuple(seen.pop("track_init")))
            seen["target"] = None
            if seen.get("tracker") is not None:
                okt, bb = seen["tracker"].update(small)
                if okt:
                    x, y, w, h = map(int, bb); seen["target"] = np.array([x + w / 2, y + h / 2])
                    if seen["spec"][0] == "map" and seen.get("map_item"): seen["map_item"]["box"] = (x, y, w, h)
                    cv2.rectangle(view, (x, y), (x + w, y + h), (255, 0, 255), 2)
        tgt = seen["click"] if seen["click"] is not None else seen["target"]
        if det:
            # smooth the heading over frames so the arrow doesn't jump around
            hs = seen.get("h_smooth")
            if hs is None: hs = det["heading"]
            else: hs = hs + 0.35 * ((det["heading"] - hs + math.pi) % (2 * math.pi) - math.pi)
            seen["h_smooth"] = hs; det["heading"] = hs
            if det.get("aruco") is not None:
                cv2.polylines(view, [det["aruco"].astype(np.int32)], True, (255, 255, 0), 2)    # cyan = marker locked
            elif det.get("arrow") is not None:
                cv2.polylines(view, [det["arrow"]["hull"]], True, (255, 0, 255), 2)       # magenta = white arrow found
            else:
                a, b = det["sensor_line"]
                cv2.line(view, tuple(map(int, a)), tuple(map(int, b)), (0, 255, 255), 3)   # yellow = sensor face (fallback)
            P = tuple(map(int, det["front"])); L = 90
            E = (int(P[0] + L * math.cos(hs)), int(P[1] + L * math.sin(hs)))
            cv2.arrowedLine(view, P, E, (0, 255, 0), 3, tipLength=0.3)
            if facing["on"] and facing.get("mode") == "home":         # go-home debug: rear vector vs car->home
                C = det["center"]; rear = hs + math.pi; Hm = home_point()
                Rr = (int(C[0] + 110 * math.cos(rear)), int(C[1] + 110 * math.sin(rear)))
                cv2.line(view, tuple(map(int, C)), tuple(map(int, Hm)), (255, 255, 255), 1)
                cv2.arrowedLine(view, tuple(map(int, C)), Rr, (0, 165, 255), 3, tipLength=0.25)
                herr = math.degrees(homing.wrap(math.atan2(Hm[1] - C[1], Hm[0] - C[0]) - rear))
                cv2.putText(view, f"rear vs home: {herr:+.0f} deg", (int(C[0]) + 15, int(C[1]) + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        if tgt is not None:
            T = tuple(map(int, tgt)); cv2.circle(view, T, 14, (0, 0, 255), 3)
            if det: cv2.line(view, tuple(map(int, det['center'])), T, (0, 0, 255), 1)
        if seen.get("lock") is not None:
            cv2.drawMarker(view, tuple(map(int, seen["lock"])), (0, 0, 255), cv2.MARKER_CROSS, 30, 3)
        hp = tuple(map(int, home_point()))
        cv2.drawMarker(view, hp, (255, 0, 255), cv2.MARKER_TILTED_CROSS, 22, 2); cv2.circle(view, hp, 26, (255, 0, 255), 2)
        cv2.putText(view, "HOME (R)", (hp[0] - 40, hp[1] - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        if time.time() - seen.get("edge", 0) < 1.0:
            cv2.putText(view, "EDGE! backing off", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        if facing["msg"]: cv2.putText(view, facing["msg"], (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    canvas = np.full((CH, CW + PW, 3), 35, np.uint8); canvas[:, :CW] = view
    k = cv2.waitKey(1) & 0xFF
    key_dir = {ord("w"): "up", ord("s"): "down", ord("a"): "left", ord("d"): "right"}.get(k)
    if k in (ord("q"), 27): break
    if k == ord("g"): rescan()
    if k == ord("r"): threading.Thread(target=start_home, daemon=True).start()
    if not objmap.get("warmed") and objmap.get("frames", 0) > 30: objmap["warmed"] = True; start_scan()   # background: fills label cache
    if k == ord(" "): seen["cmd_id"] = seen.get("cmd_id", 0) + 1; held[0] = None; facing["on"] = False; send("S")
    if k == ord("f"): facing["mode"] = "face"
    if k == ord("b"): facing["mode"] = "bulldoze"
    if k in (ord("["), ord("]")):
        vision.HEADING_TRIM[0] += -2 if k == ord("[") else 2
        json.dump({"deg": vision.HEADING_TRIM[0]}, open(TRIMF, "w"))
    if k == ord("c"):
        cur = seen["color"] if seen["spec"][0] == "color" and seen.get("color") in TARGET_COLORS else TARGET_COLORS[-1]
        i = TARGET_COLORS.index(cur); seen["color"] = TARGET_COLORS[(i + 1) % len(TARGET_COLORS)]
        seen["spec"], seen["obj"] = ("color", seen["color"]), seen["color"]
    if k == ord("b") and not facing["on"]: threading.Thread(target=bulldoze, daemon=True).start()
    if k == ord("f"):
        if facing["on"]: facing["on"] = False
        else: threading.Thread(target=face_target, daemon=True).start()
    if k in (ord("+"), ord("=")): speed = min(255, speed + 15)
    if k == ord("-"): speed = max(90, speed - 15)
    if key_dir: drive(key_dir, 300); active, last_send = key_dir, time.time()
    if held[0] and time.time() - last_send > 0.15: drive(held[0], 300); last_send = time.time(); active = held[0]
    if not held[0] and not key_dir and time.time() - last_send > 0.3: active = None
    for name, (bx, by) in BTN.items():
        on = active == name
        cv2.rectangle(canvas, (bx, by), (bx + B, by + B), (0, 200, 0) if on else ((60, 60, 200) if name == "stop" else (90, 90, 90)), -1)
        cv2.rectangle(canvas, (bx, by), (bx + B, by + B), (220, 220, 220), 2)
        label = ARROW[name]; sc = 0.6 if name == "stop" else 1.4
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, sc, 2)
        cv2.putText(canvas, label, (bx + (B - tw) // 2, by + (B + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, sc, (255, 255, 255), 2)
    x0 = CW + 20
    cv2.putText(canvas, "CAR CONTROL", (x0, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(canvas, ("Car connected" if car else "NO CAR FOUND"), (x0, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0) if car else (0, 0, 255), 1)
    cv2.putText(canvas, f"Speed {speed}  (+ / -)", (x0, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    d = dist[0]
    cv2.putText(canvas, f"Sensor: {'-' if d in (None, 999) else str(d) + ' cm'}", (x0, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(canvas, "Hold a button or W A S D", (x0, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(canvas, f"[ ] = nudge arrow ({vision.HEADING_TRIM[0]:+.0f} deg)", (x0, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(canvas, f"Target: {seen['obj'].upper()}  (C = color)", (x0, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(canvas, "B = bulldoze the target!", (x0, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 255), 1)
    cv2.putText(canvas, "F = follow target on/off", (x0, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(canvas, "space = stop   q = quit", (x0, 495), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(canvas, listener.status, (x0, 375), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(canvas, f"TARGET: {seen['obj']}  |  heard: {seen['heard'][:60]}", (15, CH - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(canvas, f"G = scan map  {objmap['status']}", (x0, 395), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    rx, ry, rw, rh = RESCAN; busy = objmap["busy"]
    cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), (0, 140, 255) if busy else (0, 150, 0), -1)
    cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), (220, 220, 220), 2)
    for i, t in enumerate(["SCANNING", "..."] if busy else ["RESCAN", "(G)"]):
        (tw, _), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(canvas, t, (rx + (rw - tw) // 2, ry + 36 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    md = seen.get("mood")
    if md and time.time() - md[1] < 2.5:
        txt, col = ("HAPPY! :D", (0, 255, 0)) if md[0] == "happy" else ("sad... :(", (255, 120, 80))
        cv2.putText(canvas, txt, (CW // 2 - 120, CH // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.6, col, 4)
    cv2.imshow(win, canvas)
send("S"); cam.release(); cv2.destroyAllWindows()
