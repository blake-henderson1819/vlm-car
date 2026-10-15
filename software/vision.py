# Finds the car (bright blue base plate + black battery box at its front) and a black target in the overhead image.
import math, os
import cv2, numpy as np

HEADING_TRIM = [0.0]   # degrees, adjusted live with [ and ] in the control pad

def dark_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (0, 0, 0), (180, 255, 55))
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

_ARUCO = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), cv2.aruco.DetectorParameters())

def find_car(img):
    return find_car_aruco(img)          # marker only; the old blue/black color fallback mistook black wheels for the battery box

def find_car_aruco(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = _ARUCO.detectMarkers(g)
    mirrored = False
    if ids is None or 0 not in ids.ravel():          # printed marker may be mirrored (viewed from its back side)
        corners, ids, _ = _ARUCO.detectMarkers(cv2.flip(g, 1)); mirrored = True
        if ids is None or 0 not in ids.ravel(): return None
    c = corners[list(ids.ravel()).index(0)][0].copy()
    if mirrored: c[:, 0] = g.shape[1] - 1 - c[:, 0]          # TL, TR, BR, BL  (top edge = front of the car)
    center = c.mean(0); front = (c[0] + c[1]) / 2
    side = float(np.mean([np.hypot(*(c[i] - c[(i + 1) % 4])) for i in range(4)]))
    heading = math.atan2(front[1] - center[1], front[0] - center[0]) + math.radians(HEADING_TRIM[0])
    pad = 1.6 * side
    return {"center": center, "front": front, "heading": heading, "size": 1.6 * side, "arrow": None, "aruco": c,
            "sensor_line": (c[0], c[1]), "dark": cv2.connectedComponentsWithStats(dark_mask(img)),
            "bbox": (center[0] - pad, center[1] - pad, center[0] + pad, center[1] + pad)}

def find_car_color(img):
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, (95, 140, 110), (115, 255, 255))        # blue plate (carpet is excluded because it touches the frame edge)
    blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(blue)
    parts = [i for i in range(1, n) if st[i, 4] > 600
             and st[i, 0] > 2 and st[i, 1] > 2 and st[i, 0] + st[i, 2] < W - 2 and st[i, 1] + st[i, 3] < H - 2]  # ignore things touching the edge
    if not parts: return None
    main = max(parts, key=lambda i: st[i, 4])
    group = [i for i in parts if np.hypot(*(cen[i] - cen[main])) < 160]   # wires split the plate into pieces
    area = sum(st[i, 4] for i in group)
    if area < 2500: return None
    P = sum(cen[i] * st[i, 4] for i in group) / area
    x0 = min(st[i, 0] for i in group); y0 = min(st[i, 1] for i in group)
    x1 = max(st[i, 0] + st[i, 2] for i in group); y1 = max(st[i, 1] + st[i, 3] for i in group)
    size = math.sqrt(area)
    dk = cv2.connectedComponentsWithStats(dark_mask(img))
    n2, lab2, st2, cen2 = dk
    # the battery box sits ON the plate: pick the dark blob with the most pixels touching the (grown) plate
    near = cv2.dilate(np.isin(lab, group).astype(np.uint8), np.ones((61, 61), np.uint8)) > 0
    best, best_ov = None, 0
    for j in range(1, n2):
        if st2[j, 4] < 1500: continue
        ov = int(np.count_nonzero(near & (lab2 == j)))
        if ov > best_ov and ov > 0.05 * st2[j, 4]: best, best_ov = j, ov
    if best is None: return None
    F = cen2[best]
    # Heading = normal to the battery box's front face (the face the ultrasonic sensor is mounted on).
    # Fit a rotated rectangle to the box, then pick whichever of its 4 face normals points most away from the plate.
    # The box is navy-black (tinted blue), the tires are neutral black: use that to cut the wheels off the box.
    navy = cv2.inRange(hsv, (100, 90, 0), (125, 255, 75))
    navy = cv2.morphologyEx(navy, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    nn, nl, ns, nc = cv2.connectedComponentsWithStats(navy)
    keep = [i for i in range(1, nn) if ns[i, 4] > 500 and np.count_nonzero(near & (nl == i)) > 0.1 * ns[i, 4]
            and np.hypot(*(nc[i] - cen2[best])) < 1.2 * size]
    if keep:
        bpts = np.column_stack(np.where(np.isin(nl, keep)))[:, ::-1].astype(np.float32)
    else:
        bpts = np.column_stack(np.where(lab2 == best))[:, ::-1].astype(np.float32)
    (bcx, bcy), (bw, bh), ba = cv2.minAreaRect(bpts)
    plate_c = sum(cen[i] * st[i, 4] for i in group) / area
    away = np.array([bcx, bcy]) - plate_c
    cands = [math.radians(ba + 90 * k) for k in range(4)]
    heading_raw = max(cands, key=lambda t: math.cos(t) * away[0] + math.sin(t) * away[1])
    k = round((math.degrees(heading_raw) - ba) / 90) % 2
    half = (bw if k == 0 else bh) / 2            # distance from box center to the front face
    span = (bh if k == 0 else bw) / 2            # half-width of the front face
    hv = np.array([math.cos(heading_raw), math.sin(heading_raw)]); nv = np.array([-hv[1], hv[0]])
    face_c = np.array([bcx, bcy]) + hv * half
    sensor_line = (face_c - nv * span, face_c + nv * span)
    # Preferred: the white arrow stuck on top of the battery box.
    arrow = find_white_arrow(hsv, ((bcx, bcy), (bw, bh), ba))
    if arrow is not None:
        heading_raw = arrow["heading"]
        hv = np.array([math.cos(heading_raw), math.sin(heading_raw)]); nv = np.array([-hv[1], hv[0]])
    heading = heading_raw + math.radians(HEADING_TRIM[0])
    P = np.array([bcx, bcy]) - hv * half         # pivot roughly at the car's middle
    bx0, by0 = min(x0, st2[best, 0]), min(y0, st2[best, 1])
    bx1, by1 = max(x1, st2[best, 0] + st2[best, 2]), max(y1, st2[best, 1] + st2[best, 3])
    pad = int(0.6 * size)
    return {"center": P, "front": face_c if arrow is None else arrow["tip"], "heading": heading, "size": size,
            "sensor_line": sensor_line, "arrow": arrow,
            "bbox": (bx0 - pad, by0 - pad, bx1 + pad, by1 + pad), "dark": dk}

def find_black_target(img, car):
    n2, lab2, st2, cen2 = car["dark"] if car else cv2.connectedComponentsWithStats(dark_mask(img))
    best, score = None, 0
    for j in range(1, n2):
        a = st2[j, 4]; w, h = st2[j, 2], st2[j, 3]
        if a < 400: continue
        if car:
            x0, y0, x1, y1 = car["bbox"]
            if x0 <= cen2[j][0] <= x1 and y0 <= cen2[j][1] <= y1: continue   # part of the car (wheels, battery box)
        if st2[j, 0] <= 2 or st2[j, 1] <= 2: continue                       # touching the frame edge
        squareness = min(w, h) / max(w, h); fill = a / float(w * h)
        s = a * squareness * fill
        if squareness > 0.5 and fill > 0.5 and s > score: best, score = j, s
    return None if best is None else cen2[best]

def angle_error(car, target):
    want = math.atan2(target[1] - car["center"][1], target[0] - car["center"][0])
    return (want - car["heading"] + math.pi) % (2 * math.pi) - math.pi   # + = target is clockwise on screen

def find_orange_target(img, car):
    """Center of the orange can: largest strongly-orange blob outside the car."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (8, 140, 150), (30, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    best = None
    for j in range(1, n):
        if st[j, 4] < 300: continue
        if car:
            x0, y0, x1, y1 = car["bbox"]
            if x0 <= cen[j][0] <= x1 and y0 <= cen[j][1] <= y1: continue
        if best is None or st[j, 4] > st[best, 4]: best = j
    if best is None: return None
    ys, xs = np.where(lab == best)
    (cx, cy), r = cv2.minEnclosingCircle(np.column_stack([xs, ys]).astype(np.float32))
    return np.array([cx, cy])

COLOR_RANGES = {
    "red":    [((0, 140, 90), (8, 255, 255)), ((170, 140, 90), (180, 255, 255))],
    "orange": [((8, 140, 150), (30, 255, 255))],
}

def find_color_target(img, car, color="red"):
    """Center of the biggest blob of the given color outside the car."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = None
    for lo, hi in COLOR_RANGES[color]:
        r = cv2.inRange(hsv, lo, hi); m = r if m is None else (m | r)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    best = None
    for j in range(1, n):
        if st[j, 4] < 120: continue
        if car:
            x0, y0, x1, y1 = car["bbox"]
            if x0 <= cen[j][0] <= x1 and y0 <= cen[j][1] <= y1: continue
        if best is None or st[j, 4] > st[best, 4]: best = j
    return None if best is None else cen[best]

def find_white_arrow(hsv, box_rect):
    """White arrow on the battery box. Returns heading (toward the arrowhead), tip, and its pixels' hull."""
    (bcx, bcy), (bw, bh), ba = box_rect
    roi = np.zeros(hsv.shape[:2], np.uint8)
    poly = cv2.boxPoints(((bcx, bcy), (bw + 30, bh + 30), ba)).astype(np.int32)
    cv2.fillConvexPoly(roi, poly, 255)
    white = cv2.inRange(hsv, (0, 0, 170), (180, 70, 255)) & roi
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(white)
    if n < 2: return None
    j = 1 + int(np.argmax(st[1:, 4]))
    if st[j, 4] < 800: return None          # the real arrow is ~1200 px; smaller white bits are noise
    if os.environ.get("ARROW_DEBUG"): print("arrow area", st[j, 4], st[j, 2], st[j, 3])
    pts = np.column_stack(np.where(lab == j))[:, ::-1].astype(np.float32)
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    axis = vt[0]; perp = np.array([-axis[1], axis[0]])
    proj = (pts - c) @ axis; side = (pts - c) @ perp
    a, b = side[proj > 0], side[proj < 0]
    if len(a) < 10 or len(b) < 10: return None
    if np.ptp(b) > np.ptp(a): axis = -axis; proj = -proj    # the wider half is the arrowhead
    tip = c + axis * proj.max()
    return {"heading": math.atan2(axis[1], axis[0]), "tip": tip, "hull": cv2.convexHull(pts.astype(np.int32))}

# ---------- dark objects (black cube / black mouse) told apart by shape ----------
def dark_objects(img, car=None):
    """All dark, compact blobs on the table that are not part of the car. Each: dict(center, box, rect, area, aspect)."""
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    vmax = int(min(105, max(45, 0.55 * np.median(hsv[..., 2]))))        # adapts to room lighting
    m = cv2.inRange(hsv, (0, 0, 0), (180, 130, vmax))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    blue = cv2.morphologyEx(cv2.inRange(hsv, (95, 120, 80), (125, 255, 255)), cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    nb, lb, sb, _ = cv2.connectedComponentsWithStats(blue)                # keep only the big blue plate
    blue = np.isin(lb, [j for j in range(1, nb) if sb[j, 4] > 3000]).astype(np.uint8) * 255
    blue = cv2.dilate(blue, np.ones((25, 25), np.uint8))
    out = []
    for c in cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        area = cv2.contourArea(c)
        if area < 600 or area > 0.05 * W * H: continue
        x, y, w, h = cv2.boundingRect(c)
        if x <= 3 or y <= 3 or x + w >= W - 3 or y + h >= H - 3: continue     # frame edge (people, chairs)
        (cx, cy), (rw, rh), ang = cv2.minAreaRect(c)
        if car is not None:
            side = car["size"] / 1.6
            if np.hypot(cx - car["center"][0], cy - car["center"][1]) < 2.6 * side: continue   # marker, battery box, wheels
        if cv2.countNonZero(blue[y:y + h, x:x + w]) > 0.05 * w * h: continue      # touching the car's blue plate
        hull = cv2.contourArea(cv2.convexHull(c))
        solidity = area / max(hull, 1)
        if solidity < 0.8: continue                                           # shadows / ragged blobs
        out.append({"center": np.array([cx, cy]), "box": (x, y, w, h), "area": area,
                    "aspect": min(rw, rh) / max(rw, rh, 1), "solidity": solidity})
    return out

def find_dark_target(img, car, kind="mouse"):
    """kind='cube': squarish (aspect>0.75); kind='mouse': elongated oval (aspect<0.8), bigger. Returns center or None."""
    objs = dark_objects(img, car)
    if kind == "cube":
        c = [o for o in objs if o["aspect"] > 0.75]
        c.sort(key=lambda o: (o["aspect"], o["solidity"]), reverse=True)
    else:
        c = [o for o in objs if 0.35 < o["aspect"] < 0.8]
        c.sort(key=lambda o: o["area"], reverse=True)
        if not c: c = sorted([o for o in objs if o["aspect"] <= 0.75 or o["area"] > 4000], key=lambda o: -o["area"])
    return c[0]["center"] if c else None
