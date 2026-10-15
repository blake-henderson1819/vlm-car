# Builds a 3D-printable ArUco marker tile (DICT_4X4_50, id 0) for the car.
# White base 56x56x1.2 mm with a front tab, black pattern raised 0.8 mm on top.
import struct, cv2, numpy as np
CELL, QUIET, BASE_H, TOP_H = 8.0, 4.0, 1.2, 0.8
D = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
bits = D.generateImageMarker(0, 6)          # 6x6 cells incl. black border; 0 = black
W = 6 * CELL + 2 * QUIET

def box(x0, y0, z0, x1, y1, z1):
    v = [(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0),(x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)]
    f = [(0,2,1),(0,3,2),(4,5,6),(4,6,7),(0,1,5),(0,5,4),(1,2,6),(1,6,5),(2,3,7),(2,7,6),(3,0,4),(3,4,7)]
    return [(v[a], v[b], v[c]) for a, b, c in f]

def prism_tab(y0, z0, z1, w=14, h=7):        # triangle tab on the FRONT edge (top of the marker image)
    cx = W / 2
    p = [(cx - w/2, y0), (cx + w/2, y0), (cx, y0 + h)]
    tris = []
    b = [(x, y, z0) for x, y in p]; t = [(x, y, z1) for x, y in p]
    tris += [(b[0], b[2], b[1]), (t[0], t[1], t[2])]
    for i in range(3):
        j = (i + 1) % 3
        tris += [(b[i], b[j], t[j]), (b[i], t[j], t[i])]
    return tris

def write_stl(path, tris):
    with open(path, "wb") as fh:
        fh.write(b"aruco marker".ljust(80, b" ")); fh.write(struct.pack("<I", len(tris)))
        for a, b, c in tris:
            n = np.cross(np.subtract(b, a), np.subtract(c, a)); n = n / (np.linalg.norm(n) or 1)
            fh.write(struct.pack("<12fH", *n, *a, *b, *c, 0))

# image row 0 = top of the marker = FRONT of the car. STL y grows toward the front.
white = box(0, 0, 0, W, W, BASE_H) + prism_tab(W, 0, BASE_H)
black = []
for r in range(6):
    for c in range(6):
        if bits[r, c] < 128:
            x0 = QUIET + c * CELL; y0 = QUIET + (5 - r) * CELL
            black += box(x0, y0, BASE_H, x0 + CELL, y0 + CELL, BASE_H + TOP_H)
write_stl("marker_white_base.stl", white)
write_stl("marker_black_pattern.stl", black)
write_stl("marker_combined_pause_swap.stl", white + black)

# check: render the top view and make sure OpenCV detects it the right way up
px = 10; img = np.full((int(W * px) + 200, int(W * px) + 200), 200, np.uint8)
o = 100
cv2.rectangle(img, (o, o), (o + int(W * px), o + int(W * px)), 255, -1)
for r in range(6):
    for c in range(6):
        if bits[r, c] < 128:
            x0 = o + int((QUIET + c * CELL) * px); y0 = o + int((QUIET + r * CELL) * px)
            cv2.rectangle(img, (x0, y0), (x0 + int(CELL * px) - 1, y0 + int(CELL * px) - 1), 0, -1)
cv2.imwrite("marker_preview.png", img)
det = cv2.aruco.ArucoDetector(D, cv2.aruco.DetectorParameters())
corners, ids, _ = det.detectMarkers(img)
print("detected ids:", None if ids is None else ids.ravel().tolist(), "corner0 (should be top-left):", None if ids is None else corners[0][0][0].round())
print("black cubes:", len(black) // 12, "size mm:", W)
