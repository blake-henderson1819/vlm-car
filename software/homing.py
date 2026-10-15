# Reverse-into-home controller (pure logic, shared by pad.py and the offline simulation).
import math

MIN_MS = 180                              # shorter full-power pulses just slip the wheels

def wrap(a): return (a + math.pi) % (2 * math.pi) - math.pi

class ReverseHome:
    START_DEG, KEEP_DEG = 25, 45          # start reversing within 25 deg, keep reversing while within 45 deg (hysteresis)
    def __init__(self, arc_slow=130):
        self.arc_slow = arc_slow          # slow wheel speed in reverse arcs (both wheels always negative)
        self.reversing = False
        self.rate = 0.06                  # learned deg per ms of spin pulse (starts conservative-high = short pulses)
        self.last_turn = None             # (err_before, ms, kind)
        self.osc = 0                      # times a minimum-length pulse overshot (car turns faster than MIN_MS allows)

    def _learn(self, err):
        if not self.last_turn: return
        e0, ms, kind = self.last_turn; self.last_turn = None
        turned = e0 - err                                     # how far the error moved
        if e0 * err < 0:                                     # overshot past zero: we turn faster than we thought
            self.rate = min(1.5, self.rate * 1.6)
            if ms <= MIN_MS + 5: self.osc += 1               # even the shortest pulse overshoots -> accept a wider window
        elif abs(turned) > 3 and turned * e0 > 0:
            r = abs(turned) / ms
            self.rate = 0.6 * self.rate + 0.4 * max(0.01, min(1.5, r))

    def step(self, center, heading, home, size):
        """Returns (cmd, wait_s, info). cmd: ('spin', dir, ms) = reverse ARC pulse | ('rev', left, right, ms) | ('done',).
        Image coords (y down): + error = home is clockwise of the car's rear; dir +1 = clockwise."""
        dx, dy = home[0] - center[0], home[1] - center[1]
        dist = math.hypot(dx, dy)
        rear = heading + math.pi
        err = math.degrees(wrap(math.atan2(dy, dx) - rear))
        self._learn(err)
        if dist < 0.35 * size: return ("done",), 0, (err, dist)
        lim = self.KEEP_DEG if self.reversing else min(70, self.START_DEG + 20 * self.osc)
        if abs(err) > lim:
            self.reversing = False
            d = 1 if err > 0 else -1
            ms = int(max(MIN_MS, min(450, 0.8 * abs(err) / self.rate)))   # reverse arc pulse, aim for 80%, never below MIN_MS
            self.last_turn = (err, ms, "spin")
            return ("spin", d, ms), 0.3, (err, dist)            # then let car + camera settle
        self.reversing = True
        slow = int(max(self.arc_slow, 255 - abs(err) * 4))
        l, r = (-slow, -255) if err > 0 else (-255, -slow)   # clockwise while reversing: left less negative; never below arc_slow
        ms = 250 if dist > 1.5 * size else 140
        return ("rev", l, r, ms), 0.1, (err, dist)
