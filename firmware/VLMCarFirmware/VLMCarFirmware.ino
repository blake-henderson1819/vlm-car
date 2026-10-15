// VLM Car firmware (Uno). Same wiring as the figure-eight / obstacle car.
// L293D: left EN=5 IN=3,4   right EN=6 IN=7,8     HC-SR04 (optional): Trig=12 Echo=11
// Serial 115200. Commands (one per line):
//   M <left> <right> <ms>   drive wheels at -255..255 for ms, then stop
//   S                       stop now
//   H <n>                   happy chirp (variant n)     A <n>  sad beeps (variant n)   (passive buzzer on D10)
// Reports "D <cm>" five times a second (999 = nothing in range).
const int ENL = 5, L1 = 3, L2 = 4, ENR = 6, R1 = 7, R2 = 8, TRIG = 12, ECHO = 11;
const int BUZ = 10;
unsigned long stopAt = 0, lastRep = 0;

// ---- non-blocking buzzer melodies: {freq Hz (0 = rest), ms}, ended by {0,0}. Each < 1.2 s ----
struct Note { int f; int ms; };
const Note HAP0[] = {{523,90},{659,90},{784,90},{1047,220},{0,0}};                                  // rising arpeggio
const Note HAP1[] = {{1800,60},{2400,50},{1500,60},{2600,70},{0,30},{2000,50},{2800,60},{1700,80},{3000,90},{0,0}}; // R2-D2 warble
const Note HAP2[] = {{1568,45},{1976,45},{1568,45},{1976,45},{1568,45},{1976,45},{2349,200},{0,0}}; // trill
const Note HAP3[] = {{784,80},{0,40},{784,80},{1175,300},{0,0}};                                    // "yay!"
const Note SAD0[] = {{392,250},{370,250},{349,250},{294,420},{0,0}};                                 // wah-wah descending
const Note SAD1[] = {{660,300},{620,300},{560,450},{0,0}};                                           // slow whimper
const Note SAD2[] = {{523,150},{494,150},{440,150},{392,150},{349,150},{262,350},{0,0}};               // sad sigh
const Note* const HAPPY[] = {HAP0, HAP1, HAP2, HAP3};
const Note* const SAD[] = {SAD0, SAD1, SAD2};
const Note* song = 0; int noteI = 0; unsigned long noteEnd = 0;
void play(const Note* s) { song = s; noteI = -1; noteEnd = 0; }
void updateSong() {
  if (!song || millis() < noteEnd) return;
  noteI++;
  const Note n = song[noteI];
  if (n.f == 0 && n.ms == 0) { noTone(BUZ); song = 0; return; }
  if (n.f) tone(BUZ, n.f); else noTone(BUZ);
  noteEnd = millis() + n.ms;
}
String line;

void drive(int left, int right) {
  left = -left; right = -right;     // motors mounted reversed (same as ObstacleCar)
  digitalWrite(L1, left >= 0);  digitalWrite(L2, left < 0);  analogWrite(ENL, min(255, abs(left)));
  digitalWrite(R1, right >= 0); digitalWrite(R2, right < 0); analogWrite(ENR, min(255, abs(right)));
}

int distanceCm() {
  digitalWrite(TRIG, LOW); delayMicroseconds(2);
  digitalWrite(TRIG, HIGH); delayMicroseconds(10); digitalWrite(TRIG, LOW);
  unsigned long t = pulseIn(ECHO, HIGH, 20000);
  return t == 0 ? 999 : (int)(t * 0.0343 / 2);
}

void setup() {
  int pins[] = {ENL, L1, L2, ENR, R1, R2, TRIG};
  for (int p : pins) pinMode(p, OUTPUT);
  pinMode(ECHO, INPUT); pinMode(BUZ, OUTPUT);
  drive(0, 0);
  Serial.begin(115200);
  Serial.println("READY");
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      line.trim();
      if (line.startsWith("M")) {
        int l, r; long ms;
        if (sscanf(line.c_str(), "M %d %d %ld", &l, &r, &ms) == 3) { drive(l, r); stopAt = millis() + constrain(ms, 0, 5000); }
      } else if (line.startsWith("S")) { drive(0, 0); stopAt = 0; }
      else if (line.startsWith("H") || line.startsWith("A")) {
        int n = 0; sscanf(line.c_str() + 1, "%d", &n); n = abs(n);
        if (line[0] == 'H') play(HAPPY[n % 4]); else play(SAD[n % 3]);
      }
      line = "";
    } else if (line.length() < 40) line += c;
  }
  updateSong();
  if (stopAt && millis() > stopAt) { drive(0, 0); stopAt = 0; }
  if (millis() - lastRep > 200) { lastRep = millis(); Serial.print("D "); Serial.println(distanceCm()); }
}
