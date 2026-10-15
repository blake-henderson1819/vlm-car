# Wiring

Arduino Uno + L293D (16-pin DIP) driving two TT motors, plus a buzzer and an HC-SR04.

## L293D

| L293D pin | Name | Connect to |
|---|---|---|
| 1 | Enable 1,2 (left) | Arduino **D5** (PWM) |
| 2 | Input 1 (left) | Arduino **D3** |
| 3 | Output 1 | Left motor terminal A |
| 4, 5 | GND | GND |
| 6 | Output 2 | Left motor terminal B |
| 7 | Input 2 (left) | Arduino **D4** |
| 8 | Vs (motor power) | Arduino **VIN** (battery in barrel jack) or 5V |
| 9 | Enable 3,4 (right) | Arduino **D6** (PWM) |
| 10 | Input 3 (right) | Arduino **D7** |
| 11 | Output 3 | Right motor terminal A |
| 12, 13 | GND | GND |
| 14 | Output 4 | Right motor terminal B |
| 15 | Input 4 (right) | Arduino **D8** |
| 16 | Vss (logic 5V) | Arduino **5V** |

If a wheel spins the wrong way, swap that motor's two wires.

## Other parts

| Part | Pin | Arduino |
|---|---|---|
| Buzzer | + | **D10** |
| Buzzer | − | GND |
| HC-SR04 | Trig | **D12** |
| HC-SR04 | Echo | **D11** |
| HC-SR04 | VCC / GND | 5V / GND |

## Power

- Recommended: **4–6 × AA** pack into the Uno barrel jack, L293D pin 8 from **VIN**.
- A 9 V battery works but drains fast under motor load.
- On USB only you can feed pin 8 from 5V, but the motors will be weak and may brown out the board.
- All grounds (Arduino, L293D pins 4/5/12/13, battery, sensors) must be common.
