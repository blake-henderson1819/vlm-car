# ArUco marker tile

The car is tracked by an overhead webcam using an **ArUco marker, DICT_4X4_50, id 0**.
The marker's **top edge = the car's front**. If the marker is printed mirrored, the vision code falls back to detecting the mirrored version.

| File | What it is |
|---|---|
| `make_marker.py` | Generates the STLs and preview (56×56×1.2 mm white base with front tab, black pattern raised 0.8 mm). Run: `python make_marker.py` |
| `marker_white_base.stl` | White base body |
| `marker_black_pattern.stl` | Black raised pattern (print as second color / second part) |
| `marker_combined_pause_swap.stl` | Single-body version: add a pause at the layer where the pattern starts and swap to black filament |
| `marker_preview.png` | 2D preview of the pattern |

Mount it flat on top of the car, tab pointing forward. A paper printout works too.
