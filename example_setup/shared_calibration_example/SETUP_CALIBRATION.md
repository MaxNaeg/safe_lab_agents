# Optics setup — calibration reference

Beam path: **Laser → fixed horizontal polarizer → λ/4 (QWP) → λ/2 (HWP) → rotatable polarizer → detector.**
Tools: `set_angle(angle, component)` for `lambda_quarter` / `lambda_half` / `polarizer`; `measure_power()` → `{'power': {'value': W, 'unit': 'W'}}`.

## Angle convention
`true_angle = set_angle + offset`. **0° = horizontal**, the axis defined by the fixed input polarizer.
For waveplates, true 0° means the fast axis is horizontal. To command a true angle: `set = true − offset`.

## System constants
- **P₀ ≈ 1.00 W** — full power reaching the detector (fully transmitted).
- **Background ≈ 0** — extinction goes to ~0 W with linear light; there is no dark offset.
- **Detector noise: additive Gaussian, σ ≈ 0.005 W, independent of power level.** (So relative noise is ~0.5 % near full scale but ~10 % near extinction. Average N shots → σ/√N. Use uniform-weight least squares.)
- **Both waveplates are ideal:** retardances 90.1° (λ/4) and 179.9° (λ/2).

## Calibrated offsets
### λ/4 QWP — fully calibrated (mod 90°)
- **offset o_q = 78.0° (mod 90°)**, ±0.03°.
- Fast/slow axis is **horizontal at set-angle ≈ 12.0°** (and 102.0°).
- Makes **circular light at set-angle ≈ 57.0°** (and 147.0°).
- ⚠️ Only known **mod 90°**: a power meter cannot sense handedness, so fast-vs-slow is unresolved (the fast-axis offset is 78° *or* 168°). This does not affect intensity predictions.

### λ/2 HWP + rotatable polarizer — degenerate (only the combination is measurable)
- ⚠️ **The HWP and analyzing polarizer offsets cannot be separated with a power meter.** They act as one effective analyzer at `2·hwp_true − pol_true`, so only the combination is observable:
  - **C = 2·o_h − o_p = 73.0° (mod 180°)**, ±0.03°.
- Power depends on the HWP and polarizer settings **only through `u = pol_set − 2·hwp_set`**.
- To assign individual numbers you must pick a gauge, e.g. `o_h = 0 → o_p = −73.0°`, or `o_p = 0 → o_h = +36.5°`. Separating them physically needs a polarization-resolving detector or a known reference polarization.

## Ready-to-use operating points (gauge-free, in set-angles)
- **Preserve horizontal linear light** (QWP acts neutral): `lambda_quarter = 12.0°` (or 102.0°).
- **Circular light after QWP:** `lambda_quarter = 57.0°` (or 147.0°).
- **With the QWP neutral**, light after the HWP is linear at true angle `2·hwp_true`, analyzed by the polarizer. The detected power follows Malus's law in `u = pol_set − 2·hwp_set`:
  - **P ≈ P₀ · cos²(u − 73.0°)**, i.e. **maximum at `u = 73.0°`**, **extinction at `u = 163.0°`** (both mod 180°).
- **Rotate the output linear polarization** by δ (true): increase `lambda_half` by δ/2 (HWP rotates polarization at 2× its own rotation).

## Confidence
Global Mueller-model fit over 864 points: reduced χ² = 1.01, residual RMS = σ. Blind hold-out test over the full 0–360° range: predicted vs measured RMS = 0.002 W (noise-limited). Full raw data, analysis, figures and scripts are in `/agent/shared/auto_log/` (see `CALIBRATION_SUMMARY.png`).
