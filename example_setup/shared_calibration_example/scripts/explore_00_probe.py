"""Probe the optics instrument: return format, power scale, noise, and a
coarse polarizer sweep so we can design the full calibration afterwards.

Setup (fixed order): Laser -> Horizontal polarizer -> QWP -> HWP -> rotatable
Polarizer -> Detector. set_angle offsets are unknown; we calibrate them.
"""
import sys
sys.path.insert(0, "/agent/shared/scripts")
sys.path.insert(0, "/agent/workspace")
import numpy as np
from tools_client import set_angle, measure_power
from auto_log_client import start_batch, stop_batch

def P():
    """measure_power -> float power in Watts (handles nested {'value','unit'})."""
    d = measure_power()["power"]
    return float(d["value"]) if isinstance(d, dict) else float(d)

# --- 1. Single calls: inspect return formats -------------------------------
print("set_angle return:", repr(set_angle(0.0, "polarizer")))
print("set qwp:", repr(set_angle(0.0, "lambda_quarter")))
print("set hwp:", repr(set_angle(0.0, "lambda_half")))
m = measure_power()
print("measure_power return:", repr(m), "type:", type(m))

# --- 2. Noise: repeat measurement at fixed config --------------------------
start_batch("Probe: noise (20x at qwp=hwp=pol=0)")
noise = []
for _ in range(20):
    noise.append(P())
stop_batch()
pvals = np.array(noise, dtype=float)
print("\nNoise over 20 reps at (0,0,0):")
print("  mean = %.6g   std = %.3g   rel_std = %.3g%%   min=%.6g max=%.6g"
      % (pvals.mean(), pvals.std(), 100*pvals.std()/max(pvals.mean(),1e-30), pvals.min(), pvals.max()))

# --- 3. Coarse polarizer sweep at qwp=0, hwp=0 -----------------------------
set_angle(0.0, "lambda_quarter")
set_angle(0.0, "lambda_half")
angles = np.arange(0, 360, 20.0)
start_batch("Probe: coarse polarizer sweep 0:20:360 at qwp=hwp=0")
sweep = []
for a in angles:
    set_angle(float(a), "polarizer")
    sweep.append(P())
stop_batch()
sweep = np.array(sweep)
print("\nCoarse polarizer sweep at qwp=hwp=0:")
for a, p in zip(angles, sweep):
    print("  pol=%6.1f  P=%.6g" % (a, p))
print("  Pmax=%.6g at %s deg ; Pmin=%.6g at %s deg"
      % (sweep.max(), angles[sweep.argmax()], sweep.min(), angles[sweep.argmin()]))
print("  modulation depth (max-min)/(max+min) = %.4f"
      % ((sweep.max()-sweep.min())/(sweep.max()+sweep.min())))

np.savez("/tmp/claude-1001/-agent-workspace/5b062f49-458b-430d-8d6c-7c8ec035f7d0/scratchpad/probe.npz",
         noise=pvals, angles=angles, sweep=sweep)
print("\nDONE")
