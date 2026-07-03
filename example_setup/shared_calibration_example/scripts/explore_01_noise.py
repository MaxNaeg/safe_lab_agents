"""Characterize the detector noise model.

At (0,0,0) we saw ~10% relative scatter, but that was near extinction (low
power). Question: is the noise additive (constant sigma in W) or multiplicative
(constant relative sigma)? And is there a constant background / dark offset?

Strategy: park the polarizer at set-angles that (from the coarse sweep) give
low (~0.05 W, pol=0), mid (~0.5 W, pol=40), and high (~0.95 W, pol=80) power,
with qwp=hwp=0. Take 30 repeats at each and look at mean vs std.
"""
import sys
sys.path.insert(0, "/agent/shared/scripts"); sys.path.insert(0, "/agent/workspace")
import numpy as np
from tools_client import set_angle, measure_power
from auto_log_client import start_batch, stop_batch

SCR = "/tmp/claude-1001/-agent-workspace/5b062f49-458b-430d-8d6c-7c8ec035f7d0/scratchpad"

def P():
    d = measure_power()["power"]
    return float(d["value"]) if isinstance(d, dict) else float(d)

set_angle(0.0, "lambda_quarter")
set_angle(0.0, "lambda_half")

levels = {"low(pol=0)": 0.0, "mid(pol=40)": 40.0, "high(pol=80)": 80.0}
N = 30
results = {}
start_batch("Noise characterization: 30 reps at low/mid/high power")
for name, pol in levels.items():
    set_angle(pol, "polarizer")
    vals = np.array([P() for _ in range(N)])
    results[name] = vals
stop_batch()

print("level            mean        std        rel_std(%)   min        max")
rows = []
for name, vals in results.items():
    print("%-15s  %.5f    %.5f    %8.2f    %.5f   %.5f"
          % (name, vals.mean(), vals.std(), 100*vals.std()/vals.mean(), vals.min(), vals.max()))
    rows.append((results_key := name, vals.mean(), vals.std()))

means = np.array([results[k].mean() for k in levels])
stds  = np.array([results[k].std()  for k in levels])
# Fit std^2 = a + b*mean^2  (a=additive var, b=multiplicative rel-var)
# and also check std vs mean linear (shot-noise-like std ~ sqrt(mean))
print("\nmeans:", means)
print("stds :", stds)
print("std/mean:", stds/means)
print("std/sqrt(mean):", stds/np.sqrt(means))
print("std/mean (should be flat if multiplicative):", stds/means)

np.savez(f"{SCR}/noise.npz",
         low=results["low(pol=0)"], mid=results["mid(pol=40)"], high=results["high(pol=80)"],
         means=means, stds=stds)
print("\nDONE")
