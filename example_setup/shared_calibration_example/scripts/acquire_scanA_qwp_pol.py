"""Scan A (acquisition only): 2D scan qwp_set x pol_set, HWP fixed at 0.

Goal: at each QWP angle, a full polarizer sweep. The sweep AMPLITUDE
= (P0/2)|cos(2*qwp_true)| encodes the QWP offset (peaks at qwp_true=0 mod 90),
decoupled from the HWP/polarizer offsets. Analysis is done separately.
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

qwp_ax = np.arange(0, 180, 5.0)     # 36
pol_ax = np.arange(0, 180, 15.0)    # 12
HWP_FIXED = 0.0

set_angle(HWP_FIXED, "lambda_half")
power = np.full((qwp_ax.size, pol_ax.size), np.nan)

ret = start_batch("Scan A: qwp(0:5:180) x pol(0:15:180), hwp=0",
                  "2D scan for QWP-offset calibration via polarizer-sweep amplitude.")
print("start_batch:", ret)
for i, q in enumerate(qwp_ax):
    set_angle(float(q), "lambda_quarter")
    for j, p in enumerate(pol_ax):
        set_angle(float(p), "polarizer")
        power[i, j] = P()
stop = stop_batch()
print("stop_batch:", stop)

np.savez(f"{SCR}/scanA.npz", qwp_ax=qwp_ax, pol_ax=pol_ax, power=power, hwp_fixed=HWP_FIXED)
print("saved scanA.npz  shape", power.shape,
      " Pmin=%.4f Pmax=%.4f" % (np.nanmin(power), np.nanmax(power)))
print("DONE")
