"""Scan B (acquisition only): 2D scan hwp_set x pol_set, QWP parked at its
neutral (qwp_set = 11.96 -> qwp_true = 0, horizontal linear into the HWP).

With linear light, the analyzer peak sits at pol_true = 2*hwp_true, i.e.
    pol_set_peak = 2*hwp_set + C,   C = 2*o_h - o_p.
Power depends only on u = pol_set - 2*hwp_set (=> o_h/o_p degeneracy demo).
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

QWP_NEUTRAL = 11.96          # from Scan A (qwp_true = 0)
hwp_ax = np.arange(0, 180, 5.0)     # 36
pol_ax = np.arange(0, 180, 15.0)    # 12

set_angle(QWP_NEUTRAL, "lambda_quarter")
power = np.full((hwp_ax.size, pol_ax.size), np.nan)

ret = start_batch("Scan B: hwp(0:5:180) x pol(0:15:180), qwp=neutral(11.96)",
                  "2D scan for C=2o_h-o_p via analyzer peak vs HWP; degeneracy demo.")
print("start_batch:", ret)
for i, h in enumerate(hwp_ax):
    set_angle(float(h), "lambda_half")
    for j, p in enumerate(pol_ax):
        set_angle(float(p), "polarizer")
        power[i, j] = P()
stop = stop_batch()
print("stop_batch:", stop)

np.savez(f"{SCR}/scanB.npz", hwp_ax=hwp_ax, pol_ax=pol_ax, power=power, qwp_neutral=QWP_NEUTRAL)
print("saved scanB.npz shape", power.shape,
      " Pmin=%.4f Pmax=%.4f" % (np.nanmin(power), np.nanmax(power)))
print("DONE")
