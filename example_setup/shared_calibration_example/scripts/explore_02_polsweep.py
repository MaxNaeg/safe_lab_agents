"""(1) Log the detector noise model. (2) Fine, averaged polarizer sweep at
qwp=hwp=0 to (a) confirm the ~0.05 W low-power floor (background / imperfect
extinction), (b) extract P0, and (c) get a first look at the analyzer phase.

Model for a polarizer sweep of angle s (set):
    P(s) = C_bg + P0*cos^2(pi/180*(s - phi)) = A + B*cos(2*pi/180*(s - phi))
with A = C_bg + P0/2, B = P0/2  => P0 = 2B, C_bg = A - B (the minimum).
phi = set-angle of maximum transmission.
"""
import sys
sys.path.insert(0, "/agent/shared/scripts"); sys.path.insert(0, "/agent/workspace")
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tools_client import set_angle, measure_power
from auto_log_client import start_batch, stop_batch, log_analysis, AUTO_LOG_DIR

SCR = "/tmp/claude-1001/-agent-workspace/5b062f49-458b-430d-8d6c-7c8ec035f7d0/scratchpad"
def P():
    d = measure_power()["power"]
    return float(d["value"]) if isinstance(d, dict) else float(d)

# ---- (1) Noise-model figure + log ----------------------------------------
nz = np.load(f"{SCR}/noise.npz")
means, stds = nz["means"], nz["stds"]
sigma_add = float(np.median(stds))
fig, ax = plt.subplots(figsize=(5,4))
ax.plot(means, stds, "o-", label="measured std")
ax.axhline(sigma_add, ls="--", c="k", label=f"additive model σ={sigma_add:.4f} W")
ax.plot(means, np.sqrt(means)*(stds[0]/np.sqrt(means[0])), ":", c="r", label="shot-noise ∝√P (rejected)")
ax.set_xlabel("mean power (W)"); ax.set_ylabel("std of 30 reps (W)")
ax.set_title("Detector noise: additive, power-independent"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{AUTO_LOG_DIR}/noise_model.png", dpi=110); plt.close(fig)

log_analysis(
    title="Detector noise model: additive Gaussian σ≈%.4f W" % sigma_add,
    kind="analysis",
    text=(
        "30 repeats at low/mid/high power (means 0.050, 0.499, 0.948 W) give "
        "stds 0.0058, 0.0046, 0.0049 W — essentially constant. Relative scatter "
        "falls 11.5%%→0.9%%→0.5%% with power, and std does not track √P, so the "
        "noise is ADDITIVE Gaussian with σ≈%.4f W, independent of level (not shot "
        "noise, not multiplicative). Consequence: use ordinary (uniform-weight) "
        "least squares; angle offsets come from sinusoid PHASE and are robust to "
        "this noise. A constant additive background does not shift phase. Will "
        "average a few shots per point on precision sweeps (σ_mean=σ/√N)."
        % sigma_add),
    data={"means": means, "stds": stds, "sigma_additive_W": sigma_add},
    references=["batch_20260702_110203_433705"],
    script=open(__file__).read(),
    figures=["noise_model.png"],
)

# ---- (2) Fine averaged polarizer sweep at qwp=hwp=0 ----------------------
set_angle(0.0, "lambda_quarter"); set_angle(0.0, "lambda_half")
angles = np.arange(0, 180, 5.0); NAVG = 8
start_batch("Fine polarizer sweep 0:5:180 (avg 8) at qwp=hwp=0")
pw = []
for a in angles:
    set_angle(float(a), "polarizer")
    pw.append(np.mean([P() for _ in range(NAVG)]))
stop_batch()
pw = np.array(pw)

def model(s, A, B, phi):
    return A + B*np.cos(2*np.pi/180.0*(s - phi))
p0 = [pw.mean(), (pw.max()-pw.min())/2, angles[pw.argmax()]]
popt, pcov = curve_fit(model, angles, pw, p0=p0, sigma=np.full_like(pw, sigma_add/np.sqrt(NAVG)), absolute_sigma=True)
A, B, phi = popt; perr = np.sqrt(np.diag(pcov))
phi = phi % 180.0
P0 = 2*B; Cbg = A - abs(B)
resid = pw - model(angles, *popt); rms = np.sqrt(np.mean(resid**2))

fig, ax = plt.subplots(figsize=(7,4))
ax.plot(angles, pw, "o", ms=4, label="data (avg 8)")
xs = np.linspace(0,180,400); ax.plot(xs, model(xs,*popt), "r-", label="A+B·cos(2(s−φ))")
ax.axhline(Cbg, ls=":", c="g", label=f"floor C_bg={Cbg:.4f} W")
ax.set_xlabel("polarizer set-angle (deg)"); ax.set_ylabel("power (W)")
ax.set_title(f"Polarizer sweep @ qwp=hwp=0:  P0={P0:.3f} W, peak φ={phi:.2f}°, floor={Cbg:.4f} W")
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(f"{AUTO_LOG_DIR}/polsweep_qh0.png", dpi=110); plt.close(fig)

print("A=%.4f±%.4f  B=%.4f±%.4f  phi=%.3f±%.3f (deg)  P0=%.4f  Cbg=%.4f  RMSresid=%.5f"
      % (A, perr[0], B, perr[1], phi, perr[2], P0, Cbg, rms))
print("extinction ratio Cbg/(Cbg+P0) = %.4f ; modulation depth = %.4f" % (Cbg/(Cbg+P0), B/A))

log_analysis(
    title="Fine polarizer sweep @ qwp=hwp=0 — P0, background, analyzer phase",
    kind="analysis",
    text=(
        f"Malus fit A+B·cos(2(s−φ)): P0=2B={P0:.4f} W, floor C_bg=A−|B|={Cbg:.4f} W "
        f"(~{100*Cbg/(Cbg+P0):.1f}%% of peak). The floor is REAL (well above the "
        f"cos²≈0.008 expected at extinction) — a constant background / finite "
        f"extinction ratio. It is an additive vertical offset, so it does NOT bias "
        f"angle offsets (those come from the phase φ). Peak transmission at set-angle "
        f"φ={phi:.2f}° (mod 180). Fit residual RMS={rms:.5f} W ≈ noise σ/√8, so the "
        f"sinusoid model is excellent. NOTE: φ mixes the polarizer offset with the "
        f"HWP-induced rotation (light direction = 2·hwp_true); it is NOT yet the pure "
        f"polarizer offset. Will fit A,B(amplitude),phi per sweep in the full scans; "
        f"amplitude B is background-immune and is what the QWP calibration uses."),
    data={"angles": angles, "power": pw, "A": A, "B": B, "phi_deg": phi,
          "P0": P0, "C_bg": Cbg, "resid_rms": rms, "perr": perr},
    references=["batch_20260702_110113_319978"],
    script=open(__file__).read(),
    figures=["polsweep_qh0.png"],
)
np.savez(f"{SCR}/polsweep.npz", angles=angles, power=pw, popt=popt, P0=P0, Cbg=Cbg)
print("DONE")
