"""Master calibration summary figure (2x3) + final ELN summary entry.

Panels: (1) Scan A heatmap, (2) QWP amplitude fit -> o_q, (3) Scan B heatmap,
(4) degeneracy collapse -> C, (5) global-fit validation, (6) results text.
"""
import sys
sys.path.insert(0, "/agent/shared/scripts"); sys.path.insert(0, "/agent/workspace")
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from physics_model import predict_power
from auto_log_client import log_analysis, AUTO_LOG_DIR

SCR = "/tmp/claude-1001/-agent-workspace/5b062f49-458b-430d-8d6c-7c8ec035f7d0/scratchpad"
A = np.load(f"{SCR}/scanA.npz"); B = np.load(f"{SCR}/scanB.npz")
fa = np.load(f"{SCR}/scanA_fit.npz"); fb = np.load(f"{SCR}/scanB_fit.npz")
g  = np.load(f"{SCR}/global_fit.npz")
o_q = float(g["o_q"]); o_q_err = float(fa["o_q_err"]); C = float(g["C"]); C_err = float(fb["C_err"])
P0 = float(g["P0"]); Cbg = float(g["Cbg"]); redchi2 = float(g["redchi2"]); rms = float(g["rms"])
ret_q = float(g["ret_q"]); ret_h = float(g["ret_h"])

plt.rcParams.update({"font.size": 10})
fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.2))

# (1) Scan A heatmap
ax = axes[0,0]
im = ax.pcolormesh(A["pol_ax"], A["qwp_ax"], A["power"], shading="auto", cmap="viridis")
for qn in fa["neutrals"]: ax.axhline(qn, color="r", ls="--", lw=1)
for qc in fa["circulars"]: ax.axhline(qc, color="w", ls=":", lw=1)
ax.set_xlabel("polarizer set-angle (°)"); ax.set_ylabel("QWP set-angle (°)")
ax.set_title("① Scan A — power(QWP, pol), HWP=0\nred=linear(neutral), white=circular")
fig.colorbar(im, ax=ax, label="P (W)")

# (2) QWP amplitude fit
ax = axes[0,1]
amp, qax = fa["amp"], fa["qwp"]
ax.plot(qax, amp, "o", ms=4, color="#1f77b4", label="sweep amplitude")
xs = np.linspace(0,180,500)
ax.plot(xs, (P0/2)*np.abs(np.cos(np.deg2rad(2*(xs+o_q)))), "r-",
        label=r"$(P_0/2)\,|\cos 2(q+o_q)|$")
for qn in fa["neutrals"]: ax.axvline(qn, color="r", ls="--", lw=1)
ax.set_xlabel("QWP set-angle (°)"); ax.set_ylabel("amplitude (W)")
ax.set_title(f"② QWP calibration → $o_q$={o_q:.2f}±{o_q_err:.2f}° (mod 90)")
ax.legend(fontsize=8, loc="upper right")

# (3) Scan B heatmap
ax = axes[0,2]
im = ax.pcolormesh(B["pol_ax"], B["hwp_ax"], B["power"], shading="auto", cmap="magma")
hh = np.linspace(0,180,400)
for k in (-2,-1,0,1,2): ax.plot(2*hh + C + 180*k, hh, "c--", lw=1)
ax.set_xlim(B["pol_ax"].min(), B["pol_ax"].max()); ax.set_ylim(B["hwp_ax"].min(), B["hwp_ax"].max())
ax.set_xlabel("polarizer set-angle (°)"); ax.set_ylabel("HWP set-angle (°)")
ax.set_title("③ Scan B — power(HWP, pol), QWP neutral\ncyan: peak line slope +2")
fig.colorbar(im, ax=ax, label="P (W)")

# (4) degeneracy collapse
ax = axes[1,0]
H, Pp = np.meshgrid(B["hwp_ax"], B["pol_ax"], indexing="ij")
u = ((Pp - 2*H) % 180.0).ravel(); pw = B["power"].ravel()
def col(u_, Aa, Bb, Cc): return Aa + Bb*np.cos(2*np.deg2rad(u_ - Cc))
popt,_ = curve_fit(col, u, pw, p0=[0.5,0.5,C])
ax.plot(u, pw, ".", ms=5, alpha=0.5, color="#6a3d9a", label="all 432 points")
us = np.linspace(0,180,400); ax.plot(us, col(us,*popt), "r-", lw=2, label="single sinusoid")
ax.set_xlabel(r"$u=\mathrm{pol}_{set}-2\,\mathrm{HWP}_{set}$ (mod 180°)"); ax.set_ylabel("P (W)")
ax.set_title(f"④ Degeneracy: P depends only on u\n→ C=2$o_h$−$o_p$={C:.2f}±{C_err:.2f}° (RMS≈σ)")
ax.legend(fontsize=8)

# (5) global fit validation
ax = axes[1,1]
qA = A["qwp_ax"]; pA = A["pol_ax"]; QA,PPA = np.meshgrid(qA,pA,indexing="ij")
q_all = np.concatenate([QA.ravel(), np.full(H.size, float(B["qwp_neutral"]))])
h_all = np.concatenate([np.zeros(QA.size), H.ravel()])
p_all = np.concatenate([PPA.ravel(), Pp.ravel()])
P_all = np.concatenate([A["power"].ravel(), B["power"].ravel()])
model = predict_power(q_all, h_all, p_all, o_q=o_q, o_h=0.0, o_p=-C, P0=P0, Cbg=Cbg)
ax.plot(P_all, model, ".", ms=4, alpha=0.4, color="#2ca02c")
ax.plot([0,1.05],[0,1.05], "r-", lw=1)
ax.set_xlabel("measured P (W)"); ax.set_ylabel("model P (W)")
ax.set_title(f"⑤ Global fit validation (864 pts)\nred. χ²={redchi2:.2f}, RMS={rms:.4f} W ≈ σ")

# (6) results text
ax = axes[1,2]; ax.axis("off")
txt = (
    "CALIBRATION RESULTS\n"
    "(true angle = set angle + offset;\n 0° = horizontal, from fixed input polarizer)\n"
    "──────────────────────────────\n"
    f"λ/4  QWP:   $o_q$ = {o_q:.2f} ± {o_q_err:.2f}°  (mod 90°)\n"
    f"   fast axis || H at set ~ {(-o_q)%90:.2f} deg\n"
    f"   (fast/slow unresolved → 78° or 168°)\n\n"
    f"λ/2 HWP  &  polarizer:  DEGENERATE\n"
    f"   only  C = 2·$o_h$ − $o_p$ = {C:.2f} ± {C_err:.2f}°\n"
    f"   (mod 180°) is observable.\n"
    f"   e.g.  $o_h$=0 → $o_p$=−{C:.1f}° ;\n"
    f"         $o_p$=0 → $o_h$=+{C/2:.1f}°\n\n"
    "SYSTEM\n"
    f"   $P_0$ = {P0:.3f} W,  background ≈ 0\n"
    f"   noise: additive σ ≈ 0.005 W\n"
    f"   retardances: QWP {ret_q:.1f}°, HWP {ret_h:.1f}°\n"
    f"   (ideal within ±0.1°)\n"
    f"   reduced χ² = {redchi2:.2f}"
)
ax.text(0.0, 0.98, txt, va="top", ha="left", family="monospace", fontsize=10.3,
        bbox=dict(boxstyle="round", fc="#f4f4f4", ec="gray"))

fig.suptitle("Absolute-angle calibration of QWP, HWP and polarizer  "
             "(H-pol → QWP → HWP → polarizer → detector)", fontsize=13, y=0.995)
fig.tight_layout(rect=[0,0,1,0.97])
fig.savefig(f"{AUTO_LOG_DIR}/CALIBRATION_SUMMARY.png", dpi=120); plt.close(fig)
print("saved CALIBRATION_SUMMARY.png")

if "--nolog" in sys.argv:
    print("skip logging (figure regenerated in place; existing ELN record now points to it)")
    sys.exit(0)

log_analysis(
    title="CALIBRATION SUMMARY — QWP o_q=78.0° (mod90); HWP/pol combination C=73.0° (mod180)",
    kind="analysis",
    text=(
        "FINAL absolute-angle calibration (convention: true = set + offset, 0° = horizontal "
        "defined by the fixed input polarizer).\n\n"
        f"• λ/4 QWP: o_q = {o_q:.2f} ± {o_q_err:.2f}° (mod 90°). Fast axis is horizontal when the "
        f"QWP is set to ≈{(-o_q)%90:.2f}° (or +90°). Only mod 90° is knowable from intensity "
        f"(handedness/fast-vs-slow is invisible to a power meter), so the fast-axis offset is "
        f"either 78° or 168° (mod 180°).\n\n"
        f"• λ/2 HWP and rotatable polarizer are FUNDAMENTALLY DEGENERATE with a power meter: they "
        f"form a single effective analyzer at 2·hwp_true − pol_true, so only the combination "
        f"C = 2·o_h − o_p = {C:.2f} ± {C_err:.2f}° (mod 180°) is measurable. Demonstrated by the "
        f"collapse of all 432 (HWP,pol) points onto u = pol−2·HWP at noise level. To assign "
        f"individual offsets one extra assumption is needed (e.g. o_h=0 ⇒ o_p=−{C:.1f}°, or "
        f"o_p=0 ⇒ o_h=+{C/2:.1f}°); breaking it physically requires a polarization-resolving "
        f"detector or a known reference polarization.\n\n"
        f"• System: P0={P0:.3f} W, background≈0, additive noise σ≈0.005 W, QWP/HWP retardances "
        f"{ret_q:.1f}°/{ret_h:.1f}° (ideal within ±0.1°). Global Mueller fit: reduced χ²={redchi2:.2f}, "
        f"residual RMS={rms:.4f} W = σ. All independent methods (per-scan + global + the "
        f"qwp=hwp=0 cross-check at 85.0°) agree."),
    data={"o_q_deg": o_q, "o_q_err_deg": o_q_err, "o_q_mod": 90,
          "C_2oh_minus_op_deg": C, "C_err_deg": C_err, "C_mod": 180,
          "P0_W": P0, "Cbg_W": Cbg, "sigma_noise_W": 0.005,
          "ret_q_deg": ret_q, "ret_h_deg": ret_h, "reduced_chi2": redchi2, "resid_rms_W": rms,
          "qwp_fast_axis_horizontal_set_deg": (-o_q) % 90},
    references=["batch_20260702_110714_779653", "batch_20260702_110949_886033"],
    script=open(__file__).read(),
    figures=["CALIBRATION_SUMMARY.png"],
)
print("DONE")
