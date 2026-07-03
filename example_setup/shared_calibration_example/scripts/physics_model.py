"""Forward Mueller model of the setup and the identifiability analysis.

Order:  Laser -> H-polarizer(fixed) -> QWP -> HWP -> Polarizer -> Detector.
Input Stokes after the fixed horizontal polarizer: S=(1,1,0,0) (horizontal ref).
Each rotatable element has true angle = set-angle + offset (deg):
    qwp_true = q_set + o_q ,  hwp_true = h_set + o_h ,  pol_true = p_set + o_p.
Detector: P = C_bg + P0_frac * (analyzer-projected Stokes intensity),
we parametrize overall scale P0 and background C_bg empirically.

This module provides:
  * stokes_after_retarder(S, delta_deg, theta_true_deg)  general retarder
  * predict_power(q_set,h_set,p_set, o_q,o_h,o_p, P0, Cbg)  full forward model
  * self-tests proving the identifiability structure, logged to the ELN.
"""
import numpy as np


def retarder_mueller(delta_deg, theta_deg):
    """Mueller matrix (on S1,S2,S3) of a linear retarder, retardance delta,
    fast axis at theta (deg). Returns 4x4 including S0."""
    d = np.deg2rad(delta_deg); c2 = np.cos(2*np.deg2rad(theta_deg)); s2 = np.sin(2*np.deg2rad(theta_deg))
    cd, sd = np.cos(d), np.sin(d)
    M = np.array([
        [1, 0, 0, 0],
        [0, c2*c2 + s2*s2*cd,      c2*s2*(1-cd),        -s2*sd],
        [0, c2*s2*(1-cd),          s2*s2 + c2*c2*cd,     c2*sd],
        [0, s2*sd,                -c2*sd,                cd   ],
    ], dtype=float)
    return M


def polarizer_transmission(S, p_true_deg):
    """Intensity transmitted by an ideal linear polarizer at angle p_true (deg)."""
    a = 2*np.deg2rad(p_true_deg)
    return 0.5*(S[0] + S[1]*np.cos(a) + S[2]*np.sin(a))


def predict_power(q_set, h_set, p_set, o_q=0.0, o_h=0.0, o_p=0.0,
                  P0=1.0, Cbg=0.0, ret_q=90.0, ret_h=180.0):
    """Full forward model -> measured power. Scalars or broadcastable arrays."""
    q_set, h_set, p_set = np.broadcast_arrays(np.asarray(q_set, float),
                                              np.asarray(h_set, float),
                                              np.asarray(p_set, float))
    out = np.empty(q_set.shape)
    for idx in np.ndindex(q_set.shape):
        S = np.array([1.0, 1.0, 0.0, 0.0])                     # horizontal
        S = retarder_mueller(ret_q, q_set[idx] + o_q) @ S       # QWP
        S = retarder_mueller(ret_h, h_set[idx] + o_h) @ S       # HWP
        I = polarizer_transmission(S, p_set[idx] + o_p)         # analyzer
        out[idx] = Cbg + P0*I
    return out if out.shape else float(out)


# ---------------------------------------------------------------- self-test
if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/agent/shared/scripts")
    from auto_log_client import log_analysis

    rng_p = np.arange(0, 180, 0.5)   # FINE grid so (max-min)/2 == continuous amplitude

    # (A) amplitude of a polarizer sweep = (P0/2)|cos 2 q_true|, independent of h,o_p
    def sweep_amp(o_q, o_h, o_p, qset, hset):
        pw = predict_power(qset, hset, rng_p, o_q, o_h, o_p, P0=0.9, Cbg=0.04)
        return 0.5*(pw.max()-pw.min())
    q_true_vals = np.arange(0, 180, 5.0)
    amp = np.array([sweep_amp(0, 33.0, 17.0, q, 0.0) for q in q_true_vals])  # o_q=0 -> q_true=q
    pred = 0.5*0.9*np.abs(np.cos(np.deg2rad(2*q_true_vals)))
    testA = np.allclose(amp, pred, atol=1e-9)

    # amplitude independent of hwp setting and o_p:
    amp_h0 = np.array([sweep_amp(0, 0.0,  0.0,  q, 0.0)  for q in q_true_vals])
    amp_h1 = np.array([sweep_amp(0, 40.0, 25.0, q, 70.0) for q in q_true_vals])
    testA2 = np.allclose(amp_h0, amp_h1, atol=1e-9)

    # (B) DEGENERACY: (o_h, o_p) -> (o_h+D, o_p+2D) leaves ALL powers invariant
    grid_q = np.arange(0,180,15.0); grid_h=np.arange(0,180,15.0); grid_p=np.arange(0,180,15.0)
    Q,H,Pp = np.meshgrid(grid_q, grid_h, grid_p, indexing="ij")
    base = predict_power(Q,H,Pp, o_q=10, o_h=33, o_p=17, P0=0.9, Cbg=0.04)
    D = 27.3
    shifted = predict_power(Q,H,Pp, o_q=10, o_h=33+D, o_p=17+2*D, P0=0.9, Cbg=0.04)
    testB = np.allclose(base, shifted, atol=1e-9)
    maxdiff_B = float(np.max(np.abs(base-shifted)))

    # control: an independent change of o_p alone DOES change power (so C=2o_h-o_p is real signal)
    ctrl = predict_power(Q,H,Pp, o_q=10, o_h=33, o_p=17+15, P0=0.9, Cbg=0.04)
    testB_ctrl = not np.allclose(base, ctrl, atol=1e-3)

    # (C) QWP fast/slow: q_true and q_true+90 give identical power (mod-90 ambiguity)
    c1 = predict_power(Q,H,Pp, o_q=10,    o_h=33, o_p=17, P0=0.9, Cbg=0.04)
    c2 = predict_power(Q,H,Pp, o_q=10+90, o_h=33, o_p=17, P0=0.9, Cbg=0.04)
    testC = np.allclose(c1, c2, atol=1e-9)

    print("A  amp = (P0/2)|cos2q_true|            :", testA)
    print("A2 amp independent of hwp,o_p          :", testA2)
    print("B  (o_h,o_p)->(o_h+D,o_p+2D) invariant :", testB, " maxdiff=%.2e"%maxdiff_B)
    print("B' o_p alone DOES change power          :", testB_ctrl)
    print("C  q_true vs q_true+90 identical        :", testC)

    log_analysis(
        title="Physics model + identifiability: what a power meter can and cannot calibrate",
        kind="decision",
        text=(
            "Built and unit-tested the full Mueller forward model "
            "(H-pol -> QWP -> HWP -> analyzer, horizontal input S=(1,1,0,0)). "
            "Verified analytically-predicted structure numerically:\n\n"
            "(A) A polarizer sweep has amplitude (P0/2)|cos(2·qwp_true)|, INDEPENDENT "
            "of the HWP setting and the polarizer offset. => the QWP offset o_q can be "
            "read off cleanly from the sweep-amplitude vs qwp_set, decoupled from the "
            "other two offsets. Peaks (|cos2q|=1) mark qwp_true=0 mod 90.\n\n"
            "(B) FUNDAMENTAL DEGENERACY: the transform (o_h,o_p) -> (o_h+Δ, o_p+2Δ) "
            "leaves every possible power reading invariant (verified over a 3-D grid, "
            "max diff %.1e). Physically, HWP(h)+Polarizer(p) act as a single effective "
            "analyzer at angle A_eff = 2·hwp_true − pol_true; only the combination "
            "C = 2·o_h − o_p is observable with intensity measurements. o_h and o_p "
            "CANNOT be separated individually by a power meter in this element order. "
            "Control: changing o_p alone (breaking the pairing) does change the power, "
            "so C is a real, measurable quantity.\n\n"
            "(C) qwp_true and qwp_true+90 give identical power (the created circular "
            "component S3 is invisible downstream), so o_q is only determined mod 90° "
            "(fast/slow axis unresolved by intensity alone).\n\n"
            "PLAN: (1) Scan A qwp×pol (hwp fixed) -> o_q from amplitude(qwp). "
            "(2) Scan B hwp×pol (qwp neutral) -> C=2o_h−o_p from peak(hwp), slope +2, "
            "and demonstrate the degeneracy by collapsing power onto (2·hwp−pol). "
            "Report o_q (mod 90) and C (mod 180); state the o_h/o_p gauge freedom."
            % maxdiff_B),
        data={"amp_pred_match": bool(testA), "amp_indep_h": bool(testA2),
              "degeneracy_invariant": bool(testB), "op_alone_changes": bool(testB_ctrl),
              "qwp_mod90": bool(testC), "maxdiff_degeneracy": maxdiff_B},
        references=["batch_20260702_110113_319978"],
        script=open(__file__).read(),
    )
    print("DONE (logged)")
