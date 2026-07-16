# Example agent run — calibrating the optical setup

The `shared_calibration_example/` folder is a **complete, real agent run** captured
from this example setup. It's included so you can see what a session produces before
running your own. The agent drove the simulated optical bench
(`laser → horizontal polarizer → λ/4 → λ/2 → rotatable polarizer → detector`) using
only the two tools it was given — `set_angle(angle, component)` and `measure_power()` —
and was asked to **calibrate the setup from scratch**.

## What the agent did

Starting with no knowledge of the hidden mount offsets, laser power, or noise (all of
which live in `simulation.py` and are invisible to the agent), it:

1. **Probed** the system and characterized the detector noise (additive Gaussian,
   σ ≈ 0.005 W).
2. **Ran 2D angle scans** — QWP×polarizer (scan A) and HWP×polarizer (scan B) — writing
   each measurement to the auto-log.
3. **Fit a global Mueller/Jones model** over 864 points to recover the physical
   constants: the λ/4 offset (mod 90°), the degenerate λ/2 + polarizer combination
   `C = 2·o_h − o_p`, both retardances, and P₀ ≈ 1.00 W.
4. **Validated blind** on a held-out 0–360° range: predicted vs. measured RMS ≈ 0.002 W
   (noise-limited).
5. **Wrote up** the result as a calibration reference with ready-to-use operating points.

The headline finding: with the QWP neutral, detected power follows Malus's law,
`P ≈ P₀·cos²(u − 73.0°)` where `u = pol_set − 2·hwp_set`. Full details are in
[shared_calibration_example/SETUP_CALIBRATION.md](shared_calibration_example/SETUP_CALIBRATION.md).

## Where to find things

- **HTML report of the data/auto-log** (measurements, batches, analyses, embedded
  figures): [shared_calibration_example/auto_log/report_safe_lab_agents.html](https://raw.githack.com/MaxNaeg/safe_lab_agents/main/example_setup/shared_calibration_example/auto_log/report_safe_lab_agents.html) (opens in your browser)
- **HTML report of the conversation** (the full agent transcript):
  [shared_calibration_example/conversation_safe_lab_agents.html](https://raw.githack.com/MaxNaeg/safe_lab_agents/main/example_setup/shared_calibration_example/conversation_safe_lab_agents.html) (opens in your browser)
- **Calibration write-up:** [shared_calibration_example/SETUP_CALIBRATION.md](shared_calibration_example/SETUP_CALIBRATION.md)
- **Analysis scripts the agent wrote:** [shared_calibration_example/scripts/](shared_calibration_example/scripts/)
- **Raw auto-log records** (JSON + HDF5 + `.png` figures + `.eln` export):
  [shared_calibration_example/auto_log/](shared_calibration_example/auto_log/)

Both HTML reports are self-contained — open them directly in a browser.
