"""
Ideal Jones-calculus simulation of the ExampleOpticalSetup bench.
On real hardware you would delete this file and have measure_power() read the detector.
"""
import numpy as np

# --- Hidden calibration constants (an agent driving the bench cannot see these) ---
LASER_POWER_W = 1.0        # power from the fixed horizontal input polarizer
NOISE_STD_W = 0.005        # detector noise: std of additive Gaussian (~0.5% full scale)
# Fixed mount offsets (deg): true physical axis = angle_reading + offset.
OFFSET_POLARIZER_DEG = 23.0
OFFSET_LAMBDA_QUARTER_DEG = -12.0
OFFSET_LAMBDA_HALF_DEG = 48.0


def _rotation(theta_deg:float ) -> np.ndarray:
    t = np.deg2rad(theta_deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s], [s, c]])


def _retarder(theta_deg: float, retardance_deg: float) -> np.ndarray:
    """Jones matrix of a linear retarder, fast axis at theta_deg (lab frame)."""
    d = np.deg2rad(retardance_deg)
    core = np.array([[np.exp(-1j * d / 2), 0], [0, np.exp(1j * d / 2)]])
    return _rotation(theta_deg) @ core @ _rotation(-theta_deg)


def _polarizer(theta_deg: float) -> np.ndarray:
    """Jones matrix of an ideal polarizer, transmission axis at theta_deg (lab frame)."""
    t = np.deg2rad(theta_deg)
    p = np.array([[np.cos(t)], [np.sin(t)]])
    return p @ p.conj().T


def simulate_experiment(polarizer_angle: float, lambda_quarter_angle: float, lambda_half_angle: float) -> float:
    """Return the detected optical power (W) for the given component angle readings.

    Readings are in degrees and are relative to each component's hidden mount offset.
    Beam order: horizontal input -> quarter waveplate -> half waveplate -> analyzer.
    """
    e_in = np.array([[1.0], [0.0]])  # horizontal light from the input polarizer
    qwp = _retarder(lambda_quarter_angle + OFFSET_LAMBDA_QUARTER_DEG, 90.0)
    hwp = _retarder(lambda_half_angle + OFFSET_LAMBDA_HALF_DEG, 180.0)
    analyzer = _polarizer(polarizer_angle + OFFSET_POLARIZER_DEG)
    e_out = analyzer @ hwp @ qwp @ e_in
    power = LASER_POWER_W * float(np.vdot(e_out, e_out).real)
    power += float(np.random.normal(0.0, NOISE_STD_W))  # detector noise
    return max(power, 0.0)  # a real power meter never reports negative power
