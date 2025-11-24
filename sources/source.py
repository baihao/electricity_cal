from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

PEAK_CURRENT = 20000.0
DC_OFFSET = 500.0
RAMP_END = 0.01  # seconds
GRID_FREQ = 50.0  # Hz


def neutral_current(t: np.ndarray | float) -> np.ndarray:
    """
    Piecewise neutral current waveform with 10 ms ramp-up followed by
    steady 50 Hz sinusoid plus 500 A DC offset.
    """
    t_arr = np.atleast_1d(t).astype(float)
    omega = 2 * np.pi * GRID_FREQ
    base = PEAK_CURRENT * np.sin(omega * t_arr - np.pi / 2) + DC_OFFSET
    ramp_factor = np.clip(t_arr / RAMP_END, 0.0, 1.0)
    current = np.where(t_arr < RAMP_END, ramp_factor * base, base)
    return current if isinstance(t, np.ndarray) else current.item()


def plot_neutral_current(t_end: float = 0.2, num: int = 2000, save_path: str | None = None) -> None:
    """
    Plot the neutral current waveform between t=0 and t_end seconds.
    """
    times = np.linspace(0.0, t_end, num=num)
    currents = neutral_current(times)

    plt.figure(figsize=(8, 4))
    plt.plot(times * 1000, currents / 1000)
    plt.xlabel("Time (ms)")
    plt.ylabel("Current (kA)")
    plt.title("Neutral Current Waveform")
    plt.grid(True, linestyle="--", alpha=0.5)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    plot_neutral_current()

