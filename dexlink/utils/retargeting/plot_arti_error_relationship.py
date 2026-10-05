from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _sincos_l2_from_scaled_delta(delta: np.ndarray, scale: float) -> np.ndarray:
    """Single-frequency sin/cos embedding L2 error for error input (scale * delta)."""
    return 2.0 * np.abs(np.sin(0.5 * scale * delta))


def _mf_sincos_l2_from_delta(
    delta: np.ndarray, num_freqs: int, mode: str = "up"
) -> tuple[np.ndarray, np.ndarray]:
    """Multi-frequency sin/cos encoding distance for powers-of-two frequencies.

    mode="up":   [1, 2, 4, ...]
    mode="down": [1, 1/2, 1/4, ...]
    mode="sym":  [..., 1/4, 1/2, 1, 2, 4, ...] (centered at 1)
    """
    num_freqs = max(1, int(num_freqs))
    if mode == "up":
        freqs = 2.0 ** np.arange(num_freqs, dtype=np.float64)
    elif mode == "down":
        freqs = 2.0 ** (-np.arange(num_freqs, dtype=np.float64))
    elif mode == "sym":
        # For num_freqs=5 -> exponents [-2,-1,0,1,2]
        half = num_freqs // 2
        if num_freqs % 2 == 1:
            exponents = np.arange(-half, half + 1, dtype=np.float64)
        else:
            # For num_freqs=4 -> exponents [-1.5,-0.5,0.5,1.5] (still centered)
            exponents = np.arange(num_freqs, dtype=np.float64) - (num_freqs - 1) / 2.0
        freqs = 2.0**exponents
    else:
        raise ValueError(f"Unknown mode: {mode}")
    mf_pair_l2 = 2.0 * np.abs(np.sin(0.5 * delta[:, None] * freqs[None, :]))
    mf_sincos_l2 = np.sqrt(np.sum(mf_pair_l2**2, axis=1))
    return mf_sincos_l2, freqs


def compute_error_curves(
    delta_min: float = -2 * np.pi,
    delta_max: float = 2 * np.pi,
    num_points: int = 2001,
    num_freqs: int = 1,
) -> dict[str, np.ndarray]:
    """Compare absolute angular error to sinusoidal alternatives.

    Let delta = current_angle - target_angle (radians).

    Returned curves:
    - l1_abs_delta: |delta|
    - sin_abs_delta: |sin(delta)|  (periodic, zero at k*pi)
    - sincos_l2: ||[sin, cos](a) - [sin, cos](b)||_2 = 2|sin(delta/2)|
    - sincos_l1: same embedding but L1 distance
    - mf_sincos_l2: multi-frequency sin/cos encoding L2 distance using powers-of-two frequencies
    - mf_sincos_l2_normed: same, normalized by sqrt(num_freqs)
    """
    delta = np.linspace(delta_min, delta_max, num_points, dtype=np.float64)

    l1_abs_delta = np.abs(delta)
    sin_abs_delta = np.abs(np.sin(delta))

    # WLOG set target angle to 0 and current angle to delta.
    sin_cur = np.sin(delta)
    cos_cur = np.cos(delta)
    sin_tgt = 0.0
    cos_tgt = 1.0

    sincos_l2 = np.sqrt((sin_cur - sin_tgt) ** 2 + (cos_cur - cos_tgt) ** 2)
    sincos_l1 = np.abs(sin_cur - sin_tgt) + np.abs(cos_cur - cos_tgt)

    # Closed-form for verification/reference: 2|sin(delta/2)|
    sincos_l2_closed = 2.0 * np.abs(np.sin(delta / 2.0))

    mf_sincos_l2, freqs = _mf_sincos_l2_from_delta(
        delta, num_freqs=num_freqs, mode="up"
    )
    mf_sincos_l2_normed = mf_sincos_l2 / np.sqrt(num_freqs)
    mf_down_sincos_l2, freqs_down = _mf_sincos_l2_from_delta(
        delta, num_freqs=num_freqs, mode="down"
    )
    mf_down_sincos_l2_normed = mf_down_sincos_l2 / np.sqrt(num_freqs)
    mf_sym_sincos_l2, freqs_sym = _mf_sincos_l2_from_delta(
        delta, num_freqs=num_freqs, mode="sym"
    )
    mf_sym_sincos_l2_normed = mf_sym_sincos_l2 / np.sqrt(num_freqs)

    return {
        "delta": delta,
        "l1_abs_delta": l1_abs_delta,
        "sin_abs_delta": sin_abs_delta,
        "sincos_l2": sincos_l2,
        "sincos_l1": sincos_l1,
        "sincos_l2_closed": sincos_l2_closed,
        "mf_sincos_l2": mf_sincos_l2,
        "mf_sincos_l2_normed": mf_sincos_l2_normed,
        "freqs": freqs,
        "mf_down_sincos_l2": mf_down_sincos_l2,
        "mf_down_sincos_l2_normed": mf_down_sincos_l2_normed,
        "freqs_down": freqs_down,
        "mf_sym_sincos_l2": mf_sym_sincos_l2,
        "mf_sym_sincos_l2_normed": mf_sym_sincos_l2_normed,
        "freqs_sym": freqs_sym,
    }


def plot_error_relationship(
    save_path: str | Path | None = None,
    delta_min: float = -2 * np.pi,
    delta_max: float = 2 * np.pi,
    num_points: int = 2001,
    num_freqs: int = 1,
    show: bool = True,
):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required for plotting") from exc

    curves = compute_error_curves(
        delta_min=delta_min,
        delta_max=delta_max,
        num_points=num_points,
        num_freqs=num_freqs,
    )
    delta = curves["delta"]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax = axes[0]
    ax.plot(delta, curves["l1_abs_delta"], label=r"$|\Delta|$ (L1 angle error)")
    ax.plot(delta, curves["sin_abs_delta"], label=r"$|\sin(\Delta)|$")
    ax.plot(delta, curves["sincos_l2"], label=r"$\|[\sin,\cos]-[\sin,\cos]\|_2$")
    ax.plot(
        delta,
        curves["sincos_l2_closed"],
        "--",
        linewidth=1.2,
        label=r"$2|\sin(\Delta/2)|$ (closed form)",
    )

    # Multi-frequency curves for requested K and common comparisons K=4,8.
    k_list = []
    for k in (num_freqs, 4):
        if k not in k_list and k >= 1:
            k_list.append(int(k))
    for k in k_list:
        mf_curve_up, _ = _mf_sincos_l2_from_delta(delta, num_freqs=k, mode="up")
        mf_curve_down, _ = _mf_sincos_l2_from_delta(delta, num_freqs=k, mode="down")
        mf_curve_sym, _ = _mf_sincos_l2_from_delta(delta, num_freqs=k, mode="sym")
        ax.plot(
            delta,
            mf_curve_up / np.sqrt(k),
            linewidth=1.4,
            alpha=0.9,
            label=rf"MF up (1,2,4,...) normed, $K={k}$",
        )
        ax.plot(
            delta,
            mf_curve_down / np.sqrt(k),
            linewidth=1.4,
            alpha=0.9,
            linestyle="--",
            label=rf"MF down (1,1/2,1/4,...) normed, $K={k}$",
        )
        ax.plot(
            delta,
            mf_curve_sym / np.sqrt(k),
            linewidth=1.4,
            alpha=0.9,
            linestyle=":",
            label=rf"MF sym (centered at 1) normed, $K={k}$",
        )

    ax.set_ylabel("Error value")
    ax.set_title("Articulation Error: L1 vs Sinusoidal Projections")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

    # Ratio plot (only where denominator is not too small).
    ax = axes[1]
    eps = 1e-8
    valid = curves["l1_abs_delta"] > eps
    ratio_sin = np.full_like(delta, np.nan)
    ratio_sincos = np.full_like(delta, np.nan)
    ratio_sin[valid] = curves["sin_abs_delta"][valid] / curves["l1_abs_delta"][valid]
    ratio_sincos[valid] = curves["sincos_l2"][valid] / curves["l1_abs_delta"][valid]
    ax.plot(delta, ratio_sin, label=r"$|\sin(\Delta)| / |\Delta|$")
    ax.plot(delta, ratio_sincos, label=r"$2|\sin(\Delta/2)| / |\Delta|$")

    for k in k_list:
        mf_curve_up, _ = _mf_sincos_l2_from_delta(delta, num_freqs=k, mode="up")
        mf_curve_down, _ = _mf_sincos_l2_from_delta(delta, num_freqs=k, mode="down")
        mf_curve_sym, _ = _mf_sincos_l2_from_delta(delta, num_freqs=k, mode="sym")
        ratio_mf_up_k = np.full_like(delta, np.nan)
        ratio_mf_down_k = np.full_like(delta, np.nan)
        ratio_mf_sym_k = np.full_like(delta, np.nan)
        ratio_mf_up_k[valid] = (mf_curve_up[valid] / np.sqrt(k)) / curves[
            "l1_abs_delta"
        ][valid]
        ratio_mf_down_k[valid] = (mf_curve_down[valid] / np.sqrt(k)) / curves[
            "l1_abs_delta"
        ][valid]
        ratio_mf_sym_k[valid] = (mf_curve_sym[valid] / np.sqrt(k)) / curves[
            "l1_abs_delta"
        ][valid]
        ax.plot(
            delta,
            ratio_mf_up_k,
            linewidth=1.2,
            alpha=0.9,
            label=rf"MF up / $|\Delta|$ (K={k})",
        )
        ax.plot(
            delta,
            ratio_mf_down_k,
            linewidth=1.2,
            alpha=0.9,
            linestyle="--",
            label=rf"MF down / $|\Delta|$ (K={k})",
        )
        ax.plot(
            delta,
            ratio_mf_sym_k,
            linewidth=1.2,
            alpha=0.9,
            linestyle=":",
            label=rf"MF sym / $|\Delta|$ (K={k})",
        )
    ax.axhline(1.0, color="k", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_xlabel(r"Angular error $\Delta$ (radians)")
    ax.set_ylabel("Ratio to L1")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

    fig.tight_layout(rect=(0.0, 0.0, 0.78, 1.0))

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return curves


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot relationship between articulation L1 error and sinusoidal errors."
    )
    parser.add_argument(
        "--delta-min",
        type=float,
        default=float(-2 * np.pi),
        help="Minimum angular error in radians (default: -2*pi).",
    )
    parser.add_argument(
        "--delta-max",
        type=float,
        default=float(2 * np.pi),
        help="Maximum angular error in radians (default: 2*pi).",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=2001,
        help="Number of samples across the range.",
    )
    parser.add_argument(
        "--num-freqs",
        type=int,
        default=1,
        help="Number of powers-of-two frequencies for sin/cos encoding (1 => [1]).",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Optional path to save figure (e.g., /tmp/arti_error.png).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive window; useful on headless machines.",
    )
    return parser


def main():
    args = _build_argparser().parse_args()
    plot_error_relationship(
        save_path=args.save_path,
        delta_min=args.delta_min,
        delta_max=args.delta_max,
        num_points=args.num_points,
        num_freqs=args.num_freqs,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
