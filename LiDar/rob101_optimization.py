"""
ROB101 Optimization — LiDAR-Camera Extrinsic Calibration
=========================================================
Python equivalent of rob101_optimization.m (Bruce JK Huang, University of Michigan).

Finds the 4×4 rigid-body transform H_LC (rotation R + translation t) that maps
3D LiDAR points onto 2D camera pixels, by minimising the sum of squared
reprojection errors using Lie-group (SO3) parameterisation.

Copyright note: Original algorithm © The Regents of The University of Michigan.
This Python port is for educational use. See original MATLAB file for licence.

Usage
-----
    python rob101_optimization.py

Requirements
------------
    pip install numpy scipy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.io import loadmat
from scipy.optimize import minimize
import os
import sys


# =============================================================================
# 1. Lie-Group Utilities  (SO3 — rotation matrices)
# =============================================================================

def skew(w: np.ndarray) -> np.ndarray:
    """Return the 3×3 skew-symmetric matrix of vector w."""
    return np.array([
        [    0, -w[2],  w[1]],
        [ w[2],     0, -w[0]],
        [-w[1],  w[0],     0],
    ])


def exp_so3(w: np.ndarray) -> np.ndarray:
    """
    Matrix exponential on SO3 (Rodrigues' formula).
    Converts a rotation vector w (axis × angle) to a 3×3 rotation matrix.
    """
    theta = np.linalg.norm(w)
    if theta < 1e-10:
        return np.eye(3)
    K = skew(w / theta)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def log_so3(R: np.ndarray) -> np.ndarray:
    """
    Matrix logarithm on SO3.
    Converts a 3×3 rotation matrix back to a rotation vector (axis × angle).
    """
    cos_theta = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if abs(theta) < 1e-10:
        return np.zeros(3)
    factor = theta / (2 * np.sin(theta))
    return factor * np.array([R[2, 1] - R[1, 2],
                               R[0, 2] - R[2, 0],
                               R[1, 0] - R[0, 1]])


def rpy_to_rotation(roll_deg: float,
                    pitch_deg: float,
                    yaw_deg: float) -> np.ndarray:
    """
    Build a 3×3 rotation matrix from roll-pitch-yaw angles (in degrees).
    Convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    """
    r, p, y = np.deg2rad([roll_deg, pitch_deg, yaw_deg])

    Rx = np.array([[1,       0,        0      ],
                   [0,  np.cos(r), -np.sin(r) ],
                   [0,  np.sin(r),  np.cos(r) ]])

    Ry = np.array([[ np.cos(p), 0, np.sin(p)],
                   [     0,     1,     0     ],
                   [-np.sin(p), 0, np.cos(p)]])

    Rz = np.array([[np.cos(y), -np.sin(y), 0],
                   [np.sin(y),  np.cos(y), 0],
                   [    0,          0,     1]])

    return Rz @ Ry @ Rx


# =============================================================================
# 2. Transform / Projection Utilities
# =============================================================================

def build_H(omega: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Build a 4×4 homogeneous transform from a Lie-algebra vector.

    Parameters
    ----------
    omega : (3,) rotation vector (axis × angle)
    t     : (3,) translation vector in metres

    Returns
    -------
    H : (4, 4) rigid-body transform
    """
    H = np.eye(4)
    H[:3, :3] = exp_so3(omega)
    H[:3,  3] = t
    return H


def build_H_from_rpy_xyz(rpy: list, xyz: list) -> np.ndarray:
    """
    Build a 4×4 transform from RPY degrees + XYZ metres.
    Convenience wrapper used for the initial guess.
    """
    H = np.eye(4)
    H[:3, :3] = rpy_to_rotation(*rpy)
    H[:3,  3] = xyz
    return H


def build_projection_matrix(K: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Compute the 3×4 camera projection matrix P.

        P = K · [I | 0] · H

    Parameters
    ----------
    K : (3, 3) camera intrinsic matrix
    H : (4, 4) extrinsic transform H_LC

    Returns
    -------
    P : (3, 4)
    """
    RT = np.hstack([np.eye(3), np.zeros((3, 1))])  # [I | 0]
    return K @ RT @ H


def project_points(X: np.ndarray, P: np.ndarray) -> np.ndarray:
    """
    Project 3D LiDAR points into the camera image plane.

    Parameters
    ----------
    X : (4, N) homogeneous LiDAR points  [x, y, z, 1]
    P : (3, 4) projection matrix

    Returns
    -------
    uv : (2, N) pixel coordinates  [u (col), v (row)]
    """
    uvw = P @ X            # (3, N)
    return uvw[:2] / uvw[2]  # divide by depth → (2, N)


# =============================================================================
# 3. Cost Function and Optimisation
# =============================================================================

def reprojection_cost(v: np.ndarray,
                      X: np.ndarray,
                      Y: np.ndarray,
                      K: np.ndarray) -> float:
    """
    Sum of squared reprojection errors (½ Σ ‖ Π(Xᵢ; R, t) − Yᵢ ‖²).

    Parameters
    ----------
    v : (6,) optimisation vector  [ω₁, ω₂, ω₃, t₁, t₂, t₃]
    X : (4, N) LiDAR points (homogeneous)
    Y : (3, N) camera pixel observations  [u, v, 1]
    K : (3, 3) camera intrinsic matrix

    Returns
    -------
    cost : scalar float
    """
    H   = build_H(v[:3], v[3:])
    P   = build_projection_matrix(K, H)
    y_hat = project_points(X, P)         # (2, N)  predicted pixels
    diff  = y_hat - Y[:2]                # (2, N)  prediction error
    return 0.5 * np.sum(diff ** 2)


def optimize_lie(X: np.ndarray,
                 Y: np.ndarray,
                 K: np.ndarray,
                 rpy_init: list,
                 xyz_init: list,
                 use_hessian: bool = True) -> dict:
    """
    Minimise reprojection error to find the optimal H_LC.

    Uses L-BFGS-B (quasi-Newton, approximates Hessian) when use_hessian=True,
    or simple gradient descent (Nelder-Mead) when use_hessian=False.
    This mirrors the MATLAB opt.Hessian flag.

    Parameters
    ----------
    X          : (4, N) LiDAR feature points (homogeneous)
    Y          : (3, N) camera feature points  [u, v, 1]
    K          : (3, 3) camera intrinsic matrix
    rpy_init   : [roll, pitch, yaw] in degrees — initial guess
    xyz_init   : [x, y, z] in metres — initial guess
    use_hessian: if True use L-BFGS-B (Hessian method), else Nelder-Mead

    Returns
    -------
    result dict with keys: H_LC, P, cost_init, cost_final, success, v
    """
    # Convert initial RPY+XYZ guess to Lie-algebra vector
    R0     = rpy_to_rotation(*rpy_init)
    omega0 = log_so3(R0)
    v0     = np.concatenate([omega0, xyz_init])

    cost_init = reprojection_cost(v0, X, Y, K)

    if use_hessian:
        method  = 'L-BFGS-B'
        options = {'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-10}
    else:
        method  = 'Nelder-Mead'
        options = {'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8}

    result = minimize(
        reprojection_cost,
        v0,
        args=(X, Y, K),
        method=method,
        options=options,
    )

    H_final = build_H(result.x[:3], result.x[3:])
    P_final = build_projection_matrix(K, H_final)

    return {
        'H_LC'       : H_final,
        'P'          : P_final,
        'v'          : result.x,
        'cost_init'  : cost_init,
        'cost_final' : result.fun,
        'success'    : result.success,
        'message'    : result.message,
    }


# =============================================================================
# 4. Plotting
# =============================================================================

def plot_reprojection(X: np.ndarray,
                      Y: np.ndarray,
                      P_init: np.ndarray,
                      P_final: np.ndarray,
                      title_suffix: str = "") -> None:
    """
    Two-panel plot showing:
      Left  — initial guess projections vs. ground-truth camera corners
      Right — optimised projections vs. ground-truth camera corners
    """
    uv_init  = project_points(X, P_init)
    uv_final = project_points(X, P_final)

    # pixel error statistics
    err_init  = np.linalg.norm(uv_init  - Y[:2], axis=0)
    err_final = np.linalg.norm(uv_final - Y[:2], axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"LiDAR → Camera Reprojection  {title_suffix}", fontsize=13)

    for ax, uv, err, label in [
        (axes[0], uv_init,  err_init,  "Initial guess"),
        (axes[1], uv_final, err_final, "After optimisation"),
    ]:
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_facecolor('#1a1a2e')
        ax.grid(True, color='#333355', linewidth=0.5)

        # Ground-truth camera corners (red)
        ax.scatter(Y[0], Y[1], s=80, c='red',
                   label='Camera corners (ground truth)', zorder=5)

        # Projected LiDAR corners (green)
        ax.scatter(uv[0], uv[1], s=60, c='lime', marker='^',
                   label='Projected LiDAR corners', zorder=5)

        # Error lines
        for i in range(X.shape[1]):
            ax.plot([Y[0, i], uv[0, i]], [Y[1, i], uv[1, i]],
                    'yellow', linewidth=0.8, alpha=0.6)

        ax.set_title(f"{label}\nMean pixel error: {err.mean():.2f} px  "
                     f"(max: {err.max():.2f} px)", fontsize=10)
        ax.set_xlabel("u  (pixel column)")
        ax.set_ylabel("v  (pixel row)")
        ax.legend(fontsize=8, loc='upper right')

    plt.tight_layout()
    plt.savefig("reprojection_result.png", dpi=150, bbox_inches='tight')
    print("  Saved → reprojection_result.png")
    plt.show()


def plot_cost_summary(cost_init: float, cost_final: float) -> None:
    """Bar chart comparing initial vs. final reprojection cost."""
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(['Initial cost', 'Final cost'],
                  [cost_init, cost_final],
                  color=['#e74c3c', '#2ecc71'], width=0.5)
    ax.set_ylabel("Reprojection cost  (½ Σ ‖error‖²)")
    ax.set_title("Optimisation result")
    for bar, val in zip(bars, [cost_init, cost_final]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 0.5,
                f"{val:.2f}", ha='center', va='center',
                fontsize=11, color='white', fontweight='bold')
    plt.tight_layout()
    plt.savefig("cost_summary.png", dpi=150, bbox_inches='tight')
    print("  Saved → cost_summary.png")
    plt.show()


# =============================================================================
# 5. Main
# =============================================================================

def main():
    # ------------------------------------------------------------------
    # Paths  (put your .mat files in the same folder as this script,
    #          or update data_path below)
    # ------------------------------------------------------------------
    data_path   = "."           # folder containing X_train.mat / Y_train.mat
    x_mat_file  = os.path.join(data_path, "X_train.mat")
    y_mat_file  = os.path.join(data_path, "Y_train.mat")

    for f in [x_mat_file, y_mat_file]:
        if not os.path.exists(f):
            print(f"[ERROR] File not found: {f}")
            print("  Place X_train.mat and Y_train.mat next to this script.")
            sys.exit(1)

    # ------------------------------------------------------------------
    # Parameters  (mirrors the MATLAB script)
    # ------------------------------------------------------------------
    USE_HESSIAN = True               # True  → L-BFGS-B  (quasi-Newton)
                                     # False → Nelder-Mead (gradient descent)

    rpy_init = [80.0, 0.0, 90.0]    # roll, pitch, yaw in degrees
    xyz_init = [0.4, -0.15, 0.0]    # x, y, z in metres

    # MATLAB: choisen_indices = [36:48]  (1-indexed, inclusive)
    # Python:                  [35:48]  (0-indexed)
    CHOSEN_START = 35
    CHOSEN_END   = 48               # Python slice end (exclusive)

    # Camera intrinsic matrix (from MATLAB script)
    K = np.array([
        [616.3681640625, 0.0,            319.93463134765625],
        [0.0,            616.7451171875, 243.6385955810547 ],
        [0.0,            0.0,            1.0               ],
    ])

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("Loading data...")
    X_full = loadmat(x_mat_file)['X_train']   # (4, 48)
    Y_full = loadmat(y_mat_file)['Y_train']   # (3, 48)

    # Select the training subset
    X = X_full[:, CHOSEN_START:CHOSEN_END]    # (4, 13)
    Y = Y_full[:, CHOSEN_START:CHOSEN_END]    # (3, 13)

    print(f"  X_train loaded: shape={X_full.shape}  → using cols {CHOSEN_START}–{CHOSEN_END-1}  shape={X.shape}")
    print(f"  Y_train loaded: shape={Y_full.shape}  → using cols {CHOSEN_START}–{CHOSEN_END-1}  shape={Y.shape}")

    # ------------------------------------------------------------------
    # Optimise
    # ------------------------------------------------------------------
    method_name = "Hessian (L-BFGS-B)" if USE_HESSIAN else "Gradient Descent (Nelder-Mead)"
    print(f"\nOptimising using {method_name}...")
    print(f"  Initial guess:  RPY={rpy_init}°   XYZ={xyz_init} m")

    result = optimize_lie(X, Y, K,
                          rpy_init=rpy_init,
                          xyz_init=xyz_init,
                          use_hessian=USE_HESSIAN)

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("OPTIMISATION RESULTS")
    print("=" * 55)
    print(f"  Converged : {result['success']}")
    print(f"  Message   : {result['message']}")
    print(f"  Cost init : {result['cost_init']:.4f}")
    print(f"  Cost final: {result['cost_final']:.6f}")

    print("\nOptimal H_LC  (LiDAR → Camera transform):")
    np.set_printoptions(precision=6, suppress=True)
    print(result['H_LC'])

    R_opt = result['H_LC'][:3, :3]
    t_opt = result['H_LC'][:3, 3]
    print(f"\nRotation matrix R:\n{np.round(R_opt, 6)}")
    print(f"\nTranslation t (metres): {np.round(t_opt, 6)}")

    # Reprojection error per point
    uv_final = project_points(X, result['P'])
    pixel_errors = np.linalg.norm(uv_final - Y[:2], axis=0)
    print(f"\nPer-point reprojection errors (pixels):")
    for i, err in enumerate(pixel_errors):
        print(f"  Point {i+1:2d}: {err:.3f} px")
    print(f"\nMean error : {pixel_errors.mean():.3f} px")
    print(f"Max  error : {pixel_errors.max():.3f} px")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    print("\nGenerating plots...")
    H_init  = build_H_from_rpy_xyz(rpy_init, xyz_init)
    P_init  = build_projection_matrix(K, H_init)

    plot_reprojection(X, Y, P_init, result['P'])
    plot_cost_summary(result['cost_init'], result['cost_final'])

    print("\nAll done!")


if __name__ == "__main__":
    main()
