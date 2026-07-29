"""
Finite element (P1) + mass-lumping method of lines for blow-up
in u_t = (u^m)_{xx} + u^p on (-L, L) with Dirichlet u(±L,t)=1.

Implements the semidiscrete system (De Pablo–Ferreira–Quirós–Vázquez, 2005, Sec. 10):
    M U'(t) = -A U^m + M U^p,      (11)
with lumped mass matrix M = h I on an equally spaced 1D grid; integrates in time
by explicit Euler with adaptive step. Detects blow-up when max(u) crosses
a large threshold and estimates blow-up time T via a linear fit of max(u)^{-(p-1)}.

Usage (examples):
    python blowup_fem_mol.py
    python blowup_fem_mol.py --m 1.5 --p 2 --N 800 --L 25 --A0 8 --sigma 3
    python blowup_fem_mol.py --Tmax 5 --blow_threshold 1e8

This script produces two figures and prints diagnostics.
"""

from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt


def assemble_tridiagonal(N: int, h: float):
    """Return main and off diagonals for the stiffness matrix K (scaled)
    corresponding to -∂_{xx} with P1 elements on a uniform grid.
    Here we use the standard FE stiffness for the operator -d/dx(d/dx),
    but since the PDE has (u^m)_{xx} we will apply K to V = U^m.
    """
    main_diag = (2.0 / h) * np.ones(N)
    off_diag = (-1.0 / h) * np.ones(N - 1)
    return main_diag, off_diag


def K_dot(v: np.ndarray, main_diag: np.ndarray, off_diag: np.ndarray) -> np.ndarray:
    """Efficient multiplication K @ v for the tri-diagonal stiffness matrix."""
    out = main_diag * v
    out[:-1] += off_diag * v[1:]
    out[1:] += off_diag * v[:-1]
    return out


def run_sim(m=1.5, p=2.0, L=20.0, N=500, bc=1.0, A0=6.0, sigma=3.0,
            Tmax=10.0, blow_threshold=1e6, safety_diff=0.45, safety_reac=0.10,
            dt_cap=0.05, report_K=True, seed=None):
    """
    Execute the explicit time integration with adaptive step.
    Returns a dictionary with results and histories.
    """
    if seed is not None:
        np.random.seed(seed)

    # Spatial grid and initial condition (even, bell-shaped)
    x_interior = np.linspace(-L, L, N + 2)[1:-1]   # interior nodes
    h = x_interior[1] - x_interior[0]
    U = bc + A0 * np.exp(-(x_interior / sigma) ** 2)

    # Stiffness matrix pieces
    main_diag, off_diag = assemble_tridiagonal(N, h)

    # Dirichlet boundary contribution b for V = U^m with u(±L)=bc (constant)
    Vb = bc ** m
    b_vec = np.zeros(N)
    b_vec[0] += (1.0 / h) * Vb
    b_vec[-1] += (1.0 / h) * Vb

    # Time loop
    t = 0.0
    t_hist = [t]
    umax_hist = [U.max()]
    dt_hist = []

    while t < Tmax and U.max() < blow_threshold:
        Umax = U.max()
        # Adaptive dt: diffusion-like constraint and reaction constraint
        Dmax = m * max(Umax, 1.0) ** (m - 1.0)
        dt_diff = safety_diff * h * h / (2.0 * max(Dmax, 1e-14))
        dt_reac = safety_reac / (max(Umax, 1.0) ** (p - 1.0))
        dt = min(dt_diff, dt_reac, dt_cap)

        V = U ** m
        KU = K_dot(V, main_diag, off_diag)
        # With lumped mass M = h I, M^{-1} gives the factors below
        lapVm = -(1.0 / h) * KU + (1.0 / h) * b_vec
        dU = lapVm + U ** p

        U = np.maximum(U + dt * dU, 0.0)  # positivity safeguard
        t += dt

        t_hist.append(t)
        umax_hist.append(U.max())
        dt_hist.append(dt)

    # Estimate blow-up time via linear fit of y = max(u)^{-(p-1)} ~ a t + b -> T = -b/a
    t_arr = np.array(t_hist)
    umax_arr = np.array(umax_hist)
    y = umax_arr ** (-(p - 1.0))
    T_est = None
    if len(t_arr) > 10:
        Kfit = min(200, max(20, len(t_arr) // 4))
        t_fit = t_arr[-Kfit:]
        y_fit = y[-Kfit:]
        A = np.vstack([t_fit, np.ones_like(t_fit)]).T
        coeff, *_ = np.linalg.lstsq(A, y_fit, rcond=None)
        slope, intercept = coeff
        if slope < 0:
            T_est = -intercept / slope

    # Optional: estimate number of adjacent 'large' nodes around the maximum
    K_adjacent = None
    total_nodes_large = None
    if report_K and U.max() > 0:
        threshold_local = U.max() * 1e-3
        center_idx = int(np.argmax(U))
        left = center_idx
        while left > 0 and U[left - 1] > threshold_local:
            left -= 1
        right = center_idx
        while right < len(U) - 1 and U[right + 1] > threshold_local:
            right += 1
        K_adjacent = max(center_idx - left, right - center_idx)
        total_nodes_large = right - left + 1

    results = dict(
        x=x_interior, U=U, t=t, N=N, h=h,
        t_hist=t_arr, umax_hist=umax_arr, dt_hist=np.array(dt_hist),
        T_est=T_est, blow_reached=bool(U.max() >= blow_threshold),
        K_adjacent=K_adjacent, total_nodes_large=total_nodes_large,
        params=dict(m=m, p=p, L=L, N=N, bc=bc, A0=A0, sigma=sigma,
                    Tmax=Tmax, blow_threshold=blow_threshold,
                    safety_diff=safety_diff, safety_reac=safety_reac, dt_cap=dt_cap)
    )
    return results


def main():
    parser = argparse.ArgumentParser(description="P1 FE + mass-lumping MoL for u_t=(u^m)_{xx}+u^p with Dirichlet u=1. Detects blow-up.")
    parser.add_argument("--m", type=float, default=1.5)
    parser.add_argument("--p", type=float, default=2.0)
    parser.add_argument("--L", type=float, default=20.0)
    parser.add_argument("--N", type=int, default=500)
    parser.add_argument("--bc", type=float, default=1.0)
    parser.add_argument("--A0", type=float, default=6.0)
    parser.add_argument("--sigma", type=float, default=3.0)
    parser.add_argument("--Tmax", type=float, default=10.0)
    parser.add_argument("--blow_threshold", type=float, default=1e6)
    parser.add_argument("--safety_diff", type=float, default=0.45)
    parser.add_argument("--safety_reac", type=float, default=0.10)
    parser.add_argument("--dt_cap", type=float, default=0.05)
    parser.add_argument("--no_plots", action="store_true", help="No mostrar gráficos (solo imprimir diagnósticos).")
    args = parser.parse_args()

    res = run_sim(m=args.m, p=args.p, L=args.L, N=args.N, bc=args.bc, A0=args.A0, sigma=args.sigma,
                  Tmax=args.Tmax, blow_threshold=args.blow_threshold, safety_diff=args.safety_diff,
                  safety_reac=args.safety_reac, dt_cap=args.dt_cap, report_K=True)

    x, U = res["x"], res["U"]
    print(f"N={res['N']}, h={res['h']:.6f}, pasos={len(res['dt_hist'])}, t_final={res['t']:.6f}")
    print(f"max(u) al final = {U.max():.3e}")
    if res["blow_reached"]:
        print("Criterio de explosión alcanzado: max(u) superó el umbral.")
    else:
        print("No se alcanzó el umbral de explosión; aumente A0 o Tmax si desea forzarla más claramente.")
    if res["T_est"] is not None:
        print(f"Estimación de tiempo de blow-up T ≈ {res['T_est']:.6f} (ajuste lineal de max(u)^(-(p-1)))")
    if res["total_nodes_large"] is not None:
        print(f"Nodos contiguos 'grandes' alrededor del máximo: {res['total_nodes_large']} (aprox. K={res['K_adjacent']} a cada lado)")

    if not args.no_plots:
        # Gráfica del máximo (semi-log y)
        plt.figure(figsize=(7, 4))
        plt.semilogy(res["t_hist"], res["umax_hist"])
        plt.xlabel("t")
        plt.ylabel("max_x u(x,t)")
        plt.title("Crecimiento del máximo (señal de blow-up)")
        plt.tight_layout()

        # Perfil final
        plt.figure(figsize=(7, 4))
        plt.plot(x, U)
        plt.xlabel("x")
        plt.ylabel("u(x, t_final)")
        plt.title("Perfil espacial cercano al tiempo final")
        plt.tight_layout()

        plt.show()


if __name__ == "__main__":
    main()
