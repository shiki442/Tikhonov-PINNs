"""
Compute relative L2 error for q and u in gaussian_peak and non_smooth_radial experiments.
"""

import numpy as np
import pandas as pd
import os


def q_dagger_gaussian(x, y):
    """Gaussian peak: 1.0 + 2.0*exp(-5.0*((x-0.8)^2 + (y-0.2)^2))"""
    return 1.0 + 2.0 * np.exp(-5.0 * ((x - 0.8)**2 + (y - 0.2)**2))


def u_dagger_gaussian(x, y):
    """u_exact = 1.0 + 2.0*sin(pi*x)*sin(pi*y)"""
    return 1.0 + 2.0 * np.sin(np.pi * x) * np.sin(np.pi * y)


def q_dagger_non_smooth(x, y):
    """Non-smooth radial function based on distance from (0.4, 0.4)"""
    r = np.sqrt((x - 0.4)**2 + (y - 0.4)**2)
    r_inner = 0.15
    r_outer = 0.2

    q = np.ones_like(r) * 1.0  # outer region: 1.0
    mask_inner = r <= r_inner
    mask_middle = (r > r_inner) & (r < r_outer)

    q[mask_inner] = 5.0
    q[mask_middle] = 5.0 + (1.0 - 5.0) * (r[mask_middle] - r_inner) / (r_outer - r_inner)

    return q


def u_dagger_non_smooth(x, y):
    """u_exact = 4*x*(1-x)*y*(1-y)"""
    return 4 * x * (1 - x) * y * (1 - y)


def compute_relative_error(u_pred, u_true):
    """Compute relative L2 error."""
    numerator = np.sqrt(np.mean((u_pred - u_true)**2))
    denominator = np.sqrt(np.mean(u_true**2))
    return numerator / denominator


def load_and_compute_error(experiment_name, str_param, q_dagger_func, u_dagger_func):
    """Load result file and compute relative errors."""
    file_path = f'./output/{experiment_name}/result_{str_param}/result.txt'

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    results = pd.read_csv(file_path, header=None, sep=',').values

    x = results[:, 0]
    y = results[:, 1]
    q_nn = results[:, 2]
    u_nn = results[:, 3]

    # Compute true solutions
    q_true = q_dagger_func(x, y)
    u_true = u_dagger_func(x, y)

    # Compute relative errors
    rel_err_q = compute_relative_error(q_nn, q_true)
    rel_err_u = compute_relative_error(u_nn, u_true)

    return {
        'lambda': str_param.split('_')[0],
        'noise': str_param.split('_')[1],
        'rel_err_q': rel_err_q,
        'rel_err_u': rel_err_u
    }


def main():
    print("=" * 70)
    print("Relative L2 Error Analysis")
    print("=" * 70)

    # Gaussian peak experiment
    print("\n" + "-" * 70)
    print("Experiment: Gaussian Peak")
    print("-" * 70)
    print(f"{'Noise Level':<15} {'Rel Err q':<20} {'Rel Err u':<20}")
    print("-" * 70)

    gaussian_params = ['09_00', '09_01', '09_10', '09_50']
    gaussian_results = []

    for str_param in gaussian_params:
        result = load_and_compute_error('gaussian_peak', str_param,
                                         q_dagger_gaussian, u_dagger_gaussian)
        if result:
            gaussian_results.append(result)
            noise_pct = int(result['noise'])
            print(f"{noise_pct:>3}%{'':<10} {result['rel_err_q']:<20.6e} {result['rel_err_u']:<20.6e}")

    # Non-smooth radial experiment
    print("\n" + "-" * 70)
    print("Experiment: Non-smooth Radial")
    print("-" * 70)
    print(f"{'Noise Level':<15} {'Rel Err q':<20} {'Rel Err u':<20}")
    print("-" * 70)

    non_smooth_params = ['09_01', '09_10']
    non_smooth_results = []

    for str_param in non_smooth_params:
        result = load_and_compute_error('non_smooth_radial', str_param,
                                         q_dagger_non_smooth, u_dagger_non_smooth)
        if result:
            non_smooth_results.append(result)
            noise_pct = int(result['noise'])
            print(f"{noise_pct:>3}%{'':<10} {result['rel_err_q']:<20.6e} {result['rel_err_u']:<20.6e}")

    # Summary table
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    print("\nGaussian Peak:")
    for r in gaussian_results:
        noise = int(r['noise'])
        print(f"  δ={noise:>2}%: rel_err(q) = {r['rel_err_q']:.4e}, rel_err(u) = {r['rel_err_u']:.4e}")

    print("\nNon-smooth Radial:")
    for r in non_smooth_results:
        noise = int(r['noise'])
        print(f"  δ={noise:>2}%: rel_err(q) = {r['rel_err_q']:.4e}, rel_err(u) = {r['rel_err_u']:.4e}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
