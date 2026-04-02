"""
Generate 2D data for continuous non-smooth inclusion problem (example05).

q(x,y) has a linear transition layer:
  - q = 5.0 inside inner circle (r <= 0.15)
  - q transitions linearly from 5.0 to 1.0 in annulus (0.15 < r < 0.20)
  - q = 1.0 outside outer circle (r >= 0.20)

u(x,y) = 4*x*(1-x)*y*(1-y)

This script generates data for multiple noise levels:
    - noise = 0.00 (0%, exact)
    - noise = 0.01 (1%)
    - noise = 0.10 (10%)
    - noise = 0.50 (50%)
"""

import os
from problems.example05 import Example05Problem
from generate_data import generate_data_nd, save_data_pt


def main():
    # Create problem instance
    problem = Example05Problem()

    # Create output directory
    folder = './data/'
    os.makedirs(folder, exist_ok=True)

    # Noise levels
    noise_levels = [0.00, 0.01, 0.10, 0.50]

    # Data generation Parameters
    n_samples_int = 50000  # Interior samples
    n_samples_per_face = 5000  # Samples per boundary face

    print(f"Generating data for: {problem.name}")
    print(f"Dimension: {problem.dim}")
    print(f"Noise levels: {noise_levels}")
    print(f"Interior samples: {n_samples_int}")
    print(f"Boundary samples per face: {n_samples_per_face}")
    print("-" * 60)

    for noise_level in noise_levels:
        # Generate data
        data = generate_data_nd(
            problem=problem, noise_level=noise_level, n_samples_int=n_samples_int, n_samples_per_face=n_samples_per_face
        )

        # Create filename: example05_data00.pt, example05_data01.pt, etc.
        noise_str = f"{int(noise_level * 100):02d}"
        file_name = f"example05_data{noise_str}.pt"
        file_path = os.path.join(folder, file_name)

        # Save to file
        save_data_pt(data, file_path)
        print(f"{file_name} finished. (n_bdy={data['n_samples_bdy']})")

    print("-" * 60)
    print("All data generation completed!")


if __name__ == "__main__":
    main()
