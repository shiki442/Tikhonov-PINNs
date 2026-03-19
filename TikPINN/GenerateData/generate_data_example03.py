"""
Generate 2D data for Gaussian peak problem.

q(x,y) = 1.0 + 4.0 * exp(-40 * ((x-0.6)^2 + (y-0.4)^2))
u(x,y) = 37.0 + 5.0 * sin(π*x) * sin(π*y)

This script generates data for multiple noise levels:
    - noise = 0.00 (0%, exact)
    - noise = 0.01 (1%)
    - noise = 0.05 (5%)
    - noise = 0.10 (10%)
"""

import os
from problems.example03 import Example03Problem
from generate_data import generate_data_nd, save_data_pt


def main():
    # Create problem instance
    problem = Example03Problem()

    # Create output directory
    folder = './data/'
    os.makedirs(folder, exist_ok=True)

    # Noise levels
    noise_levels = [0.00, 0.01, 0.05, 0.10]

    # Data generation parameters
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

        # Create filename: example03_data00.pt, example03_data01.pt, etc.
        noise_str = f"{int(noise_level * 100):02d}"
        file_name = f"example03_data{noise_str}.pt"
        file_path = os.path.join(folder, file_name)

        # Save to file
        save_data_pt(data, file_path)
        print(f"{file_name} finished. (n_bdy={data['n_samples_bdy']})")

    print("-" * 60)
    print("All data generation completed!")


if __name__ == "__main__":
    main()
