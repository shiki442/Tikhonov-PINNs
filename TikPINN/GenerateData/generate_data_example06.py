import os
from problems.example06 import Example06Problem
from generate_data import generate_data_1d, save_data_pt


def main():
    """Generate data for example 06 with different noise levels using Problem class."""

    # Create problem instance
    problem = Example06Problem()

    # Generate data folder
    folder = '../data/'
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Generate data for different noise levels
    for delta in [0.15, 0.25, 0.35, 0.45]:
        # Generate data using the Problem class
        data = generate_data_1d(problem, delta)

        # Save as .pt file (PyTorch format with dictionary)
        file_name = f'../data/{problem.name}_data{int(100*delta):02d}.pt'
        save_data_pt(data, file_name)
        print(f'{file_name} finished.')

    print("\nAll data generated successfully!")
    print(f"Files saved to: {os.path.abspath(folder)}")


if __name__ == "__main__":
    main()
