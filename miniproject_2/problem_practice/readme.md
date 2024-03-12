# Truss Topology Optimization

This repository contains Python code for performing topology optimization on a truss structure using the finite element method (FEM). The optimization aims to find the optimal distribution of material within a specified volume constraint, minimizing the strain energy of the structure.

## Files

1. `finite_element_model.py`: This file defines the `FEM` abstract base class and the `TrussFEM` class, which implements the finite element method for truss structures. The `TrussFEM` class computes the stiffness matrix, incorporates boundary constraints, and calculates the grid displacement.

2. `design_optimization.py`: This file contains the `VolumeConstraint` class, which enforces the volume constraint during the optimization process, and the `TopologyOptimization` class, which performs the topology optimization by iteratively updating the thickness distribution based on the gradient of the strain energy.

3. `grid.py`: This file defines the `Grid` class, which represents the truss structure. It creates nodes, links, and computes various properties such as length, angle, and Young's modulus for each link. It also provides a method to compute the gradient of the strain energy.

4. `truss_optimization.py`: This file contains the main function that sets up the initial truss structure, boundary conditions, external loads, and runs the topology optimization. It also includes helper functions for plotting the optimized and deformed structures.

## Usage

1. Ensure that you have Python and the required libraries (NumPy and Matplotlib) installed.

2. Run the `truss_optimization.py` script:

```
python truss_optimization.py
```

This will execute the topology optimization process and display the optimized and deformed structures using Matplotlib plots.

## Customization

You can customize the truss structure, boundary conditions, external loads, and optimization parameters by modifying the values in the `main` function of `truss_optimization.py`. The comments in the code provide guidance on how to modify these parameters.

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

