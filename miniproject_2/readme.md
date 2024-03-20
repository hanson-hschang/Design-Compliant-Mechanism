# Topology Optimization

This repository contains Python code for performing topology optimization on a truss or beam structure using the finite element method (FEM). The optimization aims to find the optimal distribution of material within a specified volume constraint, minimizing the strain energy of the structure or the geometric advantage.

## Files

1. `finite_element_model.py`: This file defines the `FEM` class, which implements the finite element method for both the truss and beam structures. The `FEM` class computes the stiffness matrix, incorporates boundary constraints, and calculates the grid displacement.

2. `design_optimization.py`: This file contains the `VolumeConstraint` class, which enforces the volume constraint during the optimization process, and the `TopologyOptimization` class, which performs the topology optimization by iteratively updating the thickness distribution based on the gradient of the strain energy.

3. `grid.py`: This file defines the `Grid` base abstract class with the `TrussGrid` and `BeamGrid`, which represents the truss and beam structures, respectively. This classes create nodes, links, and computes various properties such as length, angle, and Young's modulus for each link. It also provides a method to compute the strain energy and the gradient of it.

4. `plot_tools.py`: This file contains helper functions for plotting the optimized and deformed structures.

## Usage

1. Ensure that you have Python and the required libraries (NumPy and Matplotlib) installed.

2. Run the `optimization_*.py` script, for example:

```
python optimization_truss.py
```

This will execute the topology optimization process and display the optimized and deformed structures using Matplotlib plots.

## Customization

You can customize the truss structure, boundary conditions, external loads, and optimization parameters by modifying the values in the `main` function of `optimization_truss.py`. The comments in the code provide guidance on how to modify these parameters.

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

