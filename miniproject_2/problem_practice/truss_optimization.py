"""
Created on Mar. 01, 2024
@author: Heng-Sheng Hanson Chang
"""

import matplotlib.pyplot as plt

from grid import TrussGrid
from finite_element_model import TrussFEM
from design_optimization import TopologyOptimization, VolumeConstraint

def plot_optimization(grid: TrussGrid, thickness):
    fig, ax = plt.subplots()
    ax = grid.plot(
        ax,
        alpha=0.05,
        color='grey',
    )
    ax.plot(
        [],[], 
        alpha=0.2,
        color='grey',
        label='initial'
    )
    ax = grid.plot(
        ax,
        in_plane_thickness=thickness, 
        color='grey'
    )
    ax.plot(
        [],[], 
        color='grey',
        label='optimized'
    )
    ax.axis('equal')
    ax.legend()
    fig.tight_layout()
    
def plot_deformation(grid: TrussGrid, thickness, grid_displacement):
    fig, ax = plt.subplots()
    ax = grid.plot(
        ax,
        in_plane_thickness=thickness, 
        color='grey',
    )
    ax.plot(
        [],[], 
        color='grey',
        label='optimized',
    )
    ax = grid.plot(
        ax,
        grid_displacement=grid_displacement,
        in_plane_thickness=thickness, 
        color='black',
    )
    ax.plot(
        [],[], 
        color='black',
        label='optimized - deformed',
    )
    ax.axis('equal')
    ax.legend()
    fig.tight_layout()

def main():
   # Setup grid
    grid = TrussGrid(
        number_of_links=[6, 6],
        length_of_sides=[100, 100],
        youngs_modulus=150_000,
        in_plane_thickness=0.065,
        out_of_plane_thickness=1,
    )

    # Boundary conditions  
    boundary_constraints = []
    for i in range(7):
        boundary_constraints.append([i, 'position_x', 0])
        boundary_constraints.append([i, 'position_y', 0])
    for i in range(7, len(grid.nodes), 7):
        boundary_constraints.append([i, 'position_x', 0])
    for i in range(13, len(grid.nodes), 7):
        boundary_constraints.append([i, 'position_x', 0])

    # External loads
    external_loads=[
        [45, 'direction_y', -100_000],
    ]

    # Set up model
    model = TopologyOptimization(
        fem=TrussFEM(
            grid, 
            boundary_constraints, 
            external_loads
        ),
        volume_constraint=VolumeConstraint(
            total_max_volume=VolumeConstraint.compute_total_volume(
                length_of_links=grid.length_of_links,
                in_plane_thickness=grid.in_plane_thickness,
                out_of_plane_thickness=grid.out_of_plane_thickness,
            ),
            upper_bound=10,
            lower_bound=1e-4, # This cannot be zero. Zero value may cause the invertibility of the stiffness matrix.
            update_ratio=1e3,
            update_power=0.5,
            lagrange_multiplier_setting=dict(
                max=1e5, min=0, tol=1e-4
            )
        ),
        number_of_maximum_iterations=20,
    )

    # Compute optimization and deformation 
    thickness = model.optimize(plot_flag=True)
    grid_displacement = model.fem.deform(thickness)

    # Create plots
    plot_optimization(grid, thickness)
    plot_deformation(grid, thickness, grid_displacement)

    plt.show()

if __name__ == "__main__":
    main()
