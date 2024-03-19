"""
Created on Mar. 14, 2024
@author: Heng-Sheng Hanson Chang
"""

import matplotlib.pyplot as plt

import sys
sys.path.append('..')
from grid import BeamGrid, NodeRange
from finite_element_model import FEM, OutputDisplacement
from design_optimization import TopologyOptimization, VolumeConstraint
from plot_tools import plot_optimization, plot_deformation

def main():
    # Setup grid
    grid = BeamGrid(
        number_of_links=[6, 6],
        length_of_sides=[100, 100],
        youngs_modulus=169_000,
        in_plane_thickness=5,
        out_of_plane_thickness=1,
    )

    # Boundary conditions  
    boundary_constraints = [
        [0, 'position_x', 0],
        [0, 'position_y', 0],
        [0, 'angle_theta', 0],
    ]
    grid.add_conditions(
        boundary_constraints,
        node_range=NodeRange(
            condition='vertical_segement',
            x_value=100,
            y_range=[0, 100]
        ),
        conditions=['position_x', 'angle_theta'],
        value=0,
    )

    # External loads
    external_loads=[]
    grid.add_conditions(
        external_loads,
        node_range=NodeRange(
            condition='point',
            x_value=100,
            y_value=0
        ),
        conditions='direction_y',
        value=100,
    )

    output_displacement = OutputDisplacement(
        grid=grid,
        position=[100, 100],
        condition='direction_y',
        directional_spring_constant=-100,
    )

    # Set up model
    model = TopologyOptimization(
        fem=FEM(
            grid, 
            boundary_constraints, 
            external_loads,
            output_displacement,
        ),
        volume_constraint=VolumeConstraint(
            total_max_volume=VolumeConstraint.compute_total_volume(
                length_of_links=grid.length_of_links,
                in_plane_thickness=grid.in_plane_thickness,
                out_of_plane_thickness=grid.out_of_plane_thickness,
            ),
            upper_bound=5,
            lower_bound=1e-4, # This cannot be zero. Zero value may cause the invertibility of the stiffness matrix.
            update_ratio=1e3,
            update_power=0.5,
            update_step_max=0.25,
            lagrange_multiplier_setting=dict(
                max=1e5, min=0, tol=1e-4
            )
        ),
        number_of_maximum_iterations=1000,
    )
    
    grid_displacement = model.fem.deform()

    # Create plots
    # plot_deformation(
    #     grid, 
    #     thickness=None,
    #     grid_displacement=grid_displacement
    # )

    # Compute optimization and deformation 
    thickness = model.optimize_geometric_advantage(plot_flag=True)
    grid_displacement = model.fem.deform(thickness)


    # Create plots
    plot_optimization(
        grid, 
        thickness
    )
    plot_deformation(
        grid, 
        thickness=thickness,
        grid_displacement=grid_displacement
    )

    plt.show()

if __name__ == "__main__":
    main()






