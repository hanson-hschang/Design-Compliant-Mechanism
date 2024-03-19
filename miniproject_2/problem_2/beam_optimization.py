"""
Created on Mar. 14, 2024
@author: Heng-Sheng Hanson Chang
"""

import matplotlib.pyplot as plt

from grid import BeamGrid, NodeRange
from design_optimization import TopologyOptimization, VolumeConstraint

def main():
    # Setup grid
    grid = BeamGrid(
        number_of_links=[38, 76],
        length_of_sides=[76, 152],
        youngs_modulus=150_000,
        in_plane_thickness=0.065,
        out_of_plane_thickness=1,
    )
    grid.remove_nodes(x_bounds=[50.5, 76], y_bounds=[76, 152])

    # fig, ax = plt.subplots()
    # ax = grid.plot(
    #     ax,
    #     alpha=0.05,
    #     color='grey',
    # )
    # ax.axis('equal')
    # fig.tight_layout()
    # plt.show()
    # quit()

    # Boundary conditions  
    boundary_constraints = []
    grid.add_conditions(
        boundary_constraints,
        condition='position_x',
        value=0,
        node_range=NodeRange(
            condition='vertical_segement',
            x_value=0,
            y_range=[0, 30]
        )
    )
    grid.add_conditions(
        boundary_constraints,
        condition='position_y',
        value=0, 
        node_range=NodeRange(
            condition='vertical_segement',
            x_value=0,
            y_range=[0, 30]
        )
    )
    grid.add_conditions(
        boundary_constraints,
        condition='angle_theta',
        value=0, 
        node_range=NodeRange(
            condition='vertical_segement',
            x_value=0,
            y_range=[0, 30]
        )
    )
    grid.add_conditions(
        boundary_constraints,
        condition='position_x',
        value=0,
        node_range=NodeRange(
            condition='vertical_segement',
            x_value=76,
            y_range=[0, 76]
        )
    )
    grid.add_conditions(
        boundary_constraints,
        condition='angle_theta',
        value=0,
        node_range=NodeRange(
            condition='vertical_segement',
            x_value=76,
            y_range=[0, 76]
        )
    )

    # External loads
    external_loads=[]
    grid.add_conditions(
        external_loads,
        condition='direction_y',
        value=-100_000,
        node_range=NodeRange(
            condition='point',
            x_value=76,
            y_value=76
        )
    )
    grid.add_conditions(
        external_loads,
        condition='direction_x',
        value=1,
        node_range=NodeRange(
            condition='point',
            x_value=76-25.5,
            y_value=152
        )
    )
    print(external_loads)
    quit()
    plt.show()

if __name__ == "__main__":
    main()
