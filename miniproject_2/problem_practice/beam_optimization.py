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
        number_of_links=[6, 6],
        length_of_sides=[100, 100],
        youngs_modulus=150_000,
        in_plane_thickness=5,
        out_of_plane_thickness=1,
    )

    # Boundary conditions  
    boundary_constraints = [
        [0, 'position_x', 0],
        [0, 'position_y', 0],
        [0, 'angle_theta', 0],
    ]
    for i in range(7, len(grid.nodes), 7):
        boundary_constraints.append([i, 'position_x', 0])
        boundary_constraints.append([i, 'angle_theta', 0])

    # External loads
    external_loads=[]
    grid.add_conditions(
        external_loads,
        condition='direction_y',
        value=100_000,
        node_range=NodeRange(
            condition='point',
            x_value=100,
            y_value=0
        )
    )
    grid.add_conditions(
        external_loads,
        condition='direction_y',
        value=-1,
        node_range=NodeRange(
            condition='point',
            x_value=100,
            y_value=100
        )
    )

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

if __name__ == "__main__":
    main()






