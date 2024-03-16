"""
Created on Mar. 09, 2024
@author: Heng-Sheng Hanson Chang
"""

from typing import Optional
from abc import ABC, abstractmethod
from collections import defaultdict
import numpy as np

from grid import NodeRange, Grid

DOF_DICT = {
    'position_x': 0,
    'position_y': 1,
    'angle_theta': 2,
}

DIRECTION_DICT = {
    'direction_x': 0,
    'direction_y': 1,
    'direction_theta': 2,
}

class OutputDisplacement:
    def __init__(self, position: list, condition: str, spring_constant: float):
        self.position = np.array(position)
        self.condition = condition
        self.spring_constant = spring_constant

    def update_property(self, grid: Grid):
        self.node_index = NodeRange.find_nearest_node(grid.nodes, self.position)
        total_degree_of_freedom = grid.degree_of_freedom * len(grid.nodes)
        self.stiffness_matrix = np.zeros(
            (total_degree_of_freedom, total_degree_of_freedom)
        )
        index = grid.degree_of_freedom * self.node_index + DIRECTION_DICT[self.condition]
        self.stiffness_matrix[index, index] = self.spring_constant        

class FEM(ABC):
    def __init__(
        self, 
        grid: Grid, 
        boundary_constraints: list,
        external_loads: list,
        output_displacement: Optional[OutputDisplacement] = None
    ):
        self.grid = grid
        number_of_nodes = len(self.grid.nodes)
        degree_of_freedom = grid.degree_of_freedom
        
        self.vectorized_external_loads = np.zeros(
            degree_of_freedom*number_of_nodes
        )
        for external_load in external_loads:
            index, condition, value = external_load
            self.vectorized_external_loads[
                degree_of_freedom*index+DIRECTION_DICT[condition]
            ] = value

        boundary_constraints = sorted(
            boundary_constraints, 
            key= lambda boundary_constraint: boundary_constraint[0],
            reverse=True
        )

        self.sorted_boundary_constraints = defaultdict(dict)
        for boundary_constraint in boundary_constraints:
            index, condition, value = boundary_constraint
            self.sorted_boundary_constraints[index][condition] = value
        for index, boundary_constraint_conditions in self.sorted_boundary_constraints.items():
            conditions = set(boundary_constraint_conditions.keys())
            removed_entries_list = []
            for condition in conditions:
                removed_entries_list.append(
                    degree_of_freedom*index+DOF_DICT[condition]
                )
            self.vectorized_external_loads = np.delete(
                self.vectorized_external_loads,
                removed_entries_list
            )
        
        self.stiffness_matrix = np.diag(
            np.inf * np.ones(len(self.vectorized_external_loads))
        )
        
        self.output_displacement = output_displacement
        if type(self.output_displacement) == OutputDisplacement:
            self.output_displacement.update_property(self.grid)
    
    def compute_stiffness_matrix(self, in_plane_thickness):

        self.stiffness_matrix = self.grid.compute_stiffness_matrix(in_plane_thickness)

        # Imposing displacement boundary constraints
        for index, boundary_constraint_conditions in self.sorted_boundary_constraints.items():
            conditions = set(boundary_constraint_conditions.keys())
            removed_entries_list = []
            for condition in conditions:
                removed_entries_list.append(
                    self.grid.degree_of_freedom*index+DOF_DICT[condition]
                )
            self.stiffness_matrix = np.delete(self.stiffness_matrix, removed_entries_list, axis=0)
            self.stiffness_matrix = np.delete(self.stiffness_matrix, removed_entries_list, axis=1)

    @abstractmethod
    def deform(self,):
        pass

class TrussFEM(FEM):
    def deform(self, in_plane_thickness=None):
        
        self.compute_stiffness_matrix(in_plane_thickness)

        grid_displacement = self.compute_grid_displacement()
            
        return grid_displacement
    
    def compute_grid_displacement(self):
        return self.incorporate_boundary_constraints(
            grid_displacement=np.linalg.inv(self.stiffness_matrix) @ self.vectorized_external_loads
        )
    
    def incorporate_boundary_constraints(self, grid_displacement):
        complete_grid_displacement = np.zeros(2*len(self.grid.nodes))
        counter = 0
        boundary_constraints_indicies = list(self.sorted_boundary_constraints.keys())
        for i in range(len(self.grid.nodes)):
            if boundary_constraints_indicies != [] and i == boundary_constraints_indicies[-1]:
                conditions = set(self.sorted_boundary_constraints[i].keys())
                if {'position_x', 'position_y'}.issubset(conditions):
                    direction_x_displacement = self.sorted_boundary_constraints[i]['position_x']
                    direction_y_displacement = self.sorted_boundary_constraints[i]['position_y']
                elif 'position_x' in conditions:
                    direction_x_displacement = self.sorted_boundary_constraints[i]['position_x']
                    direction_y_displacement = grid_displacement[counter]
                    counter += 1
                elif 'position_y' in conditions:
                    direction_x_displacement = grid_displacement[counter]
                    direction_y_displacement = self.sorted_boundary_constraints[i]['position_y']
                    counter += 1
                boundary_constraints_indicies.pop()
            else:
                direction_x_displacement = grid_displacement[counter]
                direction_y_displacement = grid_displacement[counter+1]
                counter += 2
            complete_grid_displacement[2*i] = direction_x_displacement
            complete_grid_displacement[2*i+1] = direction_y_displacement
            
        return complete_grid_displacement


class BeamFEM(FEM):
    def deform(self, in_plane_thickness=None):
            
        return None