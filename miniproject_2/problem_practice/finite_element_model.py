"""
Created on Mar. 09, 2024
@author: Heng-Sheng Hanson Chang
"""

from abc import ABC, abstractmethod
from collections import defaultdict
import numpy as np

from grid import Grid

class FEM(ABC):
    def __init__(self, grid: Grid):
        self.grid = grid
    
    @abstractmethod
    def deform(self,):
        pass

class TrussFEM(FEM):
    def __init__(
        self, 
        grid: Grid, 
        boundary_constraints: list, 
        external_loads: list, 
    ):
        super().__init__(grid)
        number_of_nodes = len(self.grid.nodes)
        self.stiffness_matrix = np.zeros((2*number_of_nodes, 2*number_of_nodes))

        self.vectorized_external_loads = np.zeros(2*number_of_nodes)
        for external_load in external_loads:
            index, condition, value = external_load
            if condition == 'direction_x':
                self.vectorized_external_loads[2*index] = value
            elif condition == 'direction_y':
                self.vectorized_external_loads[2*index+1] = value

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
            if {'position_x', 'position_y'}.issubset(conditions):
                self.vectorized_external_loads = np.delete(
                    self.vectorized_external_loads,
                    [2*index, 2*index+1]
                )
            elif 'position_x' in conditions:
                self.vectorized_external_loads = np.delete(
                    self.vectorized_external_loads,
                    2*index
                )
            elif 'position_y' in conditions:
                self.vectorized_external_loads = np.delete(
                    self.vectorized_external_loads,
                    2*index+1
                )

    def compute_stiffness_matrix(self, in_plane_thickness):

        cross_sectional_area = in_plane_thickness * self.grid.out_of_plane_thickness

        self.stiffness_matrix *= 0
        for n, link in enumerate(self.grid.links):
            i, j = link[0], link[1]
            angle = self.grid.angle_of_links[n]
            local_stiffness = (
                self.grid.youngs_modulus[n] * cross_sectional_area[n] / self.grid.length_of_links[n]
            )
            transformation_matrix = np.array(
                [[np.cos(angle), np.sin(angle), 0, 0],
                 [0, 0, np.cos(angle), np.sin(angle)]]
            )
            local_stiffness_matrix = local_stiffness * (
                transformation_matrix.T @ (
                    Grid.LOCAL_STIFFNESS_MATRIX @ transformation_matrix
                )
            )
            
            indices = [2*i, 2*i+1, 2*j, 2*j+1]
            self.stiffness_matrix[np.ix_(indices, indices)] += local_stiffness_matrix

        # Imposing displacement boundary constraints
        reduced_stiffness_matrix = self.stiffness_matrix.copy()
        for index, boundary_constraint_conditions in self.sorted_boundary_constraints.items():
            conditions = set(boundary_constraint_conditions.keys())
            if {'position_x', 'position_y'}.issubset(conditions):
                reduced_stiffness_matrix = np.delete(reduced_stiffness_matrix, [2*index, 2*index+1], axis=0)
                reduced_stiffness_matrix = np.delete(reduced_stiffness_matrix, [2*index, 2*index+1], axis=1)
            elif 'position_x' in conditions:
                reduced_stiffness_matrix = np.delete(reduced_stiffness_matrix, 2*index, axis=0)
                reduced_stiffness_matrix = np.delete(reduced_stiffness_matrix, 2*index, axis=1)
            elif 'position_y' in conditions:
                reduced_stiffness_matrix = np.delete(reduced_stiffness_matrix, 2*index+1, axis=0)
                reduced_stiffness_matrix = np.delete(reduced_stiffness_matrix, 2*index+1, axis=1)

        return reduced_stiffness_matrix


    def deform(self, in_plane_thickness=None):
        
        stiffness_matrix = self.compute_stiffness_matrix(
            in_plane_thickness=self.grid.in_plane_thickness if type(in_plane_thickness) == type(None) else in_plane_thickness
        )

        grid_displacement = self.compute_grid_displacement(stiffness_matrix)
            
        return grid_displacement
    
    def compute_grid_displacement(self, stiffness_matrix):
        return self.incorporate_boundary_constraints(
            grid_displacement=np.linalg.inv(stiffness_matrix) @ self.vectorized_external_loads
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
