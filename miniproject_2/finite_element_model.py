"""
Created on Mar. 09, 2024
@author: Heng-Sheng Hanson Chang
"""

from collections import defaultdict
import numpy as np

from grid import NodeRange, Grid

class OutputDisplacement:
    def __init__(
        self, 
        grid: Grid, 
        position: list, 
        condition: str, 
        directional_spring_constant: float
    ):
        self.position = np.array(position)
        self.condition = condition
        self.spring_constant = np.abs(directional_spring_constant)

        total_degree_of_freedom = grid.degree_of_freedom * len(grid.nodes)
        self.stiffness_matrix = np.zeros(
            (total_degree_of_freedom, total_degree_of_freedom)
        )
        self.vectorized_external_loads = np.zeros(total_degree_of_freedom)

        self.node_index = NodeRange.find_nearest_node(grid.nodes, self.position)
        self.dof_index = grid.degree_of_freedom * self.node_index + grid.DIRECTION_DICT[self.condition]
        self.stiffness_matrix[self.dof_index, self.dof_index] = self.spring_constant
        self.vectorized_external_loads[self.dof_index] = directional_spring_constant / self.spring_constant
    
    def remove_dof(
        self,
        removed_entries_list: list,
    ):
        # TODO: add warning if the output_displacement is in the boundary constraints
        self.vectorized_external_loads = np.delete(
            self.vectorized_external_loads,
            removed_entries_list
        )
        self.stiffness_matrix = np.delete(
            self.stiffness_matrix,
            removed_entries_list,
            axis=0,
        )
        self.stiffness_matrix = np.delete(
            self.stiffness_matrix,
            removed_entries_list,
            axis=1,
        )
    
    def add_output_displacement_matrix(self, stiffness_matrix: np.ndarray):
        return stiffness_matrix + self.stiffness_matrix
    
    def compute_energy(
        self,
        grid_displacement: np.ndarray,
        grid_displacement_the_other: np.ndarray | None = None,
    ):
        if type(grid_displacement_the_other) == type(None):
            grid_displacement_the_other = grid_displacement
        return 0.5 * (
            grid_displacement_the_other[self.dof_index] * 
            self.spring_constant * 
            grid_displacement[self.dof_index]
        )

class FEM:
    def __init__(
        self, 
        grid: Grid, 
        boundary_constraints: list,
        external_loads: list,
        output_displacement: OutputDisplacement | None = None,
    ):
        self.grid = grid
        degree_of_freedom = self.grid.degree_of_freedom
        total_degree_of_freedom = degree_of_freedom * len(self.grid.nodes)
        self.grid_displacement = np.zeros(total_degree_of_freedom)

        self.output_displacement = output_displacement
        self.vectorized_external_loads = np.zeros(total_degree_of_freedom)
        for external_load in external_loads:
            index, condition, value = external_load
            self.vectorized_external_loads[
                degree_of_freedom * index + self.grid.DIRECTION_DICT[condition]
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
                    degree_of_freedom*index + self.grid.DOF_DICT[condition]
                )
            self.vectorized_external_loads = np.delete(
                self.vectorized_external_loads,
                removed_entries_list
            )
            if type(self.output_displacement) == OutputDisplacement:
                self.output_displacement.remove_dof(removed_entries_list)
        
        self.stiffness_matrix = np.diag(
            np.inf * np.ones(len(self.vectorized_external_loads))
        )
        
        if type(self.output_displacement) == OutputDisplacement:
            self.add_output_displacement_stiffness_matrix = self.output_displacement.add_output_displacement_matrix
        else:
            self.add_output_displacement_stiffness_matrix = lambda stiffness_matrix: stiffness_matrix

    def compute_stiffness_matrix(self, in_plane_thickness=None):

        self.stiffness_matrix = self.grid.compute_stiffness_matrix(in_plane_thickness)

        # Imposing displacement boundary constraints
        for index, boundary_constraint_conditions in self.sorted_boundary_constraints.items():
            conditions = set(boundary_constraint_conditions.keys())
            removed_entries_list = []
            for condition in conditions:
                removed_entries_list.append(
                    self.grid.degree_of_freedom*index+self.grid.DOF_DICT[condition]
                )
            self.stiffness_matrix = np.delete(self.stiffness_matrix, removed_entries_list, axis=0)
            self.stiffness_matrix = np.delete(self.stiffness_matrix, removed_entries_list, axis=1)

        self.stiffness_matrix = self.add_output_displacement_stiffness_matrix(
            self.stiffness_matrix
        )

    def incorporate_boundary_constraints(self, grid_displacement: np.ndarray):
        degree_of_freedom = self.grid.degree_of_freedom
        counter = 0
        boundary_constraints_indicies = list(self.sorted_boundary_constraints.keys())
        complete_grid_displacement = np.zeros(self.grid_displacement.shape)
        for i in range(len(self.grid.nodes)):
            if boundary_constraints_indicies != [] and i == boundary_constraints_indicies[-1]:
                conditions = set(self.sorted_boundary_constraints[i].keys())
                node_displacement = np.zeros(degree_of_freedom)
                for condition, index in self.grid.DOF_DICT.items():
                    if condition in conditions:
                        node_displacement[index] = self.sorted_boundary_constraints[i][condition]
                    else:
                        node_displacement[index] = grid_displacement[counter]
                        counter += 1
                boundary_constraints_indicies.pop()
            else:
                node_displacement = grid_displacement[counter:counter+degree_of_freedom]
                counter += degree_of_freedom
            complete_grid_displacement[degree_of_freedom*i:degree_of_freedom*(i+1)] = node_displacement
        return complete_grid_displacement

    def compute_grid_displacement(self,):
        self.grid_displacement = self.incorporate_boundary_constraints(
            grid_displacement=np.linalg.inv(self.stiffness_matrix) @ self.vectorized_external_loads
        )

    def compute_grid_virtual_displacement(self,):
        return self.incorporate_boundary_constraints(
            grid_displacement=np.linalg.inv(self.stiffness_matrix) @ self.output_displacement.vectorized_external_loads
        )

    def deform(
        self, 
        in_plane_thickness: np.ndarray | None = None,
    ):
        self.compute_stiffness_matrix(in_plane_thickness)
        self.compute_grid_displacement()
        return self.grid_displacement
    
    def virtual_deform(
        self, 
        in_plane_thickness: np.ndarray | None = None, 
    ):
        return self.compute_grid_virtual_displacement()
