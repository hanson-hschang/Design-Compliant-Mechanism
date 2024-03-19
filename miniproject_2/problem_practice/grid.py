"""
Created on Mar. 01, 2024
@author: Heng-Sheng Hanson Chang
"""

from abc import ABC, abstractmethod
from collections import defaultdict
import numpy as np

class NodeRange:
    def __init__(self, condition: str, **kwargs):
        self.condition = condition
        if self.condition == 'vertical_segement':
            self.x_range = [kwargs['x_value'], kwargs['x_value']]
            self.y_range = kwargs['y_range']
        if self.condition == 'horizontal_segement':
            self.x_range = kwargs['x_range']
            self.y_range = [kwargs['y_value'], kwargs['y_value']]
        if self.condition == 'point':
            self.x_value = kwargs['x_value']
            self.y_value = kwargs['y_value']

    def get_indicies(self, nodes: defaultdict):
        nodes_indicies = []
        if 'segement' in self.condition:
            for index in list(nodes.keys()):
                position_x, position_y = nodes[index]
                if self.x_range[0] <= position_x <= self.x_range[1] and self.y_range[0] <= position_y <= self.y_range[1]:
                    nodes_indicies.append(index)
        if 'point' in self.condition:
            point_position = np.array([self.x_value, self.y_value])
            minimum_distance_index = NodeRange.find_nearest_node(
                nodes=nodes,
                position=point_position
            )
            nodes_indicies.append(minimum_distance_index)
        return nodes_indicies
    
    @staticmethod
    def find_nearest_node(nodes: defaultdict, position: np.array):
        minimum_distance = np.inf
        minimum_distance_index = None
        for index in list(nodes.keys()):
            distance = np.linalg.norm(
                np.array(nodes[index]) - 
                position
            )
            if distance < minimum_distance:
                minimum_distance = distance
                minimum_distance_index = index
        return minimum_distance_index

class Grid(ABC):

    DOF_DICT = {}
    DIRECTION_DICT = {}

    def __init__(
        self,
        number_of_links: list[int], 
        length_of_sides: list[float], 
        youngs_modulus: float | np.ndarray,
        in_plane_thickness: float | np.ndarray, 
        out_of_plane_thickness: float | np.ndarray,
        degree_of_freedom: int = 0,
    ):
        self.number_of_nodes_at_each_side = np.array(number_of_links) + 1
        self.nodes = defaultdict(list)
        delta_x = length_of_sides[0] / number_of_links[0]
        delta_y = length_of_sides[1] / number_of_links[1]
        number_of_nodes = 0
        for j in range(self.number_of_nodes_at_each_side[1]):
            for i in range(self.number_of_nodes_at_each_side[0]):
                self.nodes[number_of_nodes] = [i*delta_x, j*delta_y]
                number_of_nodes += 1
        
        self.links, self.length_of_links, self.angle_of_links = self.deploy_links()
        self.youngs_modulus = self.compute(youngs_modulus)
        self.in_plane_thickness = self.compute(in_plane_thickness)
        self.out_of_plane_thickness = self.compute(out_of_plane_thickness)

        self.degree_of_freedom = degree_of_freedom
        total_degree_of_freedom = self.degree_of_freedom * number_of_nodes
        self.stiffness_matrix = np.diag(
            np.inf * np.ones(total_degree_of_freedom)
        )
        self.gradient_of_stiffness_matrix = np.zeros(
            (len(self.links), total_degree_of_freedom, total_degree_of_freedom)
        )

    def compute(self, value):
        if isinstance(value, (int, float)):
            return np.ones(len(self.links)) * value
        return np.array(value)
    
    def check_in_plane_thickness_value(
        self, 
        in_plane_thickness: np.ndarray | None = None,
    ):
        return self.in_plane_thickness if type(in_plane_thickness) == type(None) else in_plane_thickness

    def compute_cross_sectional_area(
        self, 
        in_plane_thickness: np.ndarray,
    ):
        return in_plane_thickness * self.out_of_plane_thickness
    
    def compute_second_moment_of_cross_sectional_area(
        self, 
        in_plane_thickness: np.ndarray
    ):
        return self.out_of_plane_thickness * (in_plane_thickness**3) / 12.

    def compute_gradient_of_cross_sectional_area(
        self, 
        in_plane_thickness: np.ndarray,
    ):
        return self.out_of_plane_thickness
    
    def compute_gradient_of_second_moment_of_cross_sectional_area(
        self, 
        in_plane_thickness: np.ndarray
    ):
        return self.out_of_plane_thickness * (in_plane_thickness**2) / 4.
    
    @abstractmethod
    def deploy_links(self,):
        pass

    @abstractmethod
    def compute_stiffness_matrix(
        self, 
        in_plane_thickness: np.ndarray | None = None
    ):
        pass

    @abstractmethod
    def compute_gradient_of_stiffness_matrix(
        self, 
        in_plane_thickness: np.ndarray | None = None
    ):
        pass

    def compute_strain_energy(
        self,
        grid_displacement: np.ndarray,
        grid_displacement_the_other: np.ndarray | None = None,
    ):
        if type(grid_displacement_the_other) == type(None):
            grid_displacement_the_other = grid_displacement
        energy = 0.5 * (
            grid_displacement_the_other @ (
                self.stiffness_matrix @ grid_displacement
            )
        )
        return energy
    
    def compute_gradient_of_strain_energy(
        self,
        grid_displacement: np.ndarray,
        grid_displacement_the_other: np.ndarray | None = None,
    ):
        if type(grid_displacement_the_other) == type(None):
            grid_displacement_the_other = grid_displacement
        gradient_of_strain_energy = np.zeros(len(self.links))
        for n in range(len(self.links)):
            gradient_of_strain_energy[n] = -0.5 * (
                grid_displacement_the_other @ (
                    self.gradient_of_stiffness_matrix[n] @
                    grid_displacement
                )
            )
        return gradient_of_strain_energy  

    def remove_nodes(self, x_bounds=None, y_bounds=None):
        x_bounds = [-np.inf, np.inf] if type(x_bounds) == type(None) else x_bounds
        y_bounds = [-np.inf, np.inf] if type(y_bounds) == type(None) else y_bounds
        x_bounds[0] = -np.inf if type(x_bounds[0]) == type(None) else x_bounds[0]
        x_bounds[1] = np.inf if type(x_bounds[1]) == type(None) else x_bounds[1]
        y_bounds[0] = -np.inf if type(y_bounds[0]) == type(None) else y_bounds[0]
        y_bounds[1] = np.inf if type(y_bounds[1]) == type(None) else y_bounds[1]
        
        removed_nodes_list = []
        kept_nodes_list = []        
        for index in list(self.nodes.keys()):
            position_x, position_y = self.nodes[index]
            del self.nodes[index]
            if x_bounds[0] <= position_x <= x_bounds[1] and y_bounds[0] <= position_y <= y_bounds[1]:
                removed_nodes_list.append(index)
            else:
                kept_nodes_list.append(index)
                self.nodes[len(kept_nodes_list)-1] = [position_x, position_y]

        kept_links_list = []
        kept_links = []
        for i, link in enumerate(self.links):
            if len(np.intersect1d(link, removed_nodes_list))==0:
                kept_links_list.append(i)
                kept_links.append(
                    [kept_nodes_list.index(link[0]), kept_nodes_list.index(link[1])]
                )
        self.links = np.array(kept_links)
        self.update_link_related_parameters(kept_links_list)

    def update_link_related_parameters(self, kept_links_list: list):
        self.length_of_links = np.array([
            self.length_of_links[index]
            for index in range(len(self.length_of_links)) if index in kept_links_list
        ])
        self.angle_of_links = np.array([
            self.angle_of_links[index]
            for index in range(len(self.angle_of_links)) if index in kept_links_list
        ])
        self.youngs_modulus = np.array([
            self.youngs_modulus[index]
            for index in range(len(self.youngs_modulus)) if index in kept_links_list
        ])
        self.in_plane_thickness = np.array([
            self.in_plane_thickness[index]
            for index in range(len(self.in_plane_thickness)) if index in kept_links_list
        ])
        self.out_of_plane_thickness = np.array([
            self.out_of_plane_thickness[index]
            for index in range(len(self.out_of_plane_thickness)) if index in kept_links_list
        ])
    
    def add_conditions(
        self, 
        condition_list: list,
        node_range: NodeRange, 
        conditions: str | list[str], 
        value: float, 
    ):
        nodes_indicies = node_range.get_indicies(self.nodes)
        conditions = [conditions] if type(conditions) == str else conditions
        for index in nodes_indicies:
            for condition in conditions:
                condition_list.append(
                    [index, condition, value]
                )
    
    def plot(
        self, 
        ax, 
        grid_displacement: np.ndarray | None = None, 
        in_plane_thickness: np.ndarray | None = None, 
        **kwargs
    ):
        grid_displacement = np.zeros(self.degree_of_freedom*len(self.nodes)) if type(grid_displacement) == type(None) else grid_displacement
        in_plane_thickness = self.check_in_plane_thickness_value(in_plane_thickness)
        in_plane_thickness_max = max(in_plane_thickness) / kwargs.pop('linewidth', 3)

        for link, thickness in zip(self.links, in_plane_thickness):
            i, j = link[0], link[1]
            ax.plot(
                [self.nodes[i][0]+grid_displacement[self.degree_of_freedom*i], 
                 self.nodes[j][0]+grid_displacement[self.degree_of_freedom*j]],
                [self.nodes[i][1]+grid_displacement[self.degree_of_freedom*i+1], 
                 self.nodes[j][1]+grid_displacement[self.degree_of_freedom*j+1]],
                linewidth=thickness/in_plane_thickness_max,
                **kwargs
            )
        return ax

class TrussGrid(Grid):

    DOF_DICT = {
        'position_x': 0,
        'position_y': 1,
    }

    DIRECTION_DICT = {
        'direction_x': 0,
        'direction_y': 1,
    }

    def __init__(
        self,
        number_of_links: list[int], 
        length_of_sides: list[float], 
        youngs_modulus: float | np.ndarray,
        in_plane_thickness: float | np.ndarray, 
        out_of_plane_thickness: float | np.ndarray,
    ):
        super().__init__(
            number_of_links, 
            length_of_sides, 
            youngs_modulus,
            in_plane_thickness, 
            out_of_plane_thickness,
            degree_of_freedom=2,
        )
        self.transformation_matrix = np.zeros((len(self.links), 2, 4))
        for n, angle in enumerate(self.angle_of_links):
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            self.transformation_matrix[n, :, :] = np.array(
                [[ cos_angle, sin_angle,         0,         0],
                 [         0,         0, cos_angle, sin_angle]]
            )

    def get_indicies(self, i, j):
        indicies = [2*i, 2*i+1, 2*j, 2*j+1]
        return np.ix_(indicies, indicies)
    
    def deploy_links(self):
        links = []
        length_of_links = []
        angle_of_links = []
        for i in range(len(self.nodes)):
            slope_list = []
            for j in range(i+1, len(self.nodes)):
                angle = np.arctan2(
                    self.nodes[j][1]-self.nodes[i][1], 
                    self.nodes[j][0]-self.nodes[i][0]
                )
                if not (angle in slope_list):
                    slope_list.append(angle)                
                    links.append([i, j])
                    length_of_links.append(
                        np.linalg.norm(
                            np.array(self.nodes[i]) - np.array(self.nodes[j])
                        )
                    )
                    angle_of_links.append(angle)
        return np.array(links), np.array(length_of_links), np.array(angle_of_links)

    def compute_stiffness_matrix(
        self, 
        in_plane_thickness: np.ndarray | None = None,
    ):
        in_plane_thickness = self.check_in_plane_thickness_value(in_plane_thickness)
        cross_sectional_area = self.compute_cross_sectional_area(in_plane_thickness)
        self.stiffness_matrix[:, :] = 0
        for n, link in enumerate(self.links):
            i, j = link[0], link[1]
            local_stiffness = (
                self.youngs_modulus[n] * cross_sectional_area[n] / self.length_of_links[n]
            )
            local_stiffness_matrix = np.array(
                [[  local_stiffness, -local_stiffness],
                 [ -local_stiffness,  local_stiffness]]
            )
            self.stiffness_matrix[self.get_indicies(i, j)] += (
                self.transformation_matrix[n].T @ (
                    local_stiffness_matrix @ self.transformation_matrix[n]
                )
            )
        return self.stiffness_matrix.copy()

    def compute_gradient_of_stiffness_matrix(
        self,
        in_plane_thickness: np.ndarray | None = None,  
    ):
        in_plane_thickness = self.check_in_plane_thickness_value(in_plane_thickness)
        gradient_of_cross_sectional_area = (
            self.compute_gradient_of_cross_sectional_area(
                in_plane_thickness
            )
        )
        self.gradient_of_stiffness_matrix[:, :, :] = 0
        for n, link in enumerate(self.links):
            i, j = link[0], link[1]
            gradient_of_local_stiffness = (
                self.youngs_modulus[n] * gradient_of_cross_sectional_area[n] / self.length_of_links[n]
            )
            gradient_of_local_stiffness_matrix = np.array(
                [[  gradient_of_local_stiffness, -gradient_of_local_stiffness],
                 [ -gradient_of_local_stiffness,  gradient_of_local_stiffness]]
            )
            self.gradient_of_stiffness_matrix[n][self.get_indicies(i, j)] = (
                self.transformation_matrix[n].T @ (
                    gradient_of_local_stiffness_matrix @ 
                    self.transformation_matrix[n]
                )
            )

class BeamGrid(Grid):

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

    def __init__(
        self,
        number_of_links: list[int], 
        length_of_sides: list[float], 
        youngs_modulus: float | np.ndarray,
        in_plane_thickness: float | np.ndarray, 
        out_of_plane_thickness: float | np.ndarray,
    ):
        super().__init__(
            number_of_links, 
            length_of_sides, 
            youngs_modulus,
            in_plane_thickness, 
            out_of_plane_thickness,
            degree_of_freedom=3,
        )
        self.transformation_matrix = np.zeros((len(self.links), 6, 6))
        for n, angle in enumerate(self.angle_of_links):
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            self.transformation_matrix[n, :, :] = np.array(
                [[  cos_angle, sin_angle, 0,          0,         0, 0],
                 [          0,         0, 0,  cos_angle, sin_angle, 0],
                 [ -sin_angle, cos_angle, 0,          0,         0, 0],
                 [          0,         0, 0, -sin_angle, cos_angle, 0],
                 [          0,         0, 1,          0,         0, 0],
                 [          0,         0, 0,          0,         0, 1]]
            )

    def get_indicies(self, i, j):
        indicies = [3*i, 3*i+1, 3*i+2, 3*j, 3*j+1, 3*j+2]
        return np.ix_(indicies, indicies)

    def deploy_links(self):
        links = []
        length_of_links = []
        angle_of_links = []
        for i in range(len(self.nodes)):
            if i % self.number_of_nodes_at_each_side[0] == 0:
                # i at the left boundary
                grid_list = np.array([
                    1, 
                    self.number_of_nodes_at_each_side[0], 
                    self.number_of_nodes_at_each_side[0]+1
                ])
            elif i % self.number_of_nodes_at_each_side[0] == self.number_of_nodes_at_each_side[0]-1:
                # i at the right boundary
                grid_list = np.array([
                    self.number_of_nodes_at_each_side[0]-1, 
                    self.number_of_nodes_at_each_side[0]
                ])
            else:
                # i not at boundaries
                grid_list = np.array([
                    1,
                    self.number_of_nodes_at_each_side[0]-1,
                    self.number_of_nodes_at_each_side[0],
                    self.number_of_nodes_at_each_side[0]+1
                ])
            grid_list += i
            grid_list = grid_list[np.where(grid_list<len(self.nodes))]
            for j in grid_list:
                links.append([i, j])
                length = np.linalg.norm(
                    np.array(self.nodes[i]) - np.array(self.nodes[j])
                )
                angle = np.arctan2(
                    self.nodes[j][1]-self.nodes[i][1], 
                    self.nodes[j][0]-self.nodes[i][0]
                )
                length_of_links.append(length)
                angle_of_links.append(angle)
        return np.array(links), np.array(length_of_links), np.array(angle_of_links)

    def compute_stiffness_matrix(
        self, 
        in_plane_thickness: np.ndarray | None = None,
    ):
        in_plane_thickness = self.check_in_plane_thickness_value(in_plane_thickness)
        cross_sectional_area = self.compute_cross_sectional_area(in_plane_thickness)
        second_moment_of_cross_sectional_area = self.compute_second_moment_of_cross_sectional_area(in_plane_thickness)
        self.stiffness_matrix[:, :] = 0
        for n, link in enumerate(self.links):
            i, j = link[0], link[1]
            EAdL = (
                self.youngs_modulus[n] * cross_sectional_area[n] / self.length_of_links[n]
            )
            EIdL_2 = 6 * (
                self.youngs_modulus[n] * second_moment_of_cross_sectional_area[n] / (self.length_of_links[n]**2)
            )
            EIdL_3 = 12 * (
                self.youngs_modulus[n] * second_moment_of_cross_sectional_area[n] / (self.length_of_links[n]**3)
            )
            EI2dL = (
                2 * self.youngs_modulus[n] * second_moment_of_cross_sectional_area[n] / self.length_of_links[n]
            )
            EI4dL = (
                4 * self.youngs_modulus[n] * second_moment_of_cross_sectional_area[n] / self.length_of_links[n]
            )
            local_stiffness_matrix = np.array(
                [[  EAdL, -EAdL,       0,       0,       0,       0],
                 [ -EAdL,  EAdL,       0,       0,       0,       0],
                 [     0,     0,  EIdL_3, -EIdL_3,  EIdL_2,  EIdL_2],
                 [     0,     0, -EIdL_3,  EIdL_3, -EIdL_2, -EIdL_2],
                 [     0,     0,  EIdL_2, -EIdL_2,   EI4dL,   EI2dL],
                 [     0,     0,  EIdL_2, -EIdL_2,   EI2dL,   EI4dL]]
            )
            self.stiffness_matrix[self.get_indicies(i, j)] += (
                self.transformation_matrix[n].T @ (
                    local_stiffness_matrix @ self.transformation_matrix[n]
                )
            )
        return self.stiffness_matrix.copy()
    
    def compute_gradient_of_stiffness_matrix(
        self,
        in_plane_thickness: np.ndarray | None = None,  
    ):
        in_plane_thickness = self.check_in_plane_thickness_value(in_plane_thickness)
        gradient_of_cross_sectional_area = (
            self.compute_gradient_of_cross_sectional_area(
                in_plane_thickness
            )
        )
        gradient_of_second_moment_of_cross_sectional_area = (
            self.compute_gradient_of_second_moment_of_cross_sectional_area(
                in_plane_thickness
            )
        )
        self.gradient_of_stiffness_matrix[:, :, :] = 0
        for n, link in enumerate(self.links):
            i, j = link[0], link[1]
            gradient_of_EAdL = (
                self.youngs_modulus[n] * gradient_of_cross_sectional_area[n] / self.length_of_links[n]
            )
            gradient_of_EIdL_2 = 6 * (
                self.youngs_modulus[n] * gradient_of_second_moment_of_cross_sectional_area[n] / (self.length_of_links[n]**2)
            )
            gradient_of_EIdL_3 = 12 * (
                self.youngs_modulus[n] * gradient_of_second_moment_of_cross_sectional_area[n] / (self.length_of_links[n]**3)
            )
            gradient_of_EI2dL = (
                2 * self.youngs_modulus[n] * gradient_of_second_moment_of_cross_sectional_area[n] / self.length_of_links[n]
            )
            gradient_of_EI4dL = (
                4 * self.youngs_modulus[n] * gradient_of_second_moment_of_cross_sectional_area[n] / self.length_of_links[n]
            )
            gradient_of_local_stiffness_matrix = np.array(
                [[  gradient_of_EAdL, -gradient_of_EAdL,                   0,                   0,                   0,                   0],
                 [ -gradient_of_EAdL,  gradient_of_EAdL,                   0,                   0,                   0,                   0],
                 [                 0,                 0,  gradient_of_EIdL_3, -gradient_of_EIdL_3,  gradient_of_EIdL_2,  gradient_of_EIdL_2],
                 [                 0,                 0, -gradient_of_EIdL_3,  gradient_of_EIdL_3, -gradient_of_EIdL_2, -gradient_of_EIdL_2],
                 [                 0,                 0,  gradient_of_EIdL_2, -gradient_of_EIdL_2,   gradient_of_EI4dL,   gradient_of_EI2dL],
                 [                 0,                 0,  gradient_of_EIdL_2, -gradient_of_EIdL_2,   gradient_of_EI2dL,   gradient_of_EI4dL]]
            ) 
            self.gradient_of_stiffness_matrix[n][self.get_indicies(i, j)] = (
                self.transformation_matrix[n].T @ (
                    gradient_of_local_stiffness_matrix @ 
                    self.transformation_matrix[n]
                )
            ) 
