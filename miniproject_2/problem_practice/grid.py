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

        self.stiffness_matrix = np.diag(
            np.inf * np.ones(self.degree_of_freedom * number_of_nodes)
        )

    def compute(self, value):
        if isinstance(value, (int, float)):
            return np.ones(len(self.links)) * value
        return np.array(value)
    
    @abstractmethod
    def deploy_links(self,):
        pass

    @abstractmethod
    def compute_stiffness_matrix(self, in_plane_thickness=None):
        pass

    @abstractmethod
    def compute_gradient_of_strain_energy(self, grid_displacement: np.ndarray, in_plane_thickness=None):
        pass

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
        condition: str, 
        value: float, 
    ):
        nodes_indicies = node_range.get_indicies(self.nodes)
        for index in nodes_indicies:
            condition_list.append(
                [index, condition, value]
            )
    
    def plot(self, ax, grid_displacement=None, in_plane_thickness=None, **kwargs):
        grid_displacement = np.zeros(2*len(self.nodes)) if type(grid_displacement) == type(None) else grid_displacement
        in_plane_thickness = self.in_plane_thickness if type(in_plane_thickness) == type(None) else in_plane_thickness
        in_plane_thickness_max = max(in_plane_thickness) / kwargs.pop('linewidth', 3)

        for link, thickness in zip(self.links, in_plane_thickness):
            i, j = link[0], link[1]
            ax.plot(
                [self.nodes[i][0]+grid_displacement[2*i], self.nodes[j][0]+grid_displacement[2*j]],
                [self.nodes[i][1]+grid_displacement[2*i+1], self.nodes[j][1]+grid_displacement[2*j+1]],
                linewidth=thickness/in_plane_thickness_max,
                **kwargs
            )
        return ax

class TrussGrid(Grid):

    LOCAL_STIFFNESS_MATRIX = np.array(
        [[ 1, -1],
         [-1,  1]]
    )

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
    
    def compute_stiffness_matrix(self, in_plane_thickness=None):
        in_plane_thickness = self.in_plane_thickness if type(in_plane_thickness) == type(None) else in_plane_thickness
        cross_sectional_area = in_plane_thickness * self.out_of_plane_thickness
        self.stiffness_matrix[:, :] = 0
        for n, link in enumerate(self.links):
            i, j = link[0], link[1]
            angle = self.angle_of_links[n]
            local_stiffness = (
                self.youngs_modulus[n] * cross_sectional_area[n] / self.length_of_links[n]
            )
            transformation_matrix = np.array(
                [[np.cos(angle), np.sin(angle), 0, 0],
                 [0, 0, np.cos(angle), np.sin(angle)]]
            )
            local_stiffness_matrix = local_stiffness * (
                transformation_matrix.T @ (
                    TrussGrid.LOCAL_STIFFNESS_MATRIX @ transformation_matrix
                )
            )
            
            indices = [2*i, 2*i+1, 2*j, 2*j+1]
            self.stiffness_matrix[np.ix_(indices, indices)] += local_stiffness_matrix
        return self.stiffness_matrix.copy()

    def compute_gradient_of_strain_energy(
        self,
        grid_displacement: np.ndarray,
        in_plane_thickness=None
    ):
        in_plane_thickness = self.in_plane_thickness if type(in_plane_thickness) == type(None) else in_plane_thickness
        cross_sectional_area = in_plane_thickness * self.out_of_plane_thickness

        gradient_of_strain_energy = np.zeros(len(self.links))
        
        for n, link in enumerate(self.links):
            i, j = link[0], link[1]
            angle = self.angle_of_links[n]
            local_stiffness = (
                self.youngs_modulus[n] * cross_sectional_area[n] / self.length_of_links[n]
            )
            local_displacement = np.array(
                [grid_displacement[2*i]*np.cos(angle)+grid_displacement[2*i+1]*np.sin(angle),
                 grid_displacement[2*j]*np.cos(angle)+grid_displacement[2*j+1]*np.sin(angle)]
            )
            gradient_of_strain_energy[n] = -0.5 * local_stiffness * (
                local_displacement @ (
                    TrussGrid.LOCAL_STIFFNESS_MATRIX @ 
                    local_displacement
                )
            )
        return gradient_of_strain_energy

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

    def deploy_links(self):
        links = []
        length_of_links = []
        angle_of_links = []
        for i in range(len(self.nodes)):
            if i % self.number_of_nodes_at_each_side[0] == 0:
                # i at the left boundary
                grid_list = np.array([
                    1, self.number_of_nodes_at_each_side[0], self.number_of_nodes_at_each_side[0]+1
                ])
            elif i % self.number_of_nodes_at_each_side[0] == self.number_of_nodes_at_each_side[0]-1:
                # i at the right boundary
                grid_list = np.array([self.number_of_nodes_at_each_side[0]-1, self.number_of_nodes_at_each_side[0]])
            else:
                # i not at boundaries
                grid_list = np.array([
                    1, self.number_of_nodes_at_each_side[0]-1, self.number_of_nodes_at_each_side[0], self.number_of_nodes_at_each_side[0]+1
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
