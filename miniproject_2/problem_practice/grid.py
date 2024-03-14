"""
Created on Mar. 01, 2024
@author: Heng-Sheng Hanson Chang
"""

from abc import ABC, abstractmethod
import numpy as np

class Grid(ABC):

    LOCAL_STIFFNESS_MATRIX = np.array(
        [[ 1, -1],
         [-1,  1]]
    )

    def __init__(
        self, 
        number_of_links, 
        length_of_sides, 
        youngs_modulus,
        in_plane_thickness, 
        out_of_plane_thickness,
    ):
    
        self.nodes = []
        delta_x = length_of_sides[0] / number_of_links[0]
        delta_y = length_of_sides[1] / number_of_links[1]
        for j in range(number_of_links[1]+1):
            for i in range(number_of_links[0]+1):
                self.nodes.append([i*delta_x, j*delta_y])
        self.nodes = np.array(self.nodes)
        
        self.links, self.length_of_links, self.angle_of_links = self.deploy_links()
        self.youngs_modulus = self.compute(youngs_modulus)
        self.in_plane_thickness = self.compute(in_plane_thickness)
        self.out_of_plane_thickness = self.compute(out_of_plane_thickness)

    @abstractmethod
    def deploy_links(self,):
        pass

    def compute(self, value):
        if isinstance(value, (int, float)):
            return np.ones(len(self.links)) * value
        return np.array(value)

    def compute_gradient_of_strain_energy(self, grid_displacement, in_plane_thickness=None):
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
                    Grid.LOCAL_STIFFNESS_MATRIX @ 
                    local_displacement
                )
            )
        return gradient_of_strain_energy
    
    def plot(self, ax, grid_displacement=None, in_plane_thickness=None, **kwargs):
        grid_displacement = np.zeros(2*len(self.nodes)) if type(grid_displacement) == type(None) else grid_displacement
        in_plane_thickness = self.in_plane_thickness if type(in_plane_thickness) == type(None) else in_plane_thickness
        in_plane_thickness_max = max(in_plane_thickness) / kwargs.pop('linewidth', 3)
        
        for link, thickness in zip(self.links, in_plane_thickness):
            i, j = link[0], link[1]
            ax.plot(
                [self.nodes[i, 0]+grid_displacement[2*i], self.nodes[j, 0]+grid_displacement[2*j]],
                [self.nodes[i, 1]+grid_displacement[2*i+1], self.nodes[j, 1]+grid_displacement[2*j+1]],
                linewidth=thickness/in_plane_thickness_max,
                **kwargs
            )
        return ax

class TrussGrid(Grid):
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