"""
Created on Mar. 02, 2024
@author: Heng-Sheng Hanson Chang
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from finite_element_model import FEM

class VolumeConstraint:
    def __init__(
        self, 
        total_max_volume: float, 
        upper_bound: float,
        lower_bound: float,
        update_ratio: float,
        update_power: float,
        lagrange_multiplier_setting: dict
    ):
        self.total_max_volume = total_max_volume
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.update_ratio = update_ratio
        self.update_power = update_power
        self.lagrange_multiplier_max = lagrange_multiplier_setting.get('max', 1e5)
        self.lagrange_multiplier_min = lagrange_multiplier_setting.get('min', 0)
        self.lagrange_multiplier_tol = lagrange_multiplier_setting.get('tol', 1e-4)

    @staticmethod
    def compute_total_volume(length_of_links, in_plane_thickness, out_of_plane_thickness):
        return np.sum(length_of_links * in_plane_thickness * out_of_plane_thickness)
    
    def satisfy(
        self, 
        thickness, 
        gradient_of_strain_energy, 
        length_of_links, 
        out_of_plane_thickness
    ):
        lagrange_multiplier_upper = self.lagrange_multiplier_max
        lagrange_multiplier_lower = self.lagrange_multiplier_min
        while lagrange_multiplier_upper - lagrange_multiplier_lower > self.lagrange_multiplier_tol:
            # Guess a lagrange multiplier using the average of lower and upper ones
            lagrange_multiplier = (lagrange_multiplier_lower + lagrange_multiplier_upper) / 2
            
            # Compute the new thickness
            new_thickness = thickness * (
                - self.update_ratio * gradient_of_strain_energy / 
                (lagrange_multiplier * length_of_links)
            ) ** self.update_power
            
            # Limit the thickness is in between the lower and upper bounds
            new_thickness = np.clip(
                new_thickness, self.lower_bound, self.upper_bound
            )

            # Coupute the total volume in current setting
            total_volume = VolumeConstraint.compute_total_volume(
                length_of_links,
                new_thickness,
                out_of_plane_thickness
            )

            # Check if the volume constraint is satisfied
            if total_volume > self.total_max_volume:
                # If it is not satisfied, increase the lower bound of the lagrange multiplier
                lagrange_multiplier_lower = lagrange_multiplier
            else:
                # If it is satisfied, decrease the upper bound of the lagrange multiplier
                lagrange_multiplier_upper = lagrange_multiplier
        
        return new_thickness

class TopologyOptimization:
    def __init__(
        self, 
        fem: FEM,
        volume_constraint: VolumeConstraint, 
        number_of_maximum_iterations: int,
        **kwargs
    ):
        self.fem = fem
        self.volume_constraint = volume_constraint
        self.number_of_maximum_iterations = number_of_maximum_iterations
        self.tol = kwargs.get('tol', 1e-4)

    def optimize(self, **kwargs):
        plot_flag = kwargs.get('plot_flag', False)
        if plot_flag:
            show_numbers = kwargs.get('show_numbers', 4)
            fig, ax = plt.subplots()
            colors = iter(mpl.colormaps['tab10'].colors)
            plot_indicies = np.logspace(
                0, 
                np.log10(self.number_of_maximum_iterations), 
                show_numbers, 
                dtype=int
            )
            plot_indicies[0] = 0
        
        thickness = self.fem.grid.in_plane_thickness.copy()
        for i in range(self.number_of_maximum_iterations):

            # Deform the grid based on the current thickness setting
            self.fem.deform(
                in_plane_thickness=thickness
            )

            # Compute gradient of strain energy based on current thickness setting
            gradient_of_strain_energy = self.fem.grid.compute_gradient_of_strain_energy(
                grid_displacement=self.fem.grid_displacement,
                in_plane_thickness=thickness
            )

            # Update the thickness based on the gradient for the next iteration
            thickness = self.update_thickness(thickness, gradient_of_strain_energy)

            if plot_flag and i in plot_indicies:
                color = next(colors)
                ax = self.fem.grid.plot(
                    ax,
                    in_plane_thickness=thickness, 
                    color=color,
                )
                ax.plot(
                    [],[], 
                    color=color,
                    label='iter No. '+str(i+1)
                )
                
        if plot_flag:
            color = next(colors)
            ax = self.fem.grid.plot(
                ax,
                in_plane_thickness=thickness, 
                color=color,
            )
            ax.plot(
                [],[], 
                color=color,
                label='iter No. '+str(i+1)
            )
            ax.axis('equal')
            ax.legend()
            fig.tight_layout()

        return thickness
    
    def update_thickness(self, thickness, gradient_of_strain_energy):
        return self.volume_constraint.satisfy(
            thickness, 
            gradient_of_strain_energy, 
            self.fem.grid.length_of_links, 
            self.fem.grid.out_of_plane_thickness
        )
