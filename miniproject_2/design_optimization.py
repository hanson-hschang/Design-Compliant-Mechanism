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
        update_step_max: float,
        lagrange_multiplier_setting: dict
    ):
        self.total_max_volume = total_max_volume
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.update_ratio = update_ratio
        self.update_power = update_power
        self.update_step_max = update_step_max
        self.lagrange_multiplier_max = lagrange_multiplier_setting.get('max', 1e5)
        self.lagrange_multiplier_min = lagrange_multiplier_setting.get('min', 0)
        self.lagrange_multiplier_tol = lagrange_multiplier_setting.get('tol', 1e-4)

    @staticmethod
    def compute_total_volume(
        length_of_links: np.ndarray, 
        in_plane_thickness: np.ndarray, 
        out_of_plane_thickness: np.ndarray,
    ):
        return np.sum(length_of_links * in_plane_thickness * out_of_plane_thickness)
    
    def satisfy(
        self, 
        thickness: np.ndarray,  
        gradient: np.ndarray, 
        length_of_links: np.ndarray,  
        out_of_plane_thickness: np.ndarray, 
    ):
        lagrange_multiplier_upper = self.lagrange_multiplier_max
        lagrange_multiplier_lower = self.lagrange_multiplier_min
        lower_bound = np.array([max(self.lower_bound, thickness_at_n-self.update_step_max) for thickness_at_n in thickness])
        upper_bound = np.array([min(self.upper_bound, thickness_at_n+self.update_step_max) for thickness_at_n in thickness])
        while lagrange_multiplier_upper - lagrange_multiplier_lower > self.lagrange_multiplier_tol:
            # Guess a lagrange multiplier using the average of lower and upper ones
            lagrange_multiplier = (lagrange_multiplier_lower + lagrange_multiplier_upper) / 2
            
            # Compute the new thickness
            ratio = (
                -self.update_ratio * gradient / 
                (lagrange_multiplier * length_of_links)
            )
            negative_index = ratio < 0
            ratio[negative_index] = 0
            new_thickness = thickness * (
                ratio ** self.update_power
            )
            
            # Limit the thickness is in between the lower and upper bounds
            new_thickness = np.clip(
                new_thickness, lower_bound, upper_bound
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
        self.tol = kwargs.pop('tol', 1e-4)

    def update_thickness(
        self, 
        thickness: np.ndarray, 
        gradient: np.ndarray
    ):
        return self.volume_constraint.satisfy(
            thickness, 
            gradient, 
            self.fem.grid.length_of_links, 
            self.fem.grid.out_of_plane_thickness
        )

    def optimize_energy(self, **kwargs):
        plot_flag = kwargs.pop('plot_flag', False)
        if plot_flag:
            show_numbers = kwargs.pop('show_numbers', 4)
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
            grid_displacement = self.fem.deform(
                in_plane_thickness=thickness,
                optimization=True,
            )

            # Compute gradient of strain energy based on current thickness setting
            self.fem.grid.compute_gradient_of_stiffness_matrix(
                in_plane_thickness=thickness,
            )
            gradient_of_strain_energy = self.fem.grid.compute_gradient_of_strain_energy(
                grid_displacement=grid_displacement,
            )

            thickness[...] = self.update_thickness(thickness, gradient_of_strain_energy)

            if plot_flag and i in plot_indicies:
                color = next(colors)
                ax = self.fem.grid.plot_links(
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
            ax = self.fem.grid.plot_links(
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
    
    def optimize_geometric_advantage(self, **kwargs):
        plot_flag = kwargs.pop('plot_flag', False)
        if plot_flag:
            show_numbers = kwargs.pop('show_numbers', 4)
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
            grid_displacement = self.fem.deform(
                in_plane_thickness=thickness,
                optimization=True,
            )

            # TODO: gradient is different when there is input and output involved 
            # (i.e. self.fem.output_displacement is not None)

            grid_displacement_the_other = self.fem.virtual_deform(
                in_plane_thickness=thickness,
            )

            # Compute strain energy based on current thickness setting
            strain_energy = self.fem.grid.compute_strain_energy(
                grid_displacement=grid_displacement,
            )
            cross_strain_energy = self.fem.grid.compute_strain_energy(
                grid_displacement=grid_displacement,
                grid_displacement_the_other=grid_displacement_the_other,
            )

            # Compute gradient of strain energy based on current thickness setting
            self.fem.grid.compute_gradient_of_stiffness_matrix(
                in_plane_thickness=thickness
            )
            gradient_of_strain_energy = self.fem.grid.compute_gradient_of_strain_energy(
                grid_displacement=grid_displacement,
            )
            gradient_of_cross_strain_energy = self.fem.grid.compute_gradient_of_strain_energy(
                grid_displacement=grid_displacement,
                grid_displacement_the_other=grid_displacement_the_other,
            )

            # FIXME: the gradient is not all the same sign
            gradient = - (
                (gradient_of_cross_strain_energy * strain_energy) -
                (cross_strain_energy * gradient_of_strain_energy)
            ) / (strain_energy**2)

            # Update the thickness based on the gradient for the next iteration
            thickness[...] = self.update_thickness(thickness, gradient)

            if plot_flag and i in plot_indicies:
                color = next(colors)
                ax = self.fem.grid.plot_links(
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
            ax = self.fem.grid.plot_links(
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
    
