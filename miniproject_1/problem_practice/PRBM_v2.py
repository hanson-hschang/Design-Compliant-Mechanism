"""
Created on Jan. 17, 2024
@author: Heng-Sheng Hanson Chang
"""

import numpy as np
from scipy.optimize import minimize
from collections import defaultdict
import matplotlib.pyplot as plt

def direction(angle):
    return np.array([np.cos(angle), np.sin(angle)])

def position_translate(length, angle):
    return length * direction(angle)

def get_energy(coefficient, value):
    return np.sum(0.5 * coefficient * value**2)

class PRBM:
    GAMMA = 0.85
    SPRING_CONSTANT = 2.67

    def __init__(
            self,
            nodes,
            links, 
            youngs_modulus, 
            in_plane_thickness, 
            out_of_plane_thickness,
            boundary_constraints,
            external_loads,
        ):
        # Geometry
        self.number_of_nodes = len(nodes)
        self.nodes = np.array(nodes)
        self.number_of_links = len(links)
        self.links = np.array(links)
        self.length_of_links = self.get_initial_length_of_links()
        self.angle_of_links = self.get_initial_angle_of_links()
        self.extensible_length_of_links = PRBM.GAMMA * self.length_of_links
        self.inextensible_length_of_links = (1 - PRBM.GAMMA) * self.length_of_links / 2

        # create a list recording the angles of attached links to each node, to access or modify the angles
        # usage: angles_of_attached_links_to_each_node[node_number][link_number] = angle
        # this variable is only used in the constraints method 
        self.angles_of_attached_links_to_each_node = [defaultdict(float) for _ in range(len(self.nodes))]

        # Material properties
        self.youngs_modulus = self.compute(youngs_modulus)
        self.in_plane_thickness = self.compute(in_plane_thickness)
        self.out_of_plane_thickness = self.compute(out_of_plane_thickness)
        self.cross_sectional_area = self.out_of_plane_thickness * self.in_plane_thickness
        self.moment_of_inertia = self.out_of_plane_thickness * self.in_plane_thickness**3 / 12

        # Boundary conditions
        self.boundary_constraints = boundary_constraints
        self.external_loads = external_loads

        # Compute equivalent spring constant
        self.equivalent_rotational_spring_constant = self.get_rotational_spring_constant()
        self.equivalent_translational_spring_constant = self.get_translational_spring_constant()
        
        # Decision variables
        self.position_x = self.nodes[:, 0].copy()
        self.position_y = self.nodes[:, 1].copy()
        self.angle_theta = np.zeros(self.number_of_links)
        self.angle_0 = np.zeros(self.number_of_links)
        self.angle_1 = np.zeros(self.number_of_links)
        self.delta = np.zeros(self.number_of_links)
        
        # Bounds of decision variables
        self.bounds = np.zeros((len(self.collect_decision_variables()), 2))
        self.bounds[:, 0] = self.collect_decision_variables(
            self.position_x-100, 
            self.position_y-100,
            self.angle_theta - np.pi / 2,
            self.angle_0 - np.pi / 2,
            self.angle_1 - np.pi / 2,
            self.delta - np.min(self.length_of_links) / 2,
        )
        self.bounds[:, 1] = self.collect_decision_variables(
            self.position_x+100, 
            self.position_y+100,
            self.angle_theta + np.pi / 2,
            self.angle_0 + np.pi / 2,
            self.angle_1 + np.pi / 2,
            self.delta + np.min(self.length_of_links) / 2,
        )

        # Options
        self.options = dict(
            disp=True,
            maxiter=300,
        )
    
    def get_initial_length_of_links(self,):
        return np.array(
            [ np.linalg.norm(
                self.nodes[link[1], :] - self.nodes[link[0], :]
            ) for link in self.links]
        )
    
    def get_initial_angle_of_links(self,):
        return np.arctan2(
            self.nodes[self.links[:, 1], 1] - self.nodes[self.links[:, 0], 1],
            self.nodes[self.links[:, 1], 0] - self.nodes[self.links[:, 0], 0]
        ) 

    def compute(self, value):
        if isinstance(value, (int, float)):
            return np.ones(self.links.shape[0]) * value
        return np.array(value)

    def get_rotational_spring_constant(self,):
        return 2 * PRBM.SPRING_CONSTANT * PRBM.GAMMA * self.youngs_modulus * self.moment_of_inertia / self.length_of_links            
    
    def get_translational_spring_constant(self,):
        return self.youngs_modulus * self.cross_sectional_area / self.length_of_links
    
    def collect_decision_variables(self, *args):
        if args == ():            
            return np.concatenate((
                self.position_x,
                self.position_y,
                self.angle_theta,
                self.angle_0,
                self.angle_1,
                self.delta,
            ))
        else:
            return np.concatenate(args)

    def get_decision_variables(self, decision_variables):
        index = 0
        position_x = decision_variables[index:index+len(self.position_x)]

        index += len(self.position_x)
        position_y = decision_variables[index:index+len(self.position_y)]

        index += len(self.position_y)
        angle_theta = decision_variables[index:index+len(self.angle_theta)]

        index += len(self.angle_theta)
        angle_0 = decision_variables[index:index+len(self.angle_0)]

        index += len(self.angle_0)
        angle_1 = decision_variables[index:index+len(self.angle_1)]

        index += len(self.angle_1)
        delta = decision_variables[index:index+len(self.delta)]
        
        return (position_x, position_y, angle_theta, angle_0, angle_1, delta)

    def constraints(self, decision_variables):
        position_x, position_y, angle_theta, angle_0, angle_1, delta = self.get_decision_variables(decision_variables)

        # Kinematic constraint
        kinematic_constraint_equations = np.zeros(self.number_of_links * 2)
        for i, link in enumerate(self.links):
            kinematic_constraint_equations[2*i:2*i+2] = (
                np.array([position_x[link[0]], position_y[link[0]]])  +
                position_translate(
                    self.inextensible_length_of_links[i], 
                    self.angle_of_links[i] + angle_theta[i]
                ) +
                position_translate(
                    self.extensible_length_of_links[i] + delta[i], 
                    self.angle_of_links[i] + angle_theta[i] + angle_0[i]
                ) +
                position_translate(
                    self.inextensible_length_of_links[i], 
                    self.angle_of_links[i] + angle_theta[i] + angle_0[i] + angle_1[i]
                ) - 
                np.array([position_x[link[1]], position_y[link[1]]])
            )
            
            self.angles_of_attached_links_to_each_node[link[0]][i] = angle_theta[i]
            self.angles_of_attached_links_to_each_node[link[1]][i] = angle_theta[i] + angle_0[i] + angle_1[i]

        # No relative rotation at the interface between two links
        rotation_constraint_equations = []
        for angles_of_attached_links in self.angles_of_attached_links_to_each_node:
            # If more than one link attached to the same node, 
            # set every angle of each attached link to be the same as the first one.
            if len(angles_of_attached_links) > 1:
                angles = list(angles_of_attached_links.values())
                for angle in angles[1:]:
                    rotation_constraint_equations.append(angle - angles[0])
        rotation_constraint_equations = np.array(rotation_constraint_equations)
        
        # Boundary constraint
        boundary_constraint_equations = np.zeros(len(self.boundary_constraints))
        for i, boundary_constraint in enumerate(self.boundary_constraints):
            index, condition, value = boundary_constraint
            if condition == 'position_x':
                boundary_constraint_equations[i] = (
                    position_x[index] - self.nodes[index, 0] - value
                )
            elif condition == 'position_y':
                boundary_constraint_equations[i] = (
                    position_y[index] - self.nodes[index, 1] - value
                )
            elif condition == 'angle_theta':
                angles = list(self.angles_of_attached_links_to_each_node[index].values())
                boundary_constraint_equations[i] = angles[0] - value
                        
        return np.concatenate(
            (kinematic_constraint_equations, 
             rotation_constraint_equations,
             boundary_constraint_equations, 
            )
        )
    
    def objective(self, decision_variables):
        position_x, position_y, angle_theta, angle_0, angle_1, delta = self.get_decision_variables(decision_variables)
        
        energy = 0
        
        energy += get_energy(self.equivalent_rotational_spring_constant, angle_0)
        energy += get_energy(self.equivalent_rotational_spring_constant, angle_1)
        energy += get_energy(self.equivalent_translational_spring_constant, delta)
    
        for external_load in self.external_loads:
            index, condition, value = external_load
            if condition == 'direction_x':
                energy -= value * (position_x[index] - self.nodes[index, 0])
            elif condition == 'direction_y':
                energy -= value * (position_y[index] - self.nodes[index, 1])
        
        return energy
    
    def deform(self):
        res = minimize(
            self.objective,
            self.collect_decision_variables(),
            method='trust-constr',
            options=self.options,
            bounds=self.bounds,
            constraints=dict(type='eq', fun=self.constraints),
            tol=1e-10
        )
        self.position_x, self.position_y, self.angle_theta, self.angle_0, self.angle_1, self.delta = self.get_decision_variables(res.x)
        langrage_multiplier = res.v[0]

        self.reaction_load = []
        for i, boundary_constraint in enumerate(self.boundary_constraints):
            index, condition, _ = boundary_constraint
            if condition == 'position_x':
                condition = 'direction_x'
            elif condition == 'position_y':
                condition = 'direction_y'
            elif condition == 'angle_theta':
                condition = 'rotation_theta'
            value = -langrage_multiplier[i]
            for external_load in self.external_loads:
                if external_load[0] == index and external_load[1] == condition:
                    value = external_load[2] if external_load[2] != 0 else value
            self.reaction_load.append([index, condition, value])
            print(self.reaction_load[-1])
        return

    def plot(self, ax, **kwargs):

        for i, link in enumerate(self.links):
            positions = np.zeros((2, 4))
            positions[:, 0] = np.array(
                [self.position_x[link[0]], self.position_y[link[0]]]
            )
            positions[:, 1] = positions[:, 0] + position_translate(
                self.inextensible_length_of_links[i], 
                self.angle_of_links[i] + self.angle_theta[i]
            )
            positions[:, 2] = positions[:, 1] + position_translate(
                self.extensible_length_of_links[i] + self.delta[i], 
                self.angle_of_links[i] + self.angle_theta[i] + self.angle_0[i]
            )
            positions[:, 3] = np.array(
                [self.position_x[link[1]], self.position_y[link[1]]]
            )
        
            ax.plot(positions[0], positions[1], **kwargs)

def main():
    model = PRBM(
        nodes=[
            [0, 0],
            [10, 20],
            [40, 20],
            [40, 0]
        ],
        links=[
            [0, 1], 
            [1, 2], 
            [1, 3]
        ],
        youngs_modulus=2*1e4,
        in_plane_thickness=5,
        out_of_plane_thickness=5,
        boundary_constraints=[
            [0, 'position_x', 0],
            [0, 'position_y', 0],
            [0, 'angle_theta', 0],
            [2, 'position_x', 0],
            [2, 'angle_theta', 0],
            [3, 'position_x', 0],
            [3, 'position_y', 5],
            [3, 'angle_theta', 0],
        ],
        external_loads=[
            [3, 'direction_y', 0]
        ],
    )

    fig, ax = plt.subplots()
    model.plot(ax, color='C0', linestyle='--')

    model.deform()

    model.plot(ax, color='C0')
    ax.axis('equal')
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()