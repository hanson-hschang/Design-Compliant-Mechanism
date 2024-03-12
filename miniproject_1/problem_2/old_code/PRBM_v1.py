"""
Created on Feb. 17, 2024
@author: Heng-Sheng Hanson Chang
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

gamma = 0.85
K_equivalent = 2.65

def get_spring_K(youngs_modulus, second_moment_of_area, length):
    return gamma * K_equivalent * youngs_modulus * second_moment_of_area / length

def spring_energy(spring_K, angle):
    return 0.5 * spring_K * (angle**2)

def direction(angle):
    return np.array([np.cos(angle), np.sin(angle)])

def position_translate(length, angle):
    return length * direction(angle)

def get_beam_length(length_1, length_2, theta_2):
    return length_1, length_2 / np.cos(theta_2)

def compute_positions(
        beam_length_1,
        beam_length_2,
        theta_2,
        angle_10,
        angle_12,
        angle_21,
        angle_23
    ):

    positions = np.zeros((2, 7))

    angle = np.pi / 2
    positions[:, 1] = positions[:, 0] + (
        position_translate(beam_length_1 * (1-gamma) / 2, angle)
    )
    
    angle = angle + angle_10
    positions[:, 2] = positions[:, 1] + (
        position_translate(beam_length_1 * gamma, angle)
    )
    
    angle = angle + angle_12
    positions[:, 3] = positions[:, 2] + (
        position_translate(beam_length_1 * (1-gamma) / 2, angle)
    )

    angle = angle - (np.pi/2 - theta_2)
    positions[:, 4] = positions[:, 3] + (
        position_translate(beam_length_2 * (1-gamma) / 2, angle)
    )

    angle = angle + angle_21
    positions[:, 5] = positions[:, 4] + (
        position_translate(beam_length_2 * gamma, angle)
    )

    angle = angle + angle_23
    positions[:, 6] = positions[:, 5] + (
        position_translate(beam_length_2 * (1-gamma) / 2, angle)
    )

    return positions

def compute_endpoint_position(
        beam_length_1,
        beam_length_2,
        theta_2,
        angle_10,
        angle_12,
        angle_21,
        angle_23
    ):
    
    positions = compute_positions(
        beam_length_1,
        beam_length_2,
        theta_2,
        angle_10,
        angle_12,
        angle_21,
        angle_23
    )

    return positions[:, -1]

class DecisionVariables:
    def __init__(self, decision_variables):
        self.angle_10 = decision_variables[0]
        self.angle_12 = decision_variables[1]
        self.angle_21 = decision_variables[2]
        self.angle_23 = decision_variables[3]
        self.displacement = decision_variables[4]

    def __str__(self):
        return (
            " angle_10: " + str(self.angle_10) +
            "\n angle_12: " + str(self.angle_12) +
            "\n angle_21: " + str(self.angle_21) +
            "\n angle_23: " + str(self.angle_23) +
            "\n displacement: " + str(self.displacement)
        )

def objective(
        decision_variables, 
        length_1, 
        length_2, 
        theta_2, 
        height_1, 
        height_2, 
        youngs_modulus, 
        thickness, 
        force
    ):
    second_moment_of_area_1 = thickness * (height_1**3)
    second_moment_of_area_2 = thickness * (height_2**3)
    beam_length_1, beam_length_2 = get_beam_length(
        length_1, length_2, theta_2
    )
    spring_K_1 = get_spring_K(youngs_modulus, second_moment_of_area_1, beam_length_1/2.)
    spring_K_2 = get_spring_K(youngs_modulus, second_moment_of_area_2, beam_length_2/2.)

    dv = DecisionVariables(decision_variables)

    return (
        spring_energy(spring_K_1, dv.angle_10) +
        spring_energy(spring_K_1, dv.angle_12) +
        spring_energy(spring_K_2, dv.angle_21) +
        spring_energy(spring_K_2, dv.angle_23) -
        force / 2. * dv.displacement
    )

def constraint(
        decision_variables, 
        length_1, 
        length_2, 
        theta_2, 
        height_1, 
        height_2, 
        youngs_modulus, 
        thickness, 
        force
    ):
    
    dv = DecisionVariables(decision_variables)

    beam_length_1, beam_length_2 = get_beam_length(
        length_1, length_2, theta_2
    )

    endpoint_position = compute_endpoint_position(
        beam_length_1,
        beam_length_2,
        theta_2,
        dv.angle_10,
        dv.angle_12,
        dv.angle_21,
        dv.angle_23
    )

    endpoint_initial_position = np.array(
        [length_2, length_1+length_2*np.tan(theta_2)]
    )

    endpoint_constraint = endpoint_position - (
        endpoint_initial_position + position_translate(dv.displacement, np.pi/2)
    )
    return endpoint_constraint    

class Parameters:
    def __init__(self,):
        self.length_1 = 25 * 1e-3
        self.length_2 = 80 * 1e-3
        self.theta_2 = 20 * np.pi / 180
        self.height_1 = 5 * 1e-3
        self.height_2 = 2 * 1e-3
        self.youngs_modulus = 2 * 1e6
        self.thickness = 5 * 1e-3
        self.force = 0.0

    def set_force(self, force):
        self.force = force

    def get(self,):
        return (
            self.length_1, 
            self.length_2, 
            self.theta_2, 
            self.height_1, 
            self.height_2, 
            self.youngs_modulus, 
            self.thickness, 
            self.force
        )


def run_force_length_relation(parameters):
    bounds = [
        (-np.pi, np.pi),
        (-np.pi, np.pi),
        (-np.pi, np.pi),
        (-np.pi, np.pi),
        (-1., 1.)
        ]
    initial_decision_variables = np.array([0., 0., 0., 0., 0.])
        
    forces = [0]
    displacements = [0]
    force = -0.01
    continue_flag = True
    while continue_flag:
        
        parameters.set_force(force)
        result = minimize(
            objective, 
            initial_decision_variables, 
            method='SLSQP', 
            args=parameters.get(), 
            bounds=bounds, 
            constraints={
                'type': 'eq', 
                'fun': constraint,
                'args': parameters.get()  
            },
            tol=1e-9
            )

        dv = DecisionVariables(result.x)
        initial_decision_variables = result.x.copy()
        displacement = dv.displacement

        # print(force, displacement)

        if np.abs(displacement)>(30*1e-3):
            if displacements[-1]>(29.5*1e-3):
                continue_flag = False
            new_force = (forces[-1] + force) / 2
            if np.abs(new_force-force) < 1e-6:
                displacements.append(displacement)
                forces.append(force)
                continue_flag = False
            force = new_force
            continue

        displacement_diff_ratio = np.abs(displacement-displacements[-1]) / 1e-3
        if displacement_diff_ratio > 1:
            new_force = (forces[-1] + force) / 2
            if np.abs(new_force-force) < 1e-6:
                displacements.append(displacement)
                forces.append(force)
                continue_flag = False
        else:
            displacements.append(displacement)
            forces.append(force)
            force -= 0.01
    return forces, displacements

def main():


    

    
    # plt.figure(1)
    # for length_1 in [25, 20, 15, 10, 5]:
    #     parameters = Parameters()
    #     parameters.length_1 = length_1 * 1e-3
    #     print("length_1", length_1)
    #     forces, displacements = run_force_length_relation(parameters)
    #     plt.plot(-np.array(displacements)*1e3, -np.array(forces), label=str(length_1))
    # plt.title('length_1 [mm]')
    # plt.legend()
    
    
    # plt.figure(2)
    # for theta_2 in [20, 25, 30]:
    #     parameters = Parameters()
    #     parameters.theta_2 = theta_2 * np.pi/180
    #     print("theta_2", theta_2)
    #     forces, displacements = run_force_length_relation(parameters)
    #     plt.plot(-np.array(displacements)*1e3, -np.array(forces), label=str(theta_2))
    # plt.title('theta_2 [deg]')
    # plt.legend()

    # plt.figure(3)
    # for height_1 in [5,6,7,8,9,10]:
    #     parameters = Parameters()
    #     parameters.height_1 = height_1 * 1e-3
    #     print("height_1", height_1)
    #     forces, displacements = run_force_length_relation(parameters)
    #     plt.plot(-np.array(displacements)*1e3, -np.array(forces), label=str(height_1))
    # plt.title('height_1 [mm]')
    # plt.legend()

    plt.figure(4)
    parameters = Parameters()
    print("default")
    forces, displacements = run_force_length_relation(parameters)
    plt.plot(-np.array(displacements)*1e3, -np.array(forces), label='default')

    parameters = Parameters()
    parameters.length_1 = 5 * 1e-3
    parameters.theta_2 = 30 * np.pi/180
    parameters.height_1 = 10 * 1e-3
    print("optimized")
    forces, displacements = run_force_length_relation(parameters)
    plt.plot(-np.array(displacements)*1e3, -np.array(forces), label='optimized')

    plt.title('optimized')
    plt.legend()

    plt.show() 






if __name__ == "__main__":
    main()