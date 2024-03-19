"""
Created on Mar. 19, 2024
@author: Heng-Sheng Hanson Chang
"""

import numpy as np
import matplotlib.pyplot as plt

from grid import Grid

def plot_optimization(
    grid: Grid, 
    thickness: np.ndarray,
):
    fig, ax = plt.subplots()
    ax = grid.plot(
        ax,
        alpha=0.05,
        color='grey',
    )
    ax.plot(
        [],[], 
        alpha=0.2,
        color='grey',
        label='initial'
    )
    ax = grid.plot(
        ax,
        in_plane_thickness=thickness, 
        color='grey'
    )
    ax.plot(
        [],[], 
        color='grey',
        label='optimized'
    )
    ax.axis('equal')
    ax.legend()
    fig.tight_layout()

def plot_deformation(
    grid: Grid, 
    thickness: np.ndarray,
    grid_displacement: np.ndarray,
):
    fig, ax = plt.subplots()
    ax = grid.plot(
        ax,
        in_plane_thickness=thickness, 
        color='grey',
    )
    ax.plot(
        [],[], 
        color='grey',
        label='optimized',
    )
    ax = grid.plot(
        ax,
        grid_displacement=grid_displacement,
        in_plane_thickness=thickness, 
        color='black',
    )
    ax.plot(
        [],[], 
        color='black',
        label='optimized - deformed',
    )
    ax.axis('equal')
    ax.legend()
    fig.tight_layout()