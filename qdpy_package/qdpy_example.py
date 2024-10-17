#!/usr/bin/env python3
#    This file is part of qdpy.
#
#    qdpy is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    qdpy is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with qdpy. If not, see <http://www.gnu.org/licenses/>.


"""
A simple example to illuminate a given evaluation function, returning a one dimensional fitness score and two feature descriptors.

This code will optimize a robot controller with 3 parameters, and will return the fitness score and the behavior of the robot
the fitness will be the sum of the square of the parameters, and the behavior will be the sum of the parameters divided by the sum of the parameters.

example from : https://gitlab.com/leo.cazenille/qdpy/-/blob/master/examples/custom_eval_fn.py
"""

from qdpy import algorithms, containers, plots
from qdpy.base import ParallelismManager
import math, random, time, sys, os
import numpy as np


import mujoco
import mujoco.viewer
import mediapy
from PIL import Image
import numpy as np

# import helper functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helper_functions import quat_to_rpy, tanh_controller
from generate_video import generate_video


model = mujoco.MjModel.from_xml_path('qutee.xml')
data = mujoco.MjData(model)

duration = 15   # (seconds)
framerate = 20  # (Hz)



def eval_fn(parameters):
    """
    input:
        input is a 36 element array of control parameters

    output:
        (fitness, feautures)
        fitness is the score of the controller
        features is hwo we define the behavior of the robot
    """

    parameters = np.reshape(parameters, (12, 3))

    # simulation part
    mujoco.mj_resetData(model, data)
    #start position of robot
    body_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    initial_position = np.copy(data.xpos[body_index])
    rotation_matrix = data.xmat[body_index].reshape(3, 3)


    rpy_values = []
    while data.time < duration:
        controllers = []
        
        # applies sine wave to each parameter
        for row in parameters:
            controllers.append(tanh_controller(data.time, row[0], row[1], row[2]))

        data.ctrl = controllers
        mujoco.mj_step(model, data)

        # Get the roll, pitch, and yaw
        quaternion = data.xquat[body_index]
        rpy_values.append(quat_to_rpy(quaternion))

    # get average roll, pitch, yaw
    rpy_values = np.array(rpy_values)
    average_roll, average_pitch, average_yaw = np.mean(rpy_values, axis=0)

    # Get the roll, pitch, and yaw of the robot in the end position
    quaternion = data.xquat[body_index]
    roll, pitch, yaw = quat_to_rpy(quaternion)

    # get the end position of the robot
    end_position = np.copy(data.xpos[body_index])

    # robot's z rotation
    body_z_axis_world = rotation_matrix[:, 2]
    # Check if the body is upside down by taking the dot product
    z_dot_product = np.dot(body_z_axis_world, np.array([0, 0, 1]))


    #calculating fitness, 0 if robot is upside down
    fitness = 0.0
    x, y, z = end_position - initial_position
    if z_dot_product > 0:
        fitness = x**2 + y**2


    # Compute the features
    # features = (x,y,z)
    # features = (roll, pitch, yaw)
    features = (average_roll, average_pitch, average_yaw)

    return (fitness,), features



if __name__ == "__main__":
    # Create container and algorithm. Here we use MAP-Elites, by illuminating a Grid container by evolution.
    grid = containers.Grid(shape=(5,5,5), max_items_per_bin=1, fitness_domain=((0, 0.6),), features_domain=((0., 0.3), (-2., 2.), (-3., 3.)))
    algo = algorithms.RandomSearchMutPolyBounded(grid, budget=100, batch_size=100,
            dimension=36, optimisation_task="maximization")

    # Create a logger to pretty-print everything and generate output data files
    logger = algorithms.TQDMAlgorithmLogger(algo)

    # Run illumination process !
    # If on mac os, use "multiprocessing" instead of "none" to enable parallelism on cpu.
    # Not checked of linux support multiprocessing.
    if sys.platform == "darwin":
        with ParallelismManager("multiprocessing") as pMgr:
            best = algo.optimise(eval_fn, executor = pMgr.executor, batch_mode=False)
    else:
        with ParallelismManager("none") as pMgr:
            best = algo.optimise(eval_fn, executor = pMgr.executor, batch_mode=False) # Disable batch_mode (steady-state mode) to ask/tell new individuals without waiting the completion of each batch

    # Print results info
    print("\n" + algo.summary())

    # Plot the results
    plots.default_plots_grid(logger)

    print("\nAll results are available in the '%s' pickle file." % logger.final_filename)

    generate_video(best, model, data, duration, framerate=60)