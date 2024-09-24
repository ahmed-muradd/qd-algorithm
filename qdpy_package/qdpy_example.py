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
import math, random, time
import numpy as np


import mujoco
import mujoco.viewer
import mediapy
from PIL import Image
import numpy as np


# simulation setup
def sine_controller(time, amp, freq, phase, offset):
    return amp * np.sin(freq*time + phase) + offset

model = mujoco.MjModel.from_xml_path('qutee.xml')
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model)

# enable joint visualization option:
scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

duration = 10   # (seconds)
framerate = 60  # (Hz)

frames = []
fulfilled = 0


   

def eval_fn(controller_parameters):
    """An example evaluation function. It takes an individual as input, and returns the pair ``(fitness, features)``, where ``fitness`` and ``features`` are sequences of scores."""
    """returns a score and features for two legs, where the features descibe hoe much each leg touches the ground."""


    # Destructure controller_parameters into a 2x2 matrix
    # Deconstruct controller_parameters into a 2x2 matrix of 4x4 matrices
    print(controller_parameters)
    reshaped_parameters = np.array(controller_parameters).reshape(2, 2, 3, 4)


    # simulation part
    mujoco.mj_resetData(model, data)
    while data.time < duration:
        # INPUT CONTROLLER HERE data.ctrl should be a list of 12 controllers
        
        # Initialize the control list for the 12 actuators
        ctrl_values = []
        
        # Loop over the 2x2 structure to get each leg's actuators
        for i in range(2):
            for j in range(2):
                # Get the 3 actuators (rows) for the current leg
                actuators = reshaped_parameters[i][j]
                
                # Loop through the actuators
                for actuator_params in actuators:
                    amplitude, frequency, phase, offset = actuator_params
                    # Generate the control signal using the sine_controller and append it to ctrl_values
                    ctrl_values.append(sine_controller(data.time, amplitude, frequency, phase, offset))
        
        # Assign the computed control values to data.ctrl (12 actuators in total)
        data.ctrl = ctrl_values
        mujoco.mj_step(model, data)


    # randomn fitness and features for now    
    # first front or back, then left or right, then the 3 actuators for each leg, then the 4 parameters for each actuator
    fitness = (- reshaped_parameters[0][0][1][0]**2 - reshaped_parameters[0][0][1][1]**2 + reshaped_parameters[0][0][1][2]**2 + reshaped_parameters[0][0][1][3]**2) * 10

    # Compute the features
    feature0 = reshaped_parameters[0][0][1][0]
    feature1 = reshaped_parameters[0][0][1][0] * random.uniform(0.5, 1.5)


    return (fitness,), (feature0, feature1)




if __name__ == "__main__":
    # Create container and algorithm. Here we use MAP-Elites, by illuminating a Grid container by evolution.
    grid = containers.Grid(shape=(10,10), max_items_per_bin=1, fitness_domain=((0, 10.),), features_domain=((0., 1.), (0., 1.)))
    algo = algorithms.RandomSearchMutPolyBounded(grid, budget=100, batch_size=4,
            dimension=48, optimisation_task="maximization")

    # Create a logger to pretty-print everything and generate output data files
    logger = algorithms.TQDMAlgorithmLogger(algo)

    # Run illumination process !
    with ParallelismManager("none") as pMgr:
        best = algo.optimise(eval_fn, executor = pMgr.executor, batch_mode=False) # Disable batch_mode (steady-state mode) to ask/tell new individuals without waiting the completion of each batch

    # Print results info
    print("\n" + algo.summary())

    # Plot the results
    plots.default_plots_grid(logger)

    print("\nAll results are available in the '%s' pickle file." % logger.final_filename)
    print(f"""
To open it, you can use the following python code:
    import pickle
    # You may want to import your own packages if the pickle file contains custom objects

    with open("{logger.final_filename}", "rb") as f:
        data = pickle.load(f)
    # ``data`` is now a dictionary containing all results, including the final container, all solutions, the algorithm parameters, etc.

    grid = data['container']
    print(grid.best)
    print(grid.best.fitness)
    print(grid.best.features)
    """)

# MODELINE "{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
