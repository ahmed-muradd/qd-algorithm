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

duration = 60   # (seconds)
framerate = 60  # (Hz)



def save_video(parameters):
    '''
    input:
    controllers is a 12X4 Matrix

    output:
    saves video of robot in directory
    '''

    parameters = np.reshape(parameters, (12, 4))

    renderer = mujoco.Renderer(model)

    # enable joint visualization option:
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True


    frames = []
    mujoco.mj_resetData(model, data)
    
    fused = np.zeros((12, 5))
    fused[:, 1:] = parameters
    
    # run each frame on simulation
    while data.time < duration:
        controllers = []

        fused[:, 0] = data.time
        
        # applies sine wave to each parameter
        for row in fused:
            controllers.append(sine_controller(row[0], row[1], row[2], row[3], row[4]))

        data.ctrl = controllers
        mujoco.mj_step(model, data)

        # creates a video
        if len(frames) < data.time * framerate:
            renderer.update_scene(data, scene_option=scene_option)
            pixels = renderer.render()
            frames.append(pixels)  

    # Simulate and display video with increased size.
    bigger_frames = []
    for frame in frames: 
        image = Image.fromarray(frame)
        bigger_image = image.resize((1280, 720))
        bigger_frames.append(np.array(bigger_image))

    mediapy.write_video("qutee.mp4", bigger_frames, fps=framerate)

    renderer.close()



def eval_fn(parameters):
    """
    input:
        input is a 48 element array of control parameters

    output:
        (fitness, feauture0, feature1)
        fitness is the score of the controller
        feature 0 is ?
        feature 1 is ?
    """

    parameters = np.reshape(parameters, (12, 4))

    # simulation part
    mujoco.mj_resetData(model, data)
    #start position of robot
    body_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    initial_position = np.copy(data.xpos[body_index])   

    fused = np.zeros((12, 5))
    fused[:, 1:] = parameters

    while data.time < duration:        
        controllers = []

        fused[:, 0] = data.time
        
        # applies sine wave to each parameter
        for row in fused:
            controllers.append(sine_controller(row[0], row[1], row[2], row[3], row[4]))

        data.ctrl = controllers
        mujoco.mj_step(model, data)


    # where the robot stopped
    end_position = np.copy(data.xpos[body_index])
    #calculating fitness
    x, y, z = end_position - initial_position
    fitness = x**2 + y**2 + z**2

    # Compute the features
    feature0 = x
    feature1 = y
    feature2 = z
    feature3 = 0

    return (fitness,), (feature0, feature1, feature2, feature3)



if __name__ == "__main__":
    # Create container and algorithm. Here we use MAP-Elites, by illuminating a Grid container by evolution.
    grid = containers.Grid(shape=(16,16,16,16), max_items_per_bin=1, fitness_domain=((0, 2.),), features_domain=((-2, 2), (-2, 2), (-2, 2), (-2, 2)))
    algo = algorithms.RandomSearchMutPolyBounded(grid, budget=100, batch_size=10,
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

    save_video(best)