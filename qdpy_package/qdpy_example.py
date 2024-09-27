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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helper_functions import quat_to_rpy


# simulation setup
def sine_controller(time, amp, freq, phase, offset):
    return amp * np.sin(freq*time + phase) + offset

model = mujoco.MjModel.from_xml_path('qutee.xml')
data = mujoco.MjData(model)

duration = 20   # (seconds)
framerate = 30  # (Hz)





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


# paste the parameters from the pickle file into the save_video function
# save_video([0.10402131760590794, 0.8837563230671379, 0.8515599923224331, 0.13152828703580544, 0.6116614877161808, 0.7472725467010611, 0.5770062905058564, 0.45296427378352044, 0.965559742165479, 0.14217685612645048, 0.0348159244688373, 0.25419219138783655, 0.9112833898511875, 0.6738509939699706, 0.337630717351328, 0.7731381543629358, 0.6826967686957197, 0.26346615524744965, 0.7852935247994226, 0.5716902432368031, 0.5616338826529005, 0.36602266560105834, 0.12054381453433927, 0.7030165484661227, 0.5415126956684571, 0.8000784072179238, 0.7025336373270709, 0.22150082317563569, 0.9268738130881258, 0.2760440597661006, 0.4446339588341546, 0.923528414575281, 0.8264378793246557, 0.6205072648968746, 0.9904717354790828, 0.2759187715851352, 0.942701011522203, 0.708342600522683, 0.5617398920871732, 0.18837575300240406, 0.20186414049123613, 0.07523816524848936, 0.6556777641346274, 0.18451272052549494, 0.9167128753742309, 0.08802407666157874, 0.7611802345927873, 0.635203574335457])

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
    rotation_matrix = data.xmat[body_index].reshape(3, 3)

    fused = np.zeros((12, 5))
    fused[:, 1:] = parameters

    rpy_values = []
    while data.time < duration:        
        controllers = []
        fused[:, 0] = data.time
        
        # applies sine wave to each parameter
        for row in fused:
            controllers.append(sine_controller(row[0], row[1], row[2], row[3], row[4]))

        data.ctrl = controllers
        mujoco.mj_step(model, data)

        # Get the roll, pitch, and yaw
        quaternion = data.xquat[body_index]
        roll, pitch, yaw = quat_to_rpy(quaternion)
        rpy_values.append([roll, pitch, yaw])

    # get average roll, pitch, yaw
    rpy_values = np.array(rpy_values)
    average_rpy = np.mean(rpy_values, axis=0)
    print("Average Roll:", average_rpy[0])
    print("Average Pitch:", average_rpy[1])
    print("Average Yaw:", average_rpy[2])



    # Get the roll, pitch, and yaw of the robot in the end position
    quaternion = data.xquat[body_index]
    roll, pitch, yaw = quat_to_rpy(quaternion)
    print(f"Roll: {roll} radians")
    print(f"Pitch: {pitch} radians")
    print(f"Yaw: {yaw} radians")


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
        fitness = x**2 + y**2 + z**2


    # Compute the features
    # features = (x,y,z)
    # features = (roll, pitch, yaw)
    # features = (average_rpy[0], average_rpy[1], average_rpy[2])
    features = (average_rpy[0], average_rpy[1], z)

    return (fitness,), features



if __name__ == "__main__":
    # Create container and algorithm. Here we use MAP-Elites, by illuminating a Grid container by evolution.
    grid = containers.Grid(shape=(100,100,20), max_items_per_bin=1, fitness_domain=((0, 1.),), features_domain=((-5, 5), (-5, 5), (0., 1.)))
    algo = algorithms.RandomSearchMutPolyBounded(grid, budget=4000, batch_size=20,
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