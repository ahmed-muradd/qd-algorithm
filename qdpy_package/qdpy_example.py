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

duration = 40   # (seconds)
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
# save_video([0.566916671961166, 0.6776912232030303, 0.5753615591094349, 0.9020601533608612, 0.2825093839562801, 0.7736458482566407, 0.9130684354289595, 0.40130423698816575, 0.46340324283335366, 0.1055399988078809, 0.15861006084671547, 0.026482324082639508, 0.900416847021696, 0.6158480316582525, 0.11979641696376442, 0.514144854077119, 0.49501012987506354, 0.20303273951794243, 0.6404393998106958, 0.9300480080634687, 0.6117361840655406, 0.6091352071251489, 0.569900708645388, 0.2552246325352725, 0.09805382198080981, 0.22981176529307346, 0.5324530560340079, 0.6088716155617009, 0.25387383963824717, 0.08483812942056967, 0.8241899668049313, 0.8342512127913532, 0.5099619339001565, 0.6968724122233714, 0.8309422815176906, 0.13684285566757282, 0.08300500706687597, 0.6037105984610244, 0.5481270523953921, 0.8806448587569466, 0.19697315451971742, 0.6255277658227066, 0.6385433746359366, 0.23521359051463564, 0.35586218852374374, 0.8748018576704387, 0.5857894861631077, 0.32300714471952696])


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
        fitness = x**2 + y**2 + z**2


    # Compute the features
    # features = (x,y,z)
    # features = (roll, pitch, yaw)
    # features = (average_roll, average_pitch, average_yaw)
    features = (z, average_roll, average_pitch)

    return (fitness,), features



if __name__ == "__main__":
    # Create container and algorithm. Here we use MAP-Elites, by illuminating a Grid container by evolution.
    grid = containers.Grid(shape=(10,50,50), max_items_per_bin=1, fitness_domain=((0, 0.6),), features_domain=((0., 0.3), (-2., 2.), (-3, 3)))
    algo = algorithms.RandomSearchMutPolyBounded(grid, budget=4000, batch_size=10,
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