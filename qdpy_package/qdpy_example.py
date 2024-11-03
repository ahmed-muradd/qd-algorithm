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
import sys, os, mpmath, numpy as np
import mujoco, mujoco.viewer

# import helper functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helper_functions import quat_to_rpy, tanh_controller
from generate_video import generate_video
from mjx_qdpy_example import eval_batch_fn


model = mujoco.MjModel.from_xml_path('qutee.xml')
data = mujoco.MjData(model)
output_path = "output"
duration = 10   # (seconds)

# Check if the logs folder exists, if not, create it
if not os.path.exists(output_path):
    os.makedirs(output_path)


def is_leg_in_contact(data, leg_geom_name="leg_0_3_geom"):
    # Get the geom IDs for the leg and ground
    ground_geom_id = mujoco.mj_name2id(data.model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    leg_geom_id = mujoco.mj_name2id(data.model, mujoco.mjtObj.mjOBJ_GEOM, leg_geom_name)

    # Check each contact
    for contact in data.contact[:data.ncon]:
        # Check if the contact involves the ground and the specified leg
        if (contact.geom1 == ground_geom_id and contact.geom2 == leg_geom_id) or \
           (contact.geom1 == leg_geom_id and contact.geom2 == ground_geom_id):
            return True  # Contact found
    
    return False  # No contact



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
    mujoco.mj_step(model, data) # step to get initial position and rotation

    #start position of robot
    body_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    initial_position = np.copy(data.xpos[body_index])
    rotation_matrix = data.xmat[body_index].reshape(3, 3)

    #initial roll pitch and yaw
    quaternion = data.xquat[body_index]
    prev_rpy = quat_to_rpy(quaternion)
    rpy_values = []

    # legs to check for contact
    legs = ["leg_0_3_geom", "leg_2_3_geom", "leg_3_3_geom", "leg_5_3_geom"]
    contact_times = {leg: 0.0 for leg in legs}

    while data.time < duration:
        controllers = []
        
        # applies sine wave to each parameter
        for row in parameters:
            controllers.append(tanh_controller(data.time, row[0], row[1], row[2]))

        data.ctrl = controllers
        mujoco.mj_step(model, data)

        # # check if each leg is in contact with the ground
        # for leg in legs:
        #     if is_leg_in_contact(data, leg):
        #         contact_times[leg] += 1 / 500

        # Get the roll, pitch, and yaw SE with mpmath precision
        quaternion = data.xquat[body_index]
        this_rpy = quat_to_rpy(quaternion, prev_rpy)  # Pass prev_rpy for continuity
        rpy_values.append(this_rpy - prev_rpy)
        prev_rpy = this_rpy


    # Get SE of roll, pitch, yaw using mpmath operations
    rpy_values = [mpmath.fsum([mpmath.power(val[i], 2) for val in rpy_values]) for i in range(3)]
    roll_error, pitch_error, yaw_error = [float(err)*10 for err in rpy_values]

    # robot's z rotation
    body_z_axis_world = rotation_matrix[:, 2]
    # Check if the body is upside down by taking the dot product
    z_dot_product = np.dot(body_z_axis_world, np.array([0, 0, 1]))



    # get the end position of the robot
    end_position = np.copy(data.xpos[body_index])
    # calculating fitness, 0 if robot is upside down
    fitness = 0.0
    x, y, z = end_position - initial_position
    if z_dot_product > 0:
        fitness = x**2 + y**2


    # Compute the features
    # features = (x,y,z)
    # features = (roll, pitch, yaw)
    # features = (average_roll, average_pitch, average_yaw)
    features = (roll_error, pitch_error, yaw_error)
    # features = tuple(contact_times.values())

    return (fitness,), features



if __name__ == "__main__":

    # ask for number of simulations
    simulations = int(input("How many simulations do you want to run?: "))
    # Create container and algorithm. Here we use MAP-Elites, by illuminating a Grid container by evolution.
    grid = containers.Grid(shape=(10,10,10), max_items_per_bin=1, fitness_domain=((0., 100.),), features_domain=((0., 1.), (0., 1.), (0., 1.)))
    algo = algorithms.RandomSearchMutPolyBounded(grid, budget=simulations, batch_size=512,
            dimension=36, optimisation_task="maximization")

    # Create a logger to pretty-print everything and generate output data files
    logger = algorithms.TQDMAlgorithmLogger(algo, log_base_path=output_path)

    # Run illumination process !
    # If on mac os or linux, use "multiprocessing" instead of "none" to enable parallelism on cpu.
    # unsupported on windows
    if sys.platform == "win32":
        best = algo.optimise(eval_fn, batch_mode=False)
    else:
        with ParallelismManager("multiprocessing") as pMgr:
            # best = algo.optimise(eval_batch_fn, executor = pMgr.executor, batch_mode=True, send_several_suggestions_to_fn=True, max_nb_suggestions_per_call=128) # Disable batch_mode (steady-state mode) to ask/tell new individuals without waiting the completion of each batch
            best = algo.optimise(eval_fn, executor = pMgr.executor) # Disable batch_mode (steady-state mode) to ask/tell new individuals without waiting the completion of each batch

    # Print results info
    print("\n" + algo.summary())

    # Plot the results
    plots.default_plots_grid(logger, output_dir=output_path, fitness_domain=((0., best.fitness.values[0]),))

    print("\nAll results are available in the '%s' pickle file." % logger.final_filename)

    generate_video(best, duration, framerate=60, output_path=output_path)
