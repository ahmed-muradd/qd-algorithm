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
import jax
from jax import numpy as jp
from qdpy import algorithms, containers, plots
from qdpy.base import ParallelismManager
import random, time, sys, os
import numpy as np
from random import random


import mujoco
import mujoco.viewer
from mujoco import mjx
import numpy as np

import time

# import helper functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helper_functions import *
from generate_video import generate_video



def setup_sim():
    batchsize = 512
    duration = 10
    # converts qutee's xml file to simulation classes
    mj_model = mujoco.MjModel.from_xml_path('qutee.xml')

    #options = mujoco.MjOptions()
    #options.solver = 1 


    mj_data = mujoco.MjData(mj_model)
    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.put_data(mj_model, mj_data)

    # creates a batch of mjx_data
    rng = jax.random.PRNGKey(0)
    rng = jax.random.split(rng, batchsize)
    batchify = jax.vmap(lambda rng: mjx_data)
    mjx_data = batchify(rng)

    return mjx_model, mjx_data, batchsize, duration




def eval_batch_fn(controller_batch=None):
    """
    input:
        controller_batch: a vector of size batch_size of 12X3 matrixes of controller parameters
        batch_size: total evaluations/simulations
        model: batch of mujoco's model class
        data: batch of mujoco's data class
        duration: simulation duration (s)

    output:
        (fitness, feautures)
        fitness: a vector of fitnesses of size N, where N is the Batch_size
        features:  a batch_size X 3 matrix of each robot's average roll pitch and yaw. (how we define the behavior of the robot)
    """

    model, data, batch_size, duration = setup_sim()


    if controller_batch == None:
        # create random controllers to be evaluated
        controller_batch = create_rand_batch(batch_size, model)
    else:
        batch_size = len(controller_batch)

    controller_batch = jp.reshape(controller_batch, (batch_size, 12, 3))

    # jax functions
    jit_tan = jax.jit(jax.vmap(tan_control_mjx, (0,0)))
    jit_step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))
    jit_rpy = jax.jit(jax.vmap(mjx_quat_to_rpy))

    # rotation matrix of each robot in batch
    rotation_matrix = data.xmat[:, 1]
    # log roll pitch and yaw for each robot
    rpy_values = []

    # measure time of evalation of batch
    timecalc = time.time()

    # run batch of simulations on gpu
    while data.time[0] < duration:    
        milestone = data.time[0]  
        # info every half second of simulation
        while (data.time[0] - milestone) < 0.1:
            # set angle of each actuator for each robot
            data = jit_tan(data, controller_batch)
            # move simulation forward
            data = jit_step(model, data)
            # Get the roll, pitch, and yaw
            quaternion = data.xquat[:, 1]
            rpy = jit_rpy(quaternion)
            rpy_values.append(rpy)

        print(f"Batch Progress: {(data.time[0] / duration) * 100:.2f}%")

    #calculate average of the logged roll, pitch and yaw for each robot
    rpy_values = np.array(rpy_values)
    features = np.zeros((batch_size, 3))

    features = np.mean(rpy_values, axis=0)
            
    # measured time
    print(f"Simulation of {batch_size} robots took {(time.time() - timecalc) / 60:.2f} minutes")

    # fitness evaluation
    final_pos = data.xpos[:, 1]
    # focus only on x, y position
    fitness = final_pos[:, 0]**2 + final_pos[:, 1]**2

    return data, fitness, features






def create_rand_batch(batch_size, model):
    '''
    generates a batch of random controllers
    '''
    controllers = []
    for j in range(batch_size):
        controllers.append([])
        for i in range(model.nu):
            a = random() * model.actuator_ctrlrange[i][1]
            dc = (model.actuator_ctrlrange[i][1] - a) * random()
            if random() > 0.5:
                dc = -dc
            controllers[j].append([a, random(), dc])
    return np.array(controllers)


if __name__ == "__main__":
    # jax.config.update('jax_platform_name', "cpu")
    print(jax.devices())

    mjx_data, fitness, features = eval_batch_fn()

    # best generate video of performing controller
    best_fitness = max(fitness)
    best_index = jp.where(fitness == best_fitness)[0]
    # best_controller = controller_batch[best_index]
    # generate_video(best_controller)


    