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
import math, random, time, sys, os
from math import sin, sqrt, pi, tanh
import numpy as np
from random import random


import mujoco
import mujoco.viewer
from mujoco import mjx
import mediapy
from PIL import Image
import numpy as np

# import helper functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helper_functions import quat_to_rpy, tanh_controller
from generate_video import generate_video

def tan_control_mjx(data, cont):
    new_data = data
    amp = cont[:, 0]*pi/2
    offset = 2*(cont[:, 2]-0.5) * pi/2
    ctrl = amp * jp.tanh(4*jp.sin(2*pi*(time + cont[:, 1]))) + offset
    new_data = new_data.replace(ctrl=ctrl)
    return new_data

def eval_batch_fn(controller_batch, batch_size, generations, model, data, duration):
    """
    input:
        controller_batch: batch of controllers to be evaluated
        batch_size: total evaluations/simulations
        model: mujoco's model class
        data: mujoco's data class

    output:
        (fitness, feautures)
        fitness is the score of the controller
        features is hwo we define the behavior of the robot
    """

    # jax functions
    jit_tan = jax.jit(jax.vmap(tan_control_mjx,(0,0)))
    jit_step = jax.jit(jax.vmap(mjx.step,in_axes=(None, 0)))

    # measure time of evalation of batch
    timecalc = time.time()
    # simulation part
    # run batch of simulations on gpu
    while data.time[0] < duration:        
        data = jit_tan(data, controller_batch)
        data = jit_step(model, data)
    # measured time
    print(f'batshize: {batch_size}, xposlen: {data.xpos.__len__()}')
    print("Simulation of", batch_size, "robots took", (time.time() - timecalc) / 60, "minutes")

    return 1


def create_batch(batch_size, model):
    '''
    generates a batch of controllers
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
    duration = 10   # (seconds)
    fps = 60 
    batchsize = 20
    generations = 2
    # converts qutee's xml file to simulation classes
    mj_model = mujoco.MjModel.from_xml_path('qutee.xml')
    mj_data = mujoco.MjData(mj_model)
    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.put_data(mj_model, mj_data)

    # create random controllers to be evaluated
    controller_batch = create_batch(batchsize, mj_model)
    eval_batch_fn(controller_batch, batchsize, generations, mjx_model, mjx_data, duration)

    