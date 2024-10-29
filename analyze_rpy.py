import numpy as np

import mujoco
import mujoco.viewer
import mediapy
from PIL import Image
import numpy as np

from helper_functions import *

model = mujoco.MjModel.from_xml_path('qutee.xml')
data = mujoco.MjData(model)


def analyze_rpy(parameters, duration=10):
    '''
    input:
    controllers is a 36X1 Matrix

    output:
    generates a few pictures and prints the error of each picture
    '''
    print("Creating images from simulation...")
    parameters = np.reshape(parameters, (12, 3))

    renderer = mujoco.Renderer(model, 600, 800)

    # enable joint visualization option:
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

    # Create and initialize the camera
    camera = mujoco.MjvCamera()
    # Set the camera to track the robot's base link
    camera.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'base_link')
    
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING  # Set camera type to tracking

    mujoco.mj_resetData(model, data)
    frame_rpy = []
    rpy_values = []
    quaternion = data.xquat[1]
    i_rpy = quat_to_rpy(quaternion)
    # run each frame on simulation
    while data.time < duration:
        controllers = []
        
        # applies sine wave to each parameter
        for row in parameters:
            controllers.append(tanh_controller(data.time, row[0], row[1], row[2]))

        data.ctrl = controllers
        mujoco.mj_step(model, data)

        # creates a video
        if len(frame_rpy) < data.time:
            renderer.update_scene(data,camera=camera, scene_option=scene_option)
            pixels = renderer.render()
            frame_rpy.append(pixels)
            # Get the roll, pitch, and yaw SE
            quaternion = data.xquat[1]
            rpy_values.append(quat_to_rpy(quaternion))

    rpy_values = np.array(rpy_values)
    rpy_values = np.square(rpy_values)
    np.set_printoptions(precision=2, suppress=True)
    for index, rpy in enumerate(rpy_values):
        print(f"{index}... initial rpy = {i_rpy}... frame with {rpy} -> E of {rpy-i_rpy}, SE of {(rpy-i_rpy)**2}")
    
    for num, image in enumerate(frame_rpy):    
        mediapy.write_image(f"output/analyzing/analyze{num}.png", image)
        num += 1

    renderer.close()
    print("images generated!")


# if ran as main, generate video
if __name__ == '__main__':
   
   parameters = np.array([
    0.5, 1, 0.0,    
    0.6, 1.2, 0.25,   
    0.4, 0.8, 0.5,   
    0.7, 1, 0.75,    
    0.3, 1.5, 0.0,   
    0.5, 1.1, 0.33,  
    0.6, 1.2, 0.50,  
    0.4, 0.9, 0.66, 
    0.5, 1.3, 0.25,
    0.6, 1.4, 0.33, 
    0.7, 1.0, 0.75,
    0.4, 1.6, 0.66
    ])
   
   analyze_rpy(parameters, 10)