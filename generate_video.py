import numpy as np

import mujoco
import mujoco.viewer
import mediapy
from PIL import Image
import numpy as np

from helper_functions import tanh_controller

model = mujoco.MjModel.from_xml_path('qutee.xml')
data = mujoco.MjData(model)


def generate_video(parameters, duration=10, framerate=60):
    '''
    input:
    controllers is a 12X3 Matrix

    output:
    generates video of robot in directory
    '''
    print("Creating video of simulation...")
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

    frames = []
    mujoco.mj_resetData(model, data)
    

    # run each frame on simulation
    while data.time < duration:
        controllers = []
        
        # applies sine wave to each parameter
        for row in parameters:
            controllers.append(tanh_controller(data.time, row[0], row[1], row[2]))

        data.ctrl = controllers
        mujoco.mj_step(model, data)

        # creates a video
        if len(frames) < data.time * framerate:
            renderer.update_scene(data,camera=camera, scene_option=scene_option)
            pixels = renderer.render()
            frames.append(pixels)

    mediapy.write_video("output/qutee.mp4", frames, fps=framerate)

    renderer.close()
    print("Video generated!")


# if ran as main, generate video
if __name__ == '__main__':
    # paste the parameters from the pickle file into the save_video function
    #list of 48 zeroes
    parameters = [0.0]*36
    # every 3.th element is the offset and should be 0.5
    for i in range(2, 36, 3):
        parameters[i] = 0.5


    parameters[5] = 0.00001
    parameters[6] = 1
    parameters[8] = 0.5

    generate_video(parameters, 10, 60)