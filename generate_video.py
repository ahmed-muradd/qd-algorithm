import numpy as np

import mujoco
import mujoco.viewer
import mediapy
from PIL import Image
import numpy as np

from helper_functions import tanh_controller

model = mujoco.MjModel.from_xml_path('qutee.xml')
data = mujoco.MjData(model)


def generate_video(parameters, duration, framerate):
    '''
    input:
    controllers is a 12X3 Matrix

    output:
    generates video of robot in directory
    '''

    parameters = np.reshape(parameters, (12, 3))

    renderer = mujoco.Renderer(model)

    # enable joint visualization option:
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True


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


# if ran as main, generate video
if __name__ == '__main__':
    # paste the parameters from the pickle file into the save_video function
    #list of 48 zeroes
    parameters = [0.0]*36
    # every 4.th element is the offset and should be 0.5
    for i in range(2, 36, 3):
        parameters[i] = 0.5


    parameters[0] = 0.1

    generate_video(parameters, 10, 60)