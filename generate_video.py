import numpy as np

import mujoco
import mujoco.viewer
import mediapy
from PIL import Image
import numpy as np

from helper_functions import tanh_controller

model = mujoco.MjModel.from_xml_path('qutee.xml')
data = mujoco.MjData(model)



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

    legs = ["leg_0_3_geom", "leg_2_3_geom", "leg_3_3_geom", "leg_5_3_geom"]
    contact_times = {leg: 0.0 for leg in legs}

    # run each frame on simulation
    while data.time < duration:
        controllers = []
        
        # applies sine wave to each parameter
        for row in parameters:
            controllers.append(tanh_controller(data.time, row[0], row[1], row[2]))

        data.ctrl = controllers
        mujoco.mj_step(model, data)

        # check if each leg is in contact with the ground
        for leg in legs:
            if is_leg_in_contact(data, leg):
                contact_times[leg] += 1 / 500

        # creates a video
        if len(frames) < data.time * framerate:
            renderer.update_scene(data,camera=camera, scene_option=scene_option)
            pixels = renderer.render()
            frames.append(pixels)
        

    # Print total contact time for each leg
    # print("Total contact time for each leg:")
    # for leg, time in contact_times.items():
    #     print(f"{leg}: {time:.2f} seconds")

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


    parameters[3] = 0.00000
    parameters[4] = 0

    generate_video(parameters, 10, 60)