import math
from qdpy import plots
from qdpy.containers import Grid
import sys, os, pickle, numpy as np
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer


output_path = "output4"

# import helper functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helper_functions import tanh_controller

def fitness(controller: list, duration: int = 10) -> float:

    controller = np.reshape(controller, (12, 3))
     # simulation part
    model = mujoco.MjModel.from_xml_path('alternative_qutee.xml')
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    body_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    rotation_matrix = data.xmat[body_index].reshape(3, 3)

    while data.time < duration:
        controllers = []
        
        # applies tanh wave to each parameter
        for i, row in enumerate(controller):
            if i % 3 == 0:  # Every third controller starting from 0
                controllers.append(tanh_controller(data.time, row[0], row[1], row[2]))
            else:
                controllers.append(tanh_controller(data.time, row[0], row[1], row[2], half_rotation=True))


        data.ctrl = controllers
        mujoco.mj_step(model, data)

    # get the end position of the robot
    end_position = np.copy(data.xpos[body_index])

    # robot's z rotation
    body_z_axis_world = rotation_matrix[:, 2]
    # Check if the body is upside down by taking the dot product
    z_dot_product = np.dot(body_z_axis_world, np.array([0, 0, 1]))


    #calculating fitness, 0 if robot is upside down
    fitness = 0.0
    x, y, z = end_position
    if z_dot_product > 0:
        fitness = x**2 + y**2

    return fitness
    

def simulate_reality(grid: Grid) -> None:
    # get the grid and its controllers
    controllers: dict = grid.solutions
    # reset grid to prepare for new values
    best = grid.best_fitness
    # set fitness for each controller in grid based on new simulated reality
    quality_copy = np.copy(grid.quality_array)
    
    counter = 1
    for index, controller in controllers.items():
        if len(controller):
            controller = controller[0]
            ft = fitness(controller)
            error = grid.quality_array[index][0]-ft
            
            if error < 0:
                error = 0.0

            quality_copy[index] = [error]
            grid.quality_array[index] = [ft]
            print(f"\r{counter*100/grid.filled_bins:.2f}%", end='')
            counter+=1        

    # plot the grid subplots for reality testing
    path: str = output_path + "/realityPerformance4.pdf"
    plots.plotGridSubplots(grid.quality_array[... ,0], path, plt.get_cmap("inferno"), 
                           grid.features_domain, fitnessBounds=(0., best[0]), nbTicks=None)
    
    # finding the highest error
    flat_array = quality_copy.flatten()
    filtered_array = [x for x in flat_array if not math.isnan(x)]
    max_value = np.max(filtered_array)
    path = output_path + "/realityError4.pdf"
    plots.plotGridSubplots(quality_copy[..., 0], path, plt.get_cmap("inferno"), 
                           grid.features_domain, fitnessBounds=(0., max_value), nbTicks=None)

if __name__=="__main__":
    print("Evaluating grid...")
    pickle_path = output_path + "/final.p"
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"The file {pickle_path} does not exist. Try running qdpy_exmaple.py first")
    
    # read from pickle file
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    grid: Grid = data['container']
    
    simulate_reality(grid)