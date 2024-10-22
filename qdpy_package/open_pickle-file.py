import pickle, os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from generate_video import generate_video

pickle_path = "output/final.p"


if not os.path.exists(pickle_path):
    raise FileNotFoundError("The file 'output/final.p' does not exist. Try running qdpy_exmaple.py first")


with open(pickle_path, "rb") as f:
    data = pickle.load(f)
# ``data`` is now a dictionary containing all results, including the final container, all solutions, the algorithm parameters, etc.

grid = data['container']
print(grid.best)
print(grid.best.fitness)
print(grid.best.features)


grid_position = (4, 4)
print(f"\n\n----- print the individual in grid postion: {grid_position} -----")
print(grid.features[grid_position])
print(grid.solutions[grid_position])

generate_video(grid.best, 10)