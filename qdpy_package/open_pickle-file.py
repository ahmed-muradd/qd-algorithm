import pickle, os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from generate_video import generate_video

pickle_path = "output3/final.p"
# generate vide of the nth best individual, ordered by fitness
generate_video_of_nth_best = 1

if not os.path.exists(pickle_path):
    raise FileNotFoundError("The file 'output/final.p' does not exist. Try running qdpy_exmaple.py first")


with open(pickle_path, "rb") as f:
    data = pickle.load(f)
# ``data`` is now a dictionary containing all results, including the final container, all solutions, the algorithm parameters, etc.

grid = data['container']
# print(grid.best)
# print(grid.best.fitness)
# print(grid.best.features)


print("\n\nthis run had a budget of ", data["budget"], " evalutions")
print("the best individual has a fitness of ", grid.best.fitness)
print("The batch size was ", data["batch_size"])






sorted_fitness = sorted(
    [(grid.fitness[s][0].values[0] if len(grid.fitness[s]) != 0 else 0, s) for s in grid.fitness],
    reverse=True,
    key=lambda item: item[0]
)

if sorted_fitness[generate_video_of_nth_best][0] == 0:
    print("The {generate_video_of_nth_best}. best individual dont exist")
    sys.exit(0)


grid_position = sorted_fitness[generate_video_of_nth_best][1]
print(f"\n----- print the individual in grid postion: {grid_position}, with fitness {sorted_fitness[generate_video_of_nth_best][0]} -----")
print(grid.features[grid_position])
print(grid.solutions[grid_position])

generate_video(grid.solutions[grid_position], 10)
