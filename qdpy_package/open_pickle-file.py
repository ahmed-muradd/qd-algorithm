import pickle, os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from generate_video import generate_video


output_path = "output4"
# generate vide of the nth best individual, ordered by fitness
use_nth_best = True
nth_best = 0

if not os.path.exists(output_path + "/final.p"):
    raise FileNotFoundError("The file 'output/final.p' does not exist. Try running qdpy_exmaple.py first")


with open(output_path + "/final.p", "rb") as f:
    data = pickle.load(f)
# ``data`` is now a dictionary containing all results, including the final container, all solutions, the algorithm parameters, etc.

grid = data['container']
# print(grid.best)
# print(grid.best.fitness)
# print(grid.best.features)


print("\n\nthis run had a budget of ", data["budget"], " evalutions")
print("the best individual has a fitness: ", grid.best_fitness[0], ", features: ", grid.best_features, ", position: ", grid.best_index)
print("The batch size was ", data["batch_size"], "\n")




if use_nth_best:
    sorted_fitness = sorted(
        [(grid.fitness[s][0].values[0] if len(grid.fitness[s]) != 0 else 0, s) for s in grid.fitness],
        reverse=True,
        key=lambda item: item[0]
    )
    if sorted_fitness[nth_best][0] == 0:
        print(f"The {nth_best}. best individual dont exist")
        sys.exit(0)


    grid_position = sorted_fitness[nth_best][1]
    print(f"\n----- print the individual in grid postion: {grid_position}, with fitness {sorted_fitness[nth_best][0]} -----")
    print(grid.features[grid_position])
    print(grid.solutions[grid_position])
    generate_video(grid.solutions[grid_position], 10, output_path=output_path)

else:
    # can be used for testing
    populated_positions = [s for s in grid.fitness if len(grid.fitness[s]) != 0]
    print("Populated grid positions:")
    for position in populated_positions:
        print(position)
    
    # put in the grid position you want to generate a video of,
    # see terminal output too select from populated_positions
    generate_video(grid.solutions[0,0,0], 10, output_path=output_path)

# første er y akse
# andre er x akse
# tredje er z akse

