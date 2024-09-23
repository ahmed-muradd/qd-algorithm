import numpy as np

class SpiderRobotController:
    def __init__(self, num_legs=4, motors_per_leg=3):
        self.num_legs = num_legs
        self.motors_per_leg = motors_per_leg
        self.num_motors = self.num_legs * self.motors_per_leg
        self.time = 0.0  # Start time

    def set_parameters(self, amplitudes, phases, frequencies):
        self.amplitudes = np.array(amplitudes)
        self.phases = np.array(phases)
        self.frequencies = np.array(frequencies)

    def step(self, delta_time):
        self.time += delta_time
        outputs = []
        for i in range(self.num_motors):
            A = self.amplitudes[i]
            ω = self.frequencies[i]
            φ = self.phases[i]
            motor_output = A * np.sin(ω * self.time + φ)
            outputs.append(motor_output)
        return np.array(outputs)

# Example parameters
amplitudes = [1.0] * 12
phases = [0.0] * 12
frequencies = [2 * np.pi] * 12  # 1 Hz for simplicity

controller = SpiderRobotController()
controller.set_parameters(amplitudes, phases, frequencies)

# Simulate the controller for one second
for _ in range(100):
    motor_outputs = controller.step(0.01)
    print(motor_outputs)





def simulate_robot(controller):
    # Simulate the robot in the environment and compute fitness
    # Placeholder for actual simulation
    return np.random.rand()


from qdax import QDAlgorithm

def fitness_function(controller_params):
    # Unpack controller parameters (amplitudes, phases, frequencies)
    amplitudes = controller_params[:12]
    phases = controller_params[12:24]
    frequencies = controller_params[24:]
    
    # Simulate the robot using these parameters and return the fitness score
    controller.set_parameters(amplitudes, phases, frequencies)
    # Simulate the robot in the environment and compute fitness
    fitness = simulate_robot(controller)  # Placeholder for actual simulation
    return fitness

# Define QDAX optimization
qd_algorithm = QDAlgorithm(fitness_function=fitness_function,
                           num_parameters=36,  # 12 motors * 3 parameters (A, phase, frequency)
                           population_size=100,
                           generations=500)

# Run the algorithm
best_solutions = qd_algorithm.run()

