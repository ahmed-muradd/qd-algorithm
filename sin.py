import numpy as np
import math


# simulation setup
def sine_controller(time, amp, freq, phase, offset):
    return amp * np.sin(freq*math.pi*2*(time + phase)) + offset

# simulation setup
def tanh_controller(time, amp, freq, phase, offset):
    return amp * np.tanh(4*math.sin(2*math.pi*(time + phase))) + offset


for i in range(120):
    print(sine_controller(i/60, 1, 1, -math.pi/2, 0))
    # print(tanh_controller(i/60, 10, 10, -math.pi/2, 0))



