import numpy as np, math


# Function to convert a quaternion to Euler angles (roll, pitch, yaw)
def quat_to_rpy(quat):
    w, x, y, z = quat
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.sign(sinp) * np.pi / 2  # Use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])



# simulation setup
def sine_controller(time, amp, freq, phase, offset):
    return amp*1.507 * np.sin(freq*time + phase) + offset

# simulation setup
def tanh_controller(time, amp, phase, offset):
    amp = amp * math.pi/2
    # offset 0.5 is no offset
    # offset of 0 is -pi offset
    # offset of 1 is pi offset
    offset = 2*(offset-0.5) * math.pi/2

    return amp * np.tanh(4*math.sin(2*math.pi*(time + phase))) + offset





if __name__ == "__main__":
    for i in range(120):
        # print(sine_controller(i/60, 1, 10, 0, 0))
        print(tanh_controller(i/60, 0.01, 0, 0.5))