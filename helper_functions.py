#https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles

import numpy as np, math, mpmath
import jax
from jax import numpy as jp

mpmath.mp.dps = 64 # Set the precision to 64 decimal places


# Function to convert quaternion to continuous Euler angles
def quat_to_rpy(quat, prev_rpy=None):
    w, x, y, z = quat
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = mpmath.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = mpmath.sign(sinp) * mpmath.pi / 2
    else:
        pitch = mpmath.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = mpmath.atan2(siny_cosp, cosy_cosp)

    # Wrap angles for continuity if previous angles exist
    current_rpy = mpmath.matrix([roll, pitch, yaw])
    if prev_rpy is not None:
        for i in range(3):  # for roll, pitch, yaw
            diff = current_rpy[i] - prev_rpy[i]
            if diff > mpmath.pi:
                current_rpy[i] -= 2 * mpmath.pi
            elif diff < -mpmath.pi:
                current_rpy[i] += 2 * mpmath.pi
    
    return current_rpy



def mjx_quat_to_rpy(quat):
    w, x, y, z = quat
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = jp.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)  
    # Use jax.lax.cond for conditional logic
    pitch = jax.lax.cond(
        jp.abs(sinp) >= 1,
        lambda _: jp.sign(sinp) * jp.pi / 2,
        lambda _: jp.arcsin(sinp),
        operand=None
    )

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = jp.arctan2(siny_cosp, cosy_cosp)

    return jp.array([roll, pitch, yaw])



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

def tan_control_mjx(data, cont):
    new_data = data
    amp = cont[:, 0]*np.pi/2
    offset = 2*(cont[:, 2]-0.5) * np.pi/2
    ctrl = amp * jp.tanh(4*jp.sin(2*jp.pi*(data.time + cont[:, 1]))) + offset
    new_data = new_data.replace(ctrl=ctrl)
    return new_data



if __name__ == "__main__":
    for i in range(120):
        # print(sine_controller(i/60, 1, 10, 0, 0))
        print(tanh_controller(i/60, 0.01, 0, 0.5))