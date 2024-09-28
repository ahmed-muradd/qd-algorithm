import jax
import mujoco
from mujoco import mjx
import mujoco.viewer as viewer

import mediapy as media
import matplotlib.pyplot as plt


XML = open('src/qutee.xml').read()


mj_model = mujoco.MjModel.from_xml_string(XML)
mj_data = mujoco.MjData(mj_model)
renderer = mujoco.Renderer(mj_model)

mjx_model = mjx.put_model(mj_model)
mjx_data = mjx.put_data(mj_model, mj_data)

print(mj_data.qpos, type(mj_data.qpos))
print(mjx_data.qpos, type(mjx_data.qpos), mjx_data.qpos.devices())



# enable joint visualization option:
scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

duration = 4  # (seconds)
framerate = 60  # (Hz)

jit_step = jax.jit(mjx.step)

frames = []
mujoco.mj_resetData(mj_model, mj_data)
mjx_data = mjx.put_data(mj_model, mj_data)
while mjx_data.time < duration:
  mjx_data = jit_step(mjx_model, mjx_data)
  if len(frames) < mjx_data.time * framerate:
    mj_data = mjx.get_data(mj_model, mjx_data)
    renderer.update_scene(mj_data, scene_option=scene_option)
    pixels = renderer.render()
    frames.append(pixels)

media.write_video(images=frames, fps=framerate, path='output/output.mp4')





# @jax.vmap
# def batched_step(vel):
#     mjx_data = mjx.make_data(mjx_model)
#     qvel = mjx_data.qvel.at[0].set(vel)
#     mjx_data = mjx_data.replace(qvel=qvel)
#     pos = mjx.step(mjx_model, mjx_data).qpos[0]
#     return pos

# vel = jax.numpy.arange(0.0, 1.0, 0.01)
# pos = jax.jit(batched_step)(vel)
# print(pos)
# viewer.launch(mjx_model)