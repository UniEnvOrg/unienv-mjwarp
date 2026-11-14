# mjwarp_adaptor

MuJoCo Warp adaptor for [UniEnv](https://github.com/UniEnvOrg/UniEnv)'s node system (package `mjwarp_adaptor`), the
counterpart of [genesis_adaptor](https://github.com/UniEnvOrg/genesis_adaptor) for Genesis. Tested with mujoco 4.0.0,
mujoco-warp 4.0.0 and warp-lang 1.19.0.

- `MjWarpWorld`: one MuJoCo model in `nworld` copies on the GPU. `world.tensor("qpos")` (or any other MuJoCo array
  name) is a torch view of that array, (nworld, ...): writing to it writes to the simulation. One `step()` is one
  physics step; it and the recomputation after a reset are recorded once as CUDA graphs and replayed. With
  `stream=wp.Stream()`, several worlds (e.g. one per scene) overlap on the GPU; `world.join()` before torch reads or
  writes the world's arrays again.
- Nodes over named parts of the model:

  | node | observation / action |
  |---|---|
  | `JointStateNode` | `joint_position`, `joint_velocity` (N, n) of hinge and slide joints |
  | `ActuatorControlNode` | action: the actuators' controls (N, n), within `ctrlrange` |
  | `BodyPoseNode` | `body_pose` (N, 7: position, wxyz), world frame |
  | `FreeBodyNode` | `body_pose` and `body_velocity` (N, 6) of a body on a free joint; `set_pose`, `set_force` |
  | `ContactForceNode` | `net_force` (N, bodies, 3): total contact force on each body, world frame |
  | `ContactPointsNode` | no observation; `contacts()`: the points and frames of one body's contacts, in its frame |

A reset is three calls, in UniEnv's order: `world.reset(mask=...)` puts the chosen worlds back to the model's default
state, the nodes (or the caller) write their start state through `world.tensor(...)`, and `world.after_reset()`
recomputes positions and contacts for all worlds. After `step()`, positions and contacts are those from before the
step's integration (MuJoCo's convention).

Package `mjwarp_adaptor_collection` holds specific robots and scenes (as `unienv_genesis_collection` does for Genesis):
the MANO and Wuji hands (`load_hand`), the hand + object scene (`build_model`, physics settings in `SimConfig`) and
`HandNode`, the hand's joint state and position targets.

## Usage

```python
import mujoco
from unienv_interface.world.env_composer import WorldEnv
from mjwarp_adaptor import ActuatorControlNode, JointStateNode, MjWarpWorld

world = MjWarpWorld(mujoco.MjModel.from_xml_path("robot.xml"), nworld=4096)
env = WorldEnv(world, [JointStateNode(world, "robot_state", ["joint1", "joint2"]),
                       ActuatorControlNode(world, "robot_control", ["act1", "act2"])])

context, obs, info = env.reset()
# obs = {"joint_position": ..., "joint_velocity": ...}  (one observation node: passed as-is)
obs, reward, terminated, truncated, info = env.step(controls)  # (4096, 2)
```

As everywhere in UniEnv, observations and actions are dicts keyed by node name, except that a single node's value is
passed as-is.

## Install and test

```bash
pip install --no-deps -e .
pytest  # needs a CUDA GPU
```
