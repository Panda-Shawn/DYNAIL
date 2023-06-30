import numpy as np
import gym
import tempfile
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.envs.mujoco import half_cheetah_v3 as half_cheetah
from examples.envs.dynamic_mjc.model_builder import MJCModel


DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}

DEFAULT_CAMERA_CONFIG_HUMANOID = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 2.0)),
    "elevation": -20.0,
}


class BrokenAntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(
        self,
        xml_file="ant.xml",
        ctrl_cost_weight=0.5,
        contact_cost_weight=5e-4,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.2, 1.0),
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
        broken_joint=3,
    ):  
        # Select broken joint
        self.broken_joint = broken_joint

        utils.EzPickle.__init__(**locals())

        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._contact_force_range = contact_force_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    @property
    def healthy_reward(self):
        return (
            float(self.is_healthy or self._terminate_when_unhealthy)
            * self._healthy_reward
        )

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def contact_forces(self):
        raw_contact_forces = self.sim.data.cfrc_ext
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    @property
    def contact_cost(self):
        contact_cost = self._contact_cost_weight * np.sum(
            np.square(self.contact_forces)
        )
        return contact_cost

    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        is_healthy = np.isfinite(state).all() and min_z <= state[2] <= max_z
        return is_healthy

    @property
    def done(self):
        done = not self.is_healthy if self._terminate_when_unhealthy else False
        return done

    def step(self, action):
        # set the action of broken joint as 0
        # action = action.copy()
        if self.broken_joint is not None:
            action[self.broken_joint] = 0

        xy_position_before = self.get_body_com("torso")[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("torso")[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost

        forward_reward = x_velocity
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward
        costs = ctrl_cost + contact_cost

        reward = rewards - costs
        done = self.done
        observation = self._get_obs()
        info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_contact": -contact_cost,
            "reward_survive": healthy_reward,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        contact_force = self.contact_forces.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        observations = np.concatenate((position, velocity, contact_force))

        return observations

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv
        )
        self.set_state(qpos, qvel)

        observation = self._get_obs()

        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)


class BrokenHalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(
        self,
        xml_file="half_cheetah.xml",
        forward_reward_weight=1.0,
        ctrl_cost_weight=0.1,
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
        broken_joint=0,
    ):  
        # Select broken joint
        self.broken_joint = broken_joint

        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight

        self._ctrl_cost_weight = ctrl_cost_weight

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        # set the action of broken joint as 0
        # action = action.copy()
        if self.broken_joint is not None:
            action[self.broken_joint] = 0

        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        done = False
        info = {
            "x_position": x_position_after,
            "x_velocity": x_velocity,
            "reward_run": forward_reward,
            "reward_ctrl": -ctrl_cost,
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv
        )

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)
    

XML1 = """
<mujoco model="arm3d">
  <compiler inertiafromgeom="true" angle="radian" coordinate="local"/>
  <option timestep="0.01" gravity="0 0 0" iterations="20" integrator="Euler" />

  <default>
    <joint armature='0.04' damping="1" limited="true"/>
    <geom friction=".8 .1 .1" density="300" margin="0.002" condim="1" contype="0" conaffinity="0"/>
  </default>

  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>

    <body name="r_shoulder_pan_link" pos="0 -0.6 0">
      <geom name="e1" type="sphere" rgba="0.6 0.6 0.6 1" pos="-0.06 0.05 0.2" size="0.05" />
      <geom name="e2" type="sphere" rgba="0.6 0.6 0.6 1" pos=" 0.06 0.05 0.2" size="0.05" />
      <geom name="e1p" type="sphere" rgba="0.1 0.1 0.1 1" pos="-0.06 0.09 0.2" size="0.03" />
      <geom name="e2p" type="sphere" rgba="0.1 0.1 0.1 1" pos=" 0.06 0.09 0.2" size="0.03" />
      <geom name="sp" type="capsule" fromto="0 0 -0.4 0 0 0.2" size="0.1" />
      <joint name="r_shoulder_pan_joint" type="hinge" pos="0 0 0" axis="0 0 1" range="-2.2854 1.714602" damping="1.0" />

      <body name="r_shoulder_lift_link" pos="0.1 0 0">
        <geom name="sl" type="capsule" fromto="0 -0.1 0 0 0.1 0" size="0.1" />
        <joint name="r_shoulder_lift_joint" type="hinge" pos="0 0 0" axis="0 1 0" range="-0.5236 1.3963" damping="1.0" />

        <body name="r_upper_arm_roll_link" pos="0 0 0">
          <geom name="uar" type="capsule" fromto="-0.1 0 0 0.1 0 0" size="0.02" />
          <joint name="r_upper_arm_roll_joint" type="hinge" pos="0 0 0" axis="1 0 0" range="-1.5 1.7" damping="0.1" />

          <body name="r_upper_arm_link" pos="0 0 0">
            <geom name="ua" type="capsule" fromto="0 0 0 0.4 0 0" size="0.06" />

            <body name="r_elbow_flex_link" pos="0.4 0 0">
              <geom name="ef" type="capsule" fromto="0 -0.02 0 0.0 0.02 0" size="0.06" />
              <joint name="r_elbow_flex_joint" type="hinge" pos="0 0 0" axis="0 1 0" range="-2.3213 0" damping="0.1" />

              <body name="r_forearm_roll_link" pos="0 0 0">
                <geom name="fr" type="capsule" fromto="-0.1 0 0 0.1 0 0" size="0.02" />
                <joint name="r_forearm_roll_joint" type="hinge" limited="true" pos="0 0 0" axis="1 0 0" damping=".1" range="-1.5 1.5"/>

                <body name="r_forearm_link" pos="0 0 0">
                  <geom name="fa" type="capsule" fromto="0 0 0 0.291 0 0" size="0.05" />

                  <body name="r_wrist_flex_link" pos="0.321 0 0">
                    <geom name="wf" type="capsule" fromto="0 -0.02 0 0 0.02 0" size="0.01" />
                    <joint name="r_wrist_flex_joint" type="hinge" pos="0 0 0" axis="0 1 0" range="-1.094 0" damping=".1" />

                    <body name="r_wrist_roll_link" pos="0 0 0">
                      <joint name="r_wrist_roll_joint" type="hinge" pos="0 0 0" limited="true" axis="1 0 0" damping="0.1" range="-1.5 1.5"/>
                      <body name="tips_arm" pos="0 0 0">
                        <geom name="tip_arml" type="sphere" pos="0.1 -0.1 0." size="0.01" />
                        <geom name="tip_armr" type="sphere" pos="0.1 0.1 0." size="0.01" />
                      </body>
                      <geom type="capsule" fromto="0 -0.1 0. 0.0 +0.1 0" size="0.02" contype="1" conaffinity="1" />
                      <geom type="capsule" fromto="0 -0.1 0. 0.1 -0.1 0" size="0.02" contype="1" conaffinity="1" />
                      <geom type="capsule" fromto="0 +0.1 0. 0.1 +0.1 0." size="0.02" contype="1" conaffinity="1" />
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>

    <body name="goal" pos="0.45 -0.05 -0.3230">
      <geom rgba="0 1 1 0.5" type="sphere" size="0.05 0.05 0.05"
       density="0.00001" conaffinity="0" contype="0"/>
       <joint name="goal_free_joint" type="free" limited="false"/>
    </body>
  </worldbody>

  <actuator>
    <motor joint="r_shoulder_pan_joint" ctrlrange="-2.0 2.0" ctrllimited="true" />
    <motor joint="r_shoulder_lift_joint" ctrlrange="-2.0 2.0" ctrllimited="true" />
    <motor joint="r_upper_arm_roll_joint" ctrlrange="-2.0 2.0" ctrllimited="true" />
    <motor joint="r_elbow_flex_joint" ctrlrange="-2.0 2.0" ctrllimited="true" />
    <motor joint="r_forearm_roll_joint" ctrlrange="-2.0 2.0" ctrllimited="true" />
    <motor joint="r_wrist_flex_joint" ctrlrange="-2.0 2.0" ctrllimited="true" />
    <motor joint="r_wrist_roll_joint" ctrlrange="-2.0 2.0" ctrllimited="true"/>
  </actuator>
</mujoco>
"""


class Reacher7DofEnv(mujoco_env.MujocoEnv, utils.EzPickle):
  """7DOF robotic reaching environment."""

  def __init__(self):
    self._tempfile = tempfile.NamedTemporaryFile(mode="w", suffix=".xml")
    self._tempfile.write(XML1)
    self._tempfile.flush()
    mujoco_env.MujocoEnv.__init__(self, self._tempfile.name, 5)
    utils.EzPickle.__init__(self)

  def step(self, a):
    vec_1 = self.get_body_com("goal") - self.get_body_com("tips_arm")

    reward_near = -np.linalg.norm(vec_1)
    reward_ctrl = -np.square(a).sum()
    reward = 0.1 * reward_ctrl + 0.5 * reward_near

    self.do_simulation(a, self.frame_skip)
    ob = self._get_obs()
    done = False
    return ob, reward, done, dict(reward_ctrl=reward_ctrl)

  def _get_obs(self):
    return np.concatenate([
        self.sim.data.qpos.flat[:7],
        self.sim.data.qvel.flat[:7],
    ]).astype(np.float32)

  def reset_model(self):
    qpos = self.init_qpos
    qvel = self.init_qvel + self.np_random.uniform(
        low=-0.005, high=0.005, size=self.model.nv)
    qpos[7:10] = 0
    qvel[7:] = 0
    self.set_state(qpos, qvel)
    return self._get_obs()


class BrokenReacherEnv(Reacher7DofEnv):
  """Variant of the 7DOF reaching environment with a broken joint.

  I implemented the BrokenReacherEnv before implementing the more general
  BrokenJoint wrapper. While in theory they should do the same thing, I haven't
  confirmed this yet, so I'm keeping BrokenReacherEnv separate for now.
  """

  def __init__(self, broken_joint=2, state_includes_action=False):
    self._broken_joint = broken_joint
    self._state_includes_action = state_includes_action
    super(BrokenReacherEnv, self).__init__()
    if state_includes_action:
      obs_dim = len(self.observation_space.low)
      action_dim = len(self.action_space.low)
      self._observation_space = gym.spaces.Box(
          low=np.full(obs_dim + action_dim, -np.inf, dtype=np.float32),
          high=np.full(obs_dim + action_dim, np.inf, dtype=np.float32),
      )

  def reset(self):
    s = super(BrokenReacherEnv, self).reset()
    a = np.zeros(len(self.action_space.low))
    if self._state_includes_action:
      s = np.concatenate([s, a])
    return s

  def step(self, action):
    # action = action.copy()
    if self._broken_joint is not None:
      action[self._broken_joint] = 0.0
    ns, r, done, info = super(BrokenReacherEnv, self).step(action)
    if self._state_includes_action:
      ns = np.concatenate([ns, action])
    return ns, r, done, info


XML2 = """
<mujoco model="cheetah">
  <compiler angle="radian" coordinate="local" inertiafromgeom="true" settotalmass="14"/>
  <default>
    <joint armature=".1" damping=".01" limited="true" solimplimit="0 .8 .03" solreflimit=".02 1" stiffness="8"/>
    <geom conaffinity="0" condim="3" contype="1" friction=".4 .1 .1" rgba="0.8 0.6 .4 1" solimp="0.0 0.8 0.01" solref="0.02 1"/>
    <motor ctrllimited="true" ctrlrange="-1 1"/>
  </default>
  <size nstack="300000" nuser_geom="1"/>
  <option gravity="0 0 -9.81" timestep="0.01"/>
  <asset>
    <texture builtin="gradient" height="100" rgb1="1 1 1" rgb2="0 0 0" type="skybox" width="100"/>
    <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
    <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d" width="100"/>
    <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane"/>
    <material name="geom" texture="texgeom" texuniform="true"/>
  </asset>
  <worldbody>
    <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
    <geom conaffinity="1" condim="3" material="MatPlane" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" size="40 40 40" type="plane"/>
    <body name="torso" pos="0 0 .7">
      <camera name="track" mode="trackcom" pos="0 -3 0.3" xyaxes="1 0 0 0 0 1"/>
      <joint armature="0" axis="1 0 0" damping="0" limited="false" name="rootx" pos="0 0 0" stiffness="0" type="slide"/>
      <joint armature="0" axis="0 0 1" damping="0" limited="false" name="rootz" pos="0 0 0" stiffness="0" type="slide"/>
      <joint armature="0" axis="0 1 0" damping="0" limited="false" name="rooty" pos="0 0 0" stiffness="0" type="hinge"/>
      <geom fromto="-.5 0 0 .5 0 0" name="torso" size="0.046" type="capsule"/>
      <geom axisangle="0 1 0 .87" name="head" pos=".6 0 .1" size="0.046 .15" type="capsule"/>
      <!-- <site name='tip'  pos='.15 0 .11'/>-->
      <body name="bthigh" pos="-.5 0 0">
        <joint axis="0 1 0" damping="6" name="bthigh" pos="0 0 0" range="-.52 1.05" stiffness="240" type="hinge"/>
        <geom axisangle="0 1 0 -3.8" name="bthigh" pos=".1 0 -.13" size="0.046 .145" type="capsule"/>
        <body name="bshin" pos=".16 0 -.25">
          <joint axis="0 1 0" damping="4.5" name="bshin" pos="0 0 0" range="-.785 .785" stiffness="180" type="hinge"/>
          <geom axisangle="0 1 0 -2.03" name="bshin" pos="-.14 0 -.07" rgba="0.9 0.6 0.6 1" size="0.046 .15" type="capsule"/>
          <body name="bfoot" pos="-.28 0 -.14">
            <joint axis="0 1 0" damping="3" name="bfoot" pos="0 0 0" range="-.4 .785" stiffness="120" type="hinge"/>
            <geom axisangle="0 1 0 -.27" name="bfoot" pos=".03 0 -.097" rgba="0.9 0.6 0.6 1" size="0.046 .094" type="capsule"/>
          </body>
        </body>
      </body>
      <body name="fthigh" pos=".5 0 0">
        <joint axis="0 1 0" damping="4.5" name="fthigh" pos="0 0 0" range="-1 .7" stiffness="180" type="hinge"/>
        <geom axisangle="0 1 0 .52" name="fthigh" pos="-.07 0 -.12" size="0.046 .133" type="capsule"/>
        <body name="fshin" pos="-.14 0 -.24">
          <joint axis="0 1 0" damping="3" name="fshin" pos="0 0 0" range="-1.2 .87" stiffness="120" type="hinge"/>
          <geom axisangle="0 1 0 -.6" name="fshin" pos=".065 0 -.09" rgba="0.9 0.6 0.6 1" size="0.046 .106" type="capsule"/>
          <body name="ffoot" pos=".13 0 -.18">
            <joint axis="0 1 0" damping="1.5" name="ffoot" pos="0 0 0" range="-.5 .5" stiffness="60" type="hinge"/>
            <geom axisangle="0 1 0 -.6" name="ffoot" pos=".045 0 -.07" rgba="0.9 0.6 0.6 1" size="0.046 .07" type="capsule"/>
          </body>
        </body>
      </body>
    </body>

    <geom name="obstacle" type="box" pos="-3 0 %f" size="1 10 10" rgba="0.2 0.5 0.2 1" conaffinity="1"/>

  </worldbody>
  <actuator>
    <motor gear="120" joint="bthigh" name="bthigh"/>
    <motor gear="90" joint="bshin" name="bshin"/>
    <motor gear="60" joint="bfoot" name="bfoot"/>
    <motor gear="120" joint="fthigh" name="fthigh"/>
    <motor gear="60" joint="fshin" name="fshin"/>
    <motor gear="30" joint="ffoot" name="ffoot"/>
  </actuator>
</mujoco>
"""


class HalfCheetahDirectionEnv(half_cheetah.HalfCheetahEnv):
  """Variant of half-cheetah that includes an obstacle."""

  def __init__(self, use_obstacle=True):
    super().__init__()
    self._tempfile = tempfile.NamedTemporaryFile(mode="w", suffix=".xml")
    if use_obstacle:
      obstacle_height = 1.0
    else:
      obstacle_height = -50
    self._tempfile.write(XML2 % (obstacle_height))
    self._tempfile.flush()
    mujoco_env.MujocoEnv.__init__(self, self._tempfile.name, 5)
    utils.EzPickle.__init__(self)
    self.observation_space = gym.spaces.Box(
        low=self.observation_space.low,
        high=self.observation_space.high,
        dtype=np.float32,
    )

  def step(self, action):
    xposbefore = self.sim.data.qpos[0]
    self.do_simulation(action, self.frame_skip)
    xposafter = self.sim.data.qpos[0]
    ob = self._get_obs()
    reward_ctrl = -0.1 * np.square(action).sum()
    reward_run = abs(xposafter - xposbefore) / self.dt
    reward = reward_ctrl + reward_run
    done = False
    return ob, reward, done, dict(
        reward_run=reward_run, reward_ctrl=reward_ctrl)

  def camera_setup(self):
    super(HalfCheetahDirectionEnv, self).camera_setup()
    self.camera._render_camera.distance = 5.0  # pylint: disable=protected-access


def ant_env(gear=150, eyes=True):
    mjcmodel = MJCModel('ant_maze')
    mjcmodel.root.compiler(inertiafromgeom="true", angle="degree", coordinate="local")
    mjcmodel.root.option(timestep="0.01", integrator="RK4")
    mjcmodel.root.custom().numeric(data="0.0 0.0 0.55 1.0 0.0 0.0 0.0 0.0 1.0 0.0 -1.0 0.0 -1.0 0.0 1.0",name="init_qpos")
    asset = mjcmodel.root.asset()
    asset.texture(builtin="gradient",height="100",rgb1="1 1 1",rgb2="0 0 0",type="skybox",width="100")
    asset.texture(builtin="flat",height="1278",mark="cross",markrgb="1 1 1",name="texgeom",random="0.01",rgb1="0.8 0.6 0.4",rgb2="0.8 0.6 0.4",type="cube",width="127")
    asset.texture(builtin="checker",height="100",name="texplane",rgb1="0 0 0",rgb2="0.8 0.8 0.8",type="2d",width="100")
    asset.material(name="MatPlane",reflectance="0.5",shininess="1",specular="1",texrepeat="60 60",texture="texplane")
    asset.material(name="geom",texture="texgeom",texuniform="true")

    default = mjcmodel.root.default()
    default.joint(armature=1, damping=1, limited='true')
    default.geom(friction=[1.5,0.5,0.5], density=5.0, margin=0.01, condim=3, conaffinity=0, rgba="0.8 0.6 0.4 1")

    worldbody = mjcmodel.root.worldbody()
    worldbody.light(cutoff="100",diffuse=[.8,.8,.8],dir="-0 0 -1.3",directional="true",exponent="1",pos="0 0 1.3",specular=".1 .1 .1")
    worldbody.geom(conaffinity=1, condim=3, material="MatPlane",name="floor",pos="0 0 0",rgba="0.8 0.9 0.8 1",size="40 40 40",type="plane")

    ant = worldbody.body(name='torso', pos=[0, 0, 0.75])
    ant.geom(name='torso_geom', pos=[0, 0, 0], size="0.25", type="sphere")
    ant.joint(armature="0", damping="0", limited="false", margin="0.01", name="root", pos=[0, 0, 0], type="free")

    if eyes:
        eye_z = 0.1
        eye_y = -.21
        eye_x_offset = 0.07
        # eyes
        ant.geom(fromto=[eye_x_offset,0,eye_z,eye_x_offset,eye_y,eye_z], name='eye1', size='0.03', type='capsule', rgba=[1,1,1,1])
        ant.geom(fromto=[eye_x_offset,0,eye_z,eye_x_offset,eye_y-0.02,eye_z], name='eye1_', size='0.02', type='capsule', rgba=[0,0,0,1])
        ant.geom(fromto=[-eye_x_offset,0,eye_z,-eye_x_offset,eye_y,eye_z], name='eye2', size='0.03', type='capsule', rgba=[1,1,1,1])
        ant.geom(fromto=[-eye_x_offset,0,eye_z,-eye_x_offset,eye_y-0.02,eye_z], name='eye2_', size='0.02', type='capsule', rgba=[0,0,0,1])
        # eyebrows
        ant.geom(fromto=[eye_x_offset-0.03,eye_y, eye_z+0.07, eye_x_offset+0.03, eye_y, eye_z+0.1], name='brow1', size='0.02', type='capsule', rgba=[0,0,0,1])
        ant.geom(fromto=[-eye_x_offset+0.03,eye_y, eye_z+0.07, -eye_x_offset-0.03, eye_y, eye_z+0.1], name='brow2', size='0.02', type='capsule', rgba=[0,0,0,1])

    front_left_leg = ant.body(name="front_left_leg", pos=[0, 0, 0])
    front_left_leg.geom(fromto=[0.0, 0.0, 0.0, 0.2, 0.2, 0.0], name="aux_1_geom", size="0.08", type="capsule")
    aux_1 = front_left_leg.body(name="aux_1", pos=[0.2, 0.2, 0])
    aux_1.joint(axis=[0, 0, 1], name="hip_1", pos=[0.0, 0.0, 0.0], range=[-30, 30], type="hinge")
    aux_1.geom(fromto=[0.0, 0.0, 0.0, 0.2, 0.2, 0.0], name="left_leg_geom", size="0.08", type="capsule")
    ankle_1 = aux_1.body(pos=[0.2, 0.2, 0])
    ankle_1.joint(axis=[-1, 1, 0], name="ankle_1", pos=[0.0, 0.0, 0.0], range=[30, 70], type="hinge")
    ankle_1.geom(fromto=[0.0, 0.0, 0.0, 0.4, 0.4, 0.0], name="left_ankle_geom", size="0.08", type="capsule")

    front_right_leg = ant.body(name="front_right_leg", pos=[0, 0, 0])
    front_right_leg.geom(fromto=[0.0, 0.0, 0.0, -0.2, 0.2, 0.0], name="aux_2_geom", size="0.08", type="capsule")
    aux_2 = front_right_leg.body(name="aux_2", pos=[-0.2, 0.2, 0])
    aux_2.joint(axis=[0, 0, 1], name="hip_2", pos=[0.0, 0.0, 0.0], range=[-30, 30], type="hinge")
    aux_2.geom(fromto=[0.0, 0.0, 0.0, -0.2, 0.2, 0.0], name="right_leg_geom", size="0.08", type="capsule")
    ankle_2 = aux_2.body(pos=[-0.2, 0.2, 0])
    ankle_2.joint(axis=[1, 1, 0], name="ankle_2", pos=[0.0, 0.0, 0.0], range=[-70, -30], type="hinge")
    ankle_2.geom(fromto=[0.0, 0.0, 0.0, -0.4, 0.4, 0.0], name="right_ankle_geom", size="0.08", type="capsule")

    back_left_leg = ant.body(name="back_left_leg", pos=[0, 0, 0])
    back_left_leg.geom(fromto=[0.0, 0.0, 0.0, -0.2, -0.2, 0.0], name="aux_3_geom", size="0.08", type="capsule")
    aux_3 = back_left_leg.body(name="aux_3", pos=[-0.2, -0.2, 0])
    aux_3.joint(axis=[0, 0, 1], name="hip_3", pos=[0.0, 0.0, 0.0], range=[-30, 30], type="hinge")
    aux_3.geom(fromto=[0.0, 0.0, 0.0, -0.2, -0.2, 0.0], name="backleft_leg_geom", size="0.08", type="capsule")
    ankle_3 = aux_3.body(pos=[-0.2, -0.2, 0])
    ankle_3.joint(axis=[-1, 1, 0], name="ankle_3", pos=[0.0, 0.0, 0.0], range=[-70, -30], type="hinge")
    ankle_3.geom(fromto=[0.0, 0.0, 0.0, -0.4, -0.4, 0.0], name="backleft_ankle_geom", size="0.08", type="capsule")

    back_right_leg = ant.body(name="back_right_leg", pos=[0, 0, 0])
    back_right_leg.geom(fromto=[0.0, 0.0, 0.0, 0.2, -0.2, 0.0], name="aux_4_geom", size="0.08", type="capsule")
    aux_4 = back_right_leg.body(name="aux_4", pos=[0.2, -0.2, 0])
    aux_4.joint(axis=[0, 0, 1], name="hip_4", pos=[0.0, 0.0, 0.0], range=[-30, 30], type="hinge")
    aux_4.geom(fromto=[0.0, 0.0, 0.0, 0.2, -0.2, 0.0], name="backright_leg_geom", size="0.08", type="capsule")
    ankle_4 = aux_4.body(pos=[0.2, -0.2, 0])
    ankle_4.joint(axis=[1, 1, 0], name="ankle_4", pos=[0.0, 0.0, 0.0], range=[30, 70], type="hinge")
    ankle_4.geom(fromto=[0.0, 0.0, 0.0, 0.4, -0.4, 0.0], name="backright_ankle_geom", size="0.08", type="capsule")

    actuator = mjcmodel.root.actuator()
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="hip_4", gear=gear)
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="ankle_4", gear=gear)
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="hip_1", gear=gear)
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="ankle_1", gear=gear)
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="hip_2", gear=gear)
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="ankle_2", gear=gear)
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="hip_3", gear=gear)
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="ankle_3", gear=gear)
    return mjcmodel


def angry_ant_crippled_two(gear=150):
    mjcmodel = MJCModel('ant_maze')
    mjcmodel.root.compiler(inertiafromgeom="true", angle="degree", coordinate="local")
    mjcmodel.root.option(timestep="0.01", integrator="RK4")
    mjcmodel.root.custom().numeric(data="0.0 0.0 0.55 1.0 0.0 0.0 0.0 0.0 1.0 0.0 -1.0 0.0 -1.0 0.0 1.0",name="init_qpos")
    asset = mjcmodel.root.asset()
    asset.texture(builtin="gradient",height="100",rgb1="1 1 1",rgb2="0 0 0",type="skybox",width="100")
    asset.texture(builtin="flat",height="1278",mark="cross",markrgb="1 1 1",name="texgeom",random="0.01",rgb1="0.8 0.6 0.4",rgb2="0.8 0.6 0.4",type="cube",width="127")
    asset.texture(builtin="checker",height="100",name="texplane",rgb1="0 0 0",rgb2="0.8 0.8 0.8",type="2d",width="100")
    asset.material(name="MatPlane",reflectance="0.5",shininess="1",specular="1",texrepeat="60 60",texture="texplane")
    asset.material(name="geom",texture="texgeom",texuniform="true")



    default = mjcmodel.root.default()
    default.joint(armature=1, damping=1, limited='true')
    default.geom(friction=[1.5,0.5,0.5], density=5.0, margin=0.01, condim=3, conaffinity=0, rgba="0.8 0.6 0.4 1")

    worldbody = mjcmodel.root.worldbody()

    worldbody.geom(conaffinity=1, condim=3, material="MatPlane",name="floor",pos="0 0 0",rgba="0.8 0.9 0.8 1",size="40 40 40",type="plane")
    worldbody.light(cutoff="100",diffuse=[.8,.8,.8],dir="-0 0 -1.3",directional="true",exponent="1",pos="0 0 1.3",specular=".1 .1 .1")


    ant = worldbody.body(name='torso', pos=[0, 0, 0.75])
    ant.geom(name='torso_geom', pos=[0, 0, 0], size="0.25", type="sphere")
    ant.joint(armature="0", damping="0", limited="false", margin="0.01", name="root", pos=[0, 0, 0], type="free")

    eye_z = 0.1
    eye_y = -.21
    eye_x_offset = 0.07
    # eyes
    ant.geom(fromto=[eye_x_offset,0,eye_z,eye_x_offset,eye_y,eye_z], name='eye1', size='0.03', type='capsule', rgba=[1,1,1,1])
    ant.geom(fromto=[eye_x_offset,0,eye_z,eye_x_offset,eye_y-0.02,eye_z], name='eye1_', size='0.02', type='capsule', rgba=[0,0,0,1])
    ant.geom(fromto=[-eye_x_offset,0,eye_z,-eye_x_offset,eye_y,eye_z], name='eye2', size='0.03', type='capsule', rgba=[1,1,1,1])
    ant.geom(fromto=[-eye_x_offset,0,eye_z,-eye_x_offset,eye_y-0.02,eye_z], name='eye2_', size='0.02', type='capsule', rgba=[0,0,0,1])
    # eyebrows
    ant.geom(fromto=[eye_x_offset-0.03,eye_y, eye_z+0.07, eye_x_offset+0.03, eye_y, eye_z+0.1], name='brow1', size='0.02', type='capsule', rgba=[0,0,0,1])
    ant.geom(fromto=[-eye_x_offset+0.03,eye_y, eye_z+0.07, -eye_x_offset-0.03, eye_y, eye_z+0.1], name='brow2', size='0.02', type='capsule', rgba=[0,0,0,1])




    front_left_leg = ant.body(name="front_left_leg", pos=[0, 0, 0])
    front_left_leg.geom(fromto=[0.0, 0.0, 0.0, 0.2, 0.2, 0.0], name="aux_1_geom", size="0.08", type="capsule")
    aux_1 = front_left_leg.body(name="aux_1", pos=[0.2, 0.2, 0])
    aux_1.joint(axis=[0, 0, 1], name="hip_1", pos=[0.0, 0.0, 0.0], range=[-30, 30], type="hinge")
    aux_1.geom(fromto=[0.0, 0.0, 0.0, 0.2, 0.2, 0.0], name="left_leg_geom", size="0.08", type="capsule")
    ankle_1 = aux_1.body(pos=[0.2, 0.2, 0])
    ankle_1.joint(axis=[-1, 1, 0], name="ankle_1", pos=[0.0, 0.0, 0.0], range=[30, 70], type="hinge")
    ankle_1.geom(fromto=[0.0, 0.0, 0.0, 0.4, 0.4, 0.0], name="left_ankle_geom", size="0.08", type="capsule")

    front_right_leg = ant.body(name="front_right_leg", pos=[0, 0, 0])
    front_right_leg.geom(fromto=[0.0, 0.0, 0.0, -0.2, 0.2, 0.0], name="aux_2_geom", size="0.08", type="capsule")
    aux_2 = front_right_leg.body(name="aux_2", pos=[-0.2, 0.2, 0])
    aux_2.joint(axis=[0, 0, 1], name="hip_2", pos=[0.0, 0.0, 0.0], range=[-30, 30], type="hinge")
    aux_2.geom(fromto=[0.0, 0.0, 0.0, -0.2, 0.2, 0.0], name="right_leg_geom", size="0.08", type="capsule")
    ankle_2 = aux_2.body(pos=[-0.2, 0.2, 0])
    ankle_2.joint(axis=[1, 1, 0], name="ankle_2", pos=[0.0, 0.0, 0.0], range=[-70, -30], type="hinge")
    ankle_2.geom(fromto=[0.0, 0.0, 0.0, -0.4, 0.4, 0.0], name="right_ankle_geom", size="0.08", type="capsule")

    # Back left leg is crippled
    thigh_length = 0.1 #0.2
    ankle_length = 0.2 #0.4
    dark_red = [0.8,0.3,0.3,1.0]

    back_left_leg = ant.body(name="back_left_leg", pos=[0, 0, 0])
    back_left_leg.geom(fromto=[0.0, 0.0, 0.0, -0.2, -0.2, 0.0], name="aux_3_geom", size="0.08", type="capsule",
                       rgba=dark_red)
    aux_3 = back_left_leg.body(name="aux_3", pos=[-0.2, -0.2, 0])
    aux_3.joint(axis=[0, 0, 1], name="hip_3", pos=[0.0, 0.0, 0.0], range=[-30, 30], type="hinge")
    aux_3.geom(fromto=[0.0, 0.0, 0.0, -thigh_length, -thigh_length, 0.0], name="backleft_leg_geom", size="0.08", type="capsule",
               rgba=dark_red)
    ankle_3 = aux_3.body(pos=[-thigh_length, -thigh_length, 0])
    ankle_3.joint(axis=[-1, 1, 0], name="ankle_3", pos=[0.0, 0.0, 0.0], range=[-70, -30], type="hinge")
    ankle_3.geom(fromto=[0.0, 0.0, 0.0, -ankle_length, -ankle_length, 0.0], name="backleft_ankle_geom", size="0.08", type="capsule",
                 rgba=dark_red)

    back_right_leg = ant.body(name="back_right_leg", pos=[0, 0, 0])
    back_right_leg.geom(fromto=[0.0, 0.0, 0.0, 0.2, -0.2, 0.0], name="aux_4_geom", size="0.08", type="capsule",
                        rgba=dark_red)
    aux_4 = back_right_leg.body(name="aux_4", pos=[0.2, -0.2, 0])
    aux_4.joint(axis=[0, 0, 1], name="hip_4", pos=[0.0, 0.0, 0.0], range=[-30, 30], type="hinge")
    aux_4.geom(fromto=[0.0, 0.0, 0.0, thigh_length, -thigh_length, 0.0], name="backright_leg_geom", size="0.08", type="capsule",
               rgba=dark_red)
    ankle_4 = aux_4.body(pos=[thigh_length, -thigh_length, 0])
    ankle_4.joint(axis=[1, 1, 0], name="ankle_4", pos=[0.0, 0.0, 0.0], range=[30, 70], type="hinge")
    ankle_4.geom(fromto=[0.0, 0.0, 0.0, ankle_length, -ankle_length, 0.0], name="backright_ankle_geom", size="0.08", type="capsule",
                 rgba=dark_red)

    actuator = mjcmodel.root.actuator()
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="hip_1", gear=gear)
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="ankle_1", gear=gear)
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="hip_2", gear=gear)
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="ankle_2", gear=gear)
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="hip_3", gear=1) # cripple the joints
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="ankle_3", gear=1) # cripple the joints
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="hip_4", gear=1)
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="ankle_4", gear=1)
    return mjcmodel


def angry_ant_crippled_one(gear=150):
    mjcmodel = MJCModel('ant_maze')
    mjcmodel.root.compiler(inertiafromgeom="true", angle="degree", coordinate="local")
    mjcmodel.root.option(timestep="0.01", integrator="RK4")
    mjcmodel.root.custom().numeric(data="0.0 0.0 0.55 1.0 0.0 0.0 0.0 0.0 1.0 0.0 -1.0 0.0 -1.0 0.0 1.0",name="init_qpos")
    asset = mjcmodel.root.asset()
    asset.texture(builtin="gradient",height="100",rgb1="1 1 1",rgb2="0 0 0",type="skybox",width="100")
    asset.texture(builtin="flat",height="1278",mark="cross",markrgb="1 1 1",name="texgeom",random="0.01",rgb1="0.8 0.6 0.4",rgb2="0.8 0.6 0.4",type="cube",width="127")
    asset.texture(builtin="checker",height="100",name="texplane",rgb1="0 0 0",rgb2="0.8 0.8 0.8",type="2d",width="100")
    asset.material(name="MatPlane",reflectance="0.5",shininess="1",specular="1",texrepeat="60 60",texture="texplane")
    asset.material(name="geom",texture="texgeom",texuniform="true")



    default = mjcmodel.root.default()
    default.joint(armature=1, damping=1, limited='true')
    default.geom(friction=[1.5,0.5,0.5], density=5.0, margin=0.01, condim=3, conaffinity=0, rgba="0.8 0.6 0.4 1")

    worldbody = mjcmodel.root.worldbody()

    worldbody.geom(conaffinity=1, condim=3, material="MatPlane",name="floor",pos="0 0 0",rgba="0.8 0.9 0.8 1",size="40 40 40",type="plane")
    worldbody.light(cutoff="100",diffuse=[.8,.8,.8],dir="-0 0 -1.3",directional="true",exponent="1",pos="0 0 1.3",specular=".1 .1 .1")


    ant = worldbody.body(name='torso', pos=[0, 0, 0.75])
    ant.geom(name='torso_geom', pos=[0, 0, 0], size="0.25", type="sphere")
    ant.joint(armature="0", damping="0", limited="false", margin="0.01", name="root", pos=[0, 0, 0], type="free")

    eye_z = 0.1
    eye_y = -.21
    eye_x_offset = 0.07
    # eyes
    ant.geom(fromto=[eye_x_offset,0,eye_z,eye_x_offset,eye_y,eye_z], name='eye1', size='0.03', type='capsule', rgba=[1,1,1,1])
    ant.geom(fromto=[eye_x_offset,0,eye_z,eye_x_offset,eye_y-0.02,eye_z], name='eye1_', size='0.02', type='capsule', rgba=[0,0,0,1])
    ant.geom(fromto=[-eye_x_offset,0,eye_z,-eye_x_offset,eye_y,eye_z], name='eye2', size='0.03', type='capsule', rgba=[1,1,1,1])
    ant.geom(fromto=[-eye_x_offset,0,eye_z,-eye_x_offset,eye_y-0.02,eye_z], name='eye2_', size='0.02', type='capsule', rgba=[0,0,0,1])
    # eyebrows
    ant.geom(fromto=[eye_x_offset-0.03,eye_y, eye_z+0.07, eye_x_offset+0.03, eye_y, eye_z+0.1], name='brow1', size='0.02', type='capsule', rgba=[0,0,0,1])
    ant.geom(fromto=[-eye_x_offset+0.03,eye_y, eye_z+0.07, -eye_x_offset-0.03, eye_y, eye_z+0.1], name='brow2', size='0.02', type='capsule', rgba=[0,0,0,1])




    front_left_leg = ant.body(name="front_left_leg", pos=[0, 0, 0])
    front_left_leg.geom(fromto=[0.0, 0.0, 0.0, 0.2, 0.2, 0.0], name="aux_1_geom", size="0.08", type="capsule")
    aux_1 = front_left_leg.body(name="aux_1", pos=[0.2, 0.2, 0])
    aux_1.joint(axis=[0, 0, 1], name="hip_1", pos=[0.0, 0.0, 0.0], range=[-30, 30], type="hinge")
    aux_1.geom(fromto=[0.0, 0.0, 0.0, 0.2, 0.2, 0.0], name="left_leg_geom", size="0.08", type="capsule")
    ankle_1 = aux_1.body(pos=[0.2, 0.2, 0])
    ankle_1.joint(axis=[-1, 1, 0], name="ankle_1", pos=[0.0, 0.0, 0.0], range=[30, 70], type="hinge")
    ankle_1.geom(fromto=[0.0, 0.0, 0.0, 0.4, 0.4, 0.0], name="left_ankle_geom", size="0.08", type="capsule")

    front_right_leg = ant.body(name="front_right_leg", pos=[0, 0, 0])
    front_right_leg.geom(fromto=[0.0, 0.0, 0.0, -0.2, 0.2, 0.0], name="aux_2_geom", size="0.08", type="capsule")
    aux_2 = front_right_leg.body(name="aux_2", pos=[-0.2, 0.2, 0])
    aux_2.joint(axis=[0, 0, 1], name="hip_2", pos=[0.0, 0.0, 0.0], range=[-30, 30], type="hinge")
    aux_2.geom(fromto=[0.0, 0.0, 0.0, -0.2, 0.2, 0.0], name="right_leg_geom", size="0.08", type="capsule")
    ankle_2 = aux_2.body(pos=[-0.2, 0.2, 0])
    ankle_2.joint(axis=[1, 1, 0], name="ankle_2", pos=[0.0, 0.0, 0.0], range=[-70, -30], type="hinge")
    ankle_2.geom(fromto=[0.0, 0.0, 0.0, -0.4, 0.4, 0.0], name="right_ankle_geom", size="0.08", type="capsule")

    # Back left leg is crippled
    thigh_length = 0.1 #0.2
    ankle_length = 0.2 #0.4
    dark_red = [0.8,0.3,0.3,1.0]

    # back_left_leg = ant.body(name="back_left_leg", pos=[0, 0, 0])
    # back_left_leg.geom(fromto=[0.0, 0.0, 0.0, -0.2, -0.2, 0.0], name="aux_3_geom", size="0.08", type="capsule",
    #                    rgba=dark_red)
    # aux_3 = back_left_leg.body(name="aux_3", pos=[-0.2, -0.2, 0])
    # aux_3.joint(axis=[0, 0, 1], name="hip_3", pos=[0.0, 0.0, 0.0], range=[-30, 30], type="hinge")
    # aux_3.geom(fromto=[0.0, 0.0, 0.0, -thigh_length, -thigh_length, 0.0], name="backleft_leg_geom", size="0.08", type="capsule",
    #            rgba=dark_red)
    # ankle_3 = aux_3.body(pos=[-thigh_length, -thigh_length, 0])
    # ankle_3.joint(axis=[-1, 1, 0], name="ankle_3", pos=[0.0, 0.0, 0.0], range=[-70, -30], type="hinge")
    # ankle_3.geom(fromto=[0.0, 0.0, 0.0, -ankle_length, -ankle_length, 0.0], name="backleft_ankle_geom", size="0.08", type="capsule",
    #              rgba=dark_red)
    back_left_leg = ant.body(name="back_left_leg", pos=[0, 0, 0])
    back_left_leg.geom(fromto=[0.0, 0.0, 0.0, -0.2, -0.2, 0.0], name="aux_3_geom", size="0.08", type="capsule")
    aux_3 = back_left_leg.body(name="aux_3", pos=[-0.2, -0.2, 0])
    aux_3.joint(axis=[0, 0, 1], name="hip_3", pos=[0.0, 0.0, 0.0], range=[-30, 30], type="hinge")
    aux_3.geom(fromto=[0.0, 0.0, 0.0, -0.2, -0.2, 0.0], name="backleft_leg_geom", size="0.08", type="capsule")
    ankle_3 = aux_3.body(pos=[-0.2, -0.2, 0])
    ankle_3.joint(axis=[-1, 1, 0], name="ankle_3", pos=[0.0, 0.0, 0.0], range=[-70, -30], type="hinge")
    ankle_3.geom(fromto=[0.0, 0.0, 0.0, -0.4, -0.4, 0.0], name="backleft_ankle_geom", size="0.08", type="capsule")


    back_right_leg = ant.body(name="back_right_leg", pos=[0, 0, 0])
    back_right_leg.geom(fromto=[0.0, 0.0, 0.0, 0.2, -0.2, 0.0], name="aux_4_geom", size="0.08", type="capsule",
                        rgba=dark_red)
    aux_4 = back_right_leg.body(name="aux_4", pos=[0.2, -0.2, 0])
    aux_4.joint(axis=[0, 0, 1], name="hip_4", pos=[0.0, 0.0, 0.0], range=[-30, 30], type="hinge")
    aux_4.geom(fromto=[0.0, 0.0, 0.0, thigh_length, -thigh_length, 0.0], name="backright_leg_geom", size="0.08", type="capsule",
               rgba=dark_red)
    ankle_4 = aux_4.body(pos=[thigh_length, -thigh_length, 0])
    ankle_4.joint(axis=[1, 1, 0], name="ankle_4", pos=[0.0, 0.0, 0.0], range=[30, 70], type="hinge")
    ankle_4.geom(fromto=[0.0, 0.0, 0.0, ankle_length, -ankle_length, 0.0], name="backright_ankle_geom", size="0.08", type="capsule",
                 rgba=dark_red)

    actuator = mjcmodel.root.actuator()
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="hip_1", gear=gear)
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="ankle_1", gear=gear)
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="hip_2", gear=gear)
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="ankle_2", gear=gear)
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="hip_3", gear=gear) # cripple the joints
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="ankle_3", gear=gear) # cripple the joints
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="hip_4", gear=1)
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="ankle_4", gear=1)
    return mjcmodel


class CustomAntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """
    A modified ant env with lower joint gear ratios so it flips less often and learns faster.
    """
    def __init__(self, max_timesteps=1000, disabled=0, gear=150):
        #mujoco_env.MujocoEnv.__init__(self, 'ant.xml', 5)
        utils.EzPickle.__init__(self)
        self.timesteps = 0
        self.max_timesteps=max_timesteps

        if disabled==1:
            model = angry_ant_crippled_one(gear=gear)
        elif disabled==2:
            model = angry_ant_crippled_two(gear=gear)
        elif disabled==0:
            model = ant_env(gear=gear)
        with model.asfile() as f:
            mujoco_env.MujocoEnv.__init__(self, f.name, 5)

    def step(self, a):
        vel = self.data.qvel.flat[0]
        forward_reward = vel
        self.do_simulation(a, self.frame_skip)

        ctrl_cost = .01 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.data.cfrc_ext, -1, 1)))
        state = self.state_vector()
        flipped = not (state[2] >= 0.2) 
        flipped_rew = -1 if flipped else 0
        reward = forward_reward - ctrl_cost - contact_cost +flipped_rew

        self.timesteps += 1
        done = self.timesteps >= self.max_timesteps

        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_flipped=flipped_rew)

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos.flat[2:],
            self.data.qvel.flat,
            np.clip(self.data.cfrc_ext, -1, 1).flat,
        ])

    def reset_model(self):
        self.timesteps = 0
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def log_diagnostics(self, paths):
        forward_rew = np.array([np.mean(traj['env_infos']['reward_forward']) for traj in paths])
        reward_ctrl = np.array([np.mean(traj['env_infos']['reward_ctrl']) for traj in paths])
        reward_cont = np.array([np.mean(traj['env_infos']['reward_contact']) for traj in paths])
        reward_flip = np.array([np.mean(traj['env_infos']['reward_flipped']) for traj in paths])

        # logger.record_tabular('AvgRewardFwd', np.mean(forward_rew))
        # logger.record_tabular('AvgRewardCtrl', np.mean(reward_ctrl))
        # logger.record_tabular('AvgRewardContact', np.mean(reward_cont))
        # logger.record_tabular('AvgRewardFlipped', np.mean(reward_flip))


def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, axis=1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, axis=0) / np.sum(mass))[0:2].copy()


XML3 = """
<mujoco model="humanoid">
    <compiler angle="degree" inertiafromgeom="true"/>
    <default>
        <joint armature="1" damping="1" limited="true"/>
        <geom conaffinity="1" condim="1" contype="1" margin="0.001" material="geom" rgba="0.8 0.6 .4 1"/>
        <motor ctrllimited="true" ctrlrange="-.4 .4"/>
    </default>
    <option integrator="RK4" iterations="50" solver="PGS" timestep="0.003">
        <!-- <flags solverstat="enable" energy="enable"/>-->
    </option>
    <size nkey="5" nuser_geom="1"/>
    <visual>
        <map fogend="5" fogstart="3"/>
    </visual>
    <asset>
        <texture builtin="gradient" height="100" rgb1=".4 .5 .6" rgb2="0 0 0" type="skybox" width="100"/>
        <!-- <texture builtin="gradient" height="100" rgb1="1 1 1" rgb2="0 0 0" type="skybox" width="100"/>-->
        <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
        <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d" width="100"/>
        <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane"/>
        <material name="geom" texture="texgeom" texuniform="true"/>
    </asset>
    <worldbody>
        <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
        <geom condim="3" friction="1 .1 .1" material="MatPlane" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" size="20 20 0.125" type="plane"/>
        <!-- <geom condim="3" material="MatPlane" name="floor" pos="0 0 0" size="10 10 0.125" type="plane"/>-->
        <body name="torso" pos="0 0 1.4">
            <camera name="track" mode="trackcom" pos="0 -4 0" xyaxes="1 0 0 0 0 1"/>
            <joint armature="0" damping="0" limited="false" name="root" pos="0 0 0" stiffness="0" type="free"/>
            <geom fromto="0 -.07 0 0 .07 0" name="torso1" size="0.07" type="capsule"/>
            <geom name="head" pos="0 0 .19" size=".09" type="sphere" user="258"/>
            <geom fromto="-.01 -.06 -.12 -.01 .06 -.12" name="uwaist" size="0.06" type="capsule"/>
            <body name="lwaist" pos="-.01 0 -0.260" quat="1.000 0 -0.002 0">
                <geom fromto="0 -.06 0 0 .06 0" name="lwaist" size="0.06" type="capsule" rgba="0.8 0.3 0.3 1.0"/>
                <joint armature="0.02" axis="0 0 1" damping="5" name="abdomen_z" pos="0 0 0.065" range="-45 45" stiffness="20" type="hinge"/>
                <joint armature="0.02" axis="0 1 0" damping="5" name="abdomen_y" pos="0 0 0.065" range="-75 30" stiffness="10" type="hinge"/>
                <body name="pelvis" pos="0 0 -0.165" quat="1.000 0 -0.002 0">
                    <joint armature="0.02" axis="1 0 0" damping="5" name="abdomen_x" pos="0 0 0.1" range="-35 35" stiffness="10" type="hinge"/>
                    <geom fromto="-.02 -.07 0 -.02 .07 0" name="butt" size="0.09" type="capsule"/>
                    <body name="right_thigh" pos="0 -0.1 -0.04">
                        <joint armature="0.01" axis="1 0 0" damping="5" name="right_hip_x" pos="0 0 0" range="-25 5" stiffness="10" type="hinge"/>
                        <joint armature="0.01" axis="0 0 1" damping="5" name="right_hip_z" pos="0 0 0" range="-60 35" stiffness="10" type="hinge"/>
                        <joint armature="0.0080" axis="0 1 0" damping="5" name="right_hip_y" pos="0 0 0" range="-110 20" stiffness="20" type="hinge"/>
                        <geom fromto="0 0 0 0 0.01 -.34" name="right_thigh1" size="0.06" type="capsule"/>
                        <body name="right_shin" pos="0 0.01 -0.403">
                            <joint armature="0.0060" axis="0 -1 0" name="right_knee" pos="0 0 .02" range="-160 -2" type="hinge"/>
                            <geom fromto="0 0 0 0 0 -.3" name="right_shin1" size="0.049" type="capsule"/>
                            <body name="right_foot" pos="0 0 -0.45">
                                <geom name="right_foot" pos="0 0 0.1" size="0.075" type="sphere" user="0"/>
                            </body>
                        </body>
                    </body>
                    <body name="left_thigh" pos="0 0.1 -0.04">
                        <joint armature="0.01" axis="-1 0 0" damping="5" name="left_hip_x" pos="0 0 0" range="-25 5" stiffness="10" type="hinge"/>
                        <joint armature="0.01" axis="0 0 -1" damping="5" name="left_hip_z" pos="0 0 0" range="-60 35" stiffness="10" type="hinge"/>
                        <joint armature="0.01" axis="0 1 0" damping="5" name="left_hip_y" pos="0 0 0" range="-110 20" stiffness="20" type="hinge"/>
                        <geom fromto="0 0 0 0 -0.01 -.34" name="left_thigh1" size="0.06" type="capsule"/>
                        <body name="left_shin" pos="0 -0.01 -0.403">
                            <joint armature="0.0060" axis="0 -1 0" name="left_knee" pos="0 0 .02" range="-160 -2" stiffness="1" type="hinge"/>
                            <geom fromto="0 0 0 0 0 -.3" name="left_shin1" size="0.049" type="capsule"/>
                            <body name="left_foot" pos="0 0 -0.45">
                                <geom name="left_foot" type="sphere" size="0.075" pos="0 0 0.1" user="0" />
                            </body>
                        </body>
                    </body>
                </body>
            </body>
            <body name="right_upper_arm" pos="0 -0.17 0.06">
                <joint armature="0.0068" axis="2 1 1" name="right_shoulder1" pos="0 0 0" range="-85 60" stiffness="1" type="hinge"/>
                <joint armature="0.0051" axis="0 -1 1" name="right_shoulder2" pos="0 0 0" range="-85 60" stiffness="1" type="hinge"/>
                <geom fromto="0 0 0 .16 -.16 -.16" name="right_uarm1" size="0.04 0.16" type="capsule"/>
                <body name="right_lower_arm" pos=".18 -.18 -.18">
                    <joint armature="0.0028" axis="0 -1 1" name="right_elbow" pos="0 0 0" range="-90 50" stiffness="0" type="hinge"/>
                    <geom fromto="0.01 0.01 0.01 .17 .17 .17" name="right_larm" size="0.031" type="capsule"/>
                    <geom name="right_hand" pos=".18 .18 .18" size="0.04" type="sphere"/>
                    <camera pos="0 0 0"/>
                </body>
            </body>
            <body name="left_upper_arm" pos="0 0.17 0.06">
                <joint armature="0.0068" axis="2 -1 1" name="left_shoulder1" pos="0 0 0" range="-60 85" stiffness="1" type="hinge"/>
                <joint armature="0.0051" axis="0 1 1" name="left_shoulder2" pos="0 0 0" range="-60 85" stiffness="1" type="hinge"/>
                <geom fromto="0 0 0 .16 .16 -.16" name="left_uarm1" size="0.04 0.16" type="capsule"/>
                <body name="left_lower_arm" pos=".18 .18 -.18">
                    <joint armature="0.0028" axis="0 -1 -1" name="left_elbow" pos="0 0 0" range="-90 50" stiffness="0" type="hinge"/>
                    <geom fromto="0.01 -0.01 0.01 .17 -.17 .17" name="left_larm" size="0.031" type="capsule"/>
                    <geom name="left_hand" pos=".18 -.18 .18" size="0.04" type="sphere"/>
                </body>
            </body>
        </body>
    </worldbody>
    <tendon>
        <fixed name="left_hipknee">
            <joint coef="-1" joint="left_hip_y"/>
            <joint coef="1" joint="left_knee"/>
        </fixed>
        <fixed name="right_hipknee">
            <joint coef="-1" joint="right_hip_y"/>
            <joint coef="1" joint="right_knee"/>
        </fixed>
    </tendon>

    <actuator>
        <motor gear="100" joint="abdomen_y" name="abdomen_y"/>
        <motor gear="100" joint="abdomen_z" name="abdomen_z"/>
        <motor gear="100" joint="abdomen_x" name="abdomen_x"/>
        <motor gear="100" joint="right_hip_x" name="right_hip_x"/>
        <motor gear="100" joint="right_hip_z" name="right_hip_z"/>
        <motor gear="300" joint="right_hip_y" name="right_hip_y"/>
        <motor gear="200" joint="right_knee" name="right_knee"/>
        <motor gear="100" joint="left_hip_x" name="left_hip_x"/>
        <motor gear="100" joint="left_hip_z" name="left_hip_z"/>
        <motor gear="300" joint="left_hip_y" name="left_hip_y"/>
        <motor gear="200" joint="left_knee" name="left_knee"/>
        <motor gear="25" joint="right_shoulder1" name="right_shoulder1"/>
        <motor gear="25" joint="right_shoulder2" name="right_shoulder2"/>
        <motor gear="25" joint="right_elbow" name="right_elbow"/>
        <motor gear="25" joint="left_shoulder1" name="left_shoulder1"/>
        <motor gear="25" joint="left_shoulder2" name="left_shoulder2"/>
        <motor gear="25" joint="left_elbow" name="left_elbow"/>
    </actuator>
</mujoco>
"""


class BrokenHumanoidEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(
        self,
        xml_file="humanoid.xml",
        forward_reward_weight=1.25,
        ctrl_cost_weight=0.1,
        contact_cost_weight=5e-7,
        contact_cost_range=(-np.inf, 10.0),
        healthy_reward=5.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(1.0, 2.0),
        reset_noise_scale=1e-2,
        exclude_current_positions_from_observation=True,
        broken_joint=6,
    ):
        # Select broken joint
        self.broken_joint = broken_joint
        self._tempfile = tempfile.NamedTemporaryFile(mode="w", suffix=".xml")
        self._tempfile.write(XML3)
        self._tempfile.flush()
        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight
        self._contact_cost_range = contact_cost_range
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        # mujoco_env.MujocoEnv.__init__(self, xml_file, 5)
        mujoco_env.MujocoEnv.__init__(self, self._tempfile.name, 5)

    @property
    def healthy_reward(self):
        return (
            float(self.is_healthy or self._terminate_when_unhealthy)
            * self._healthy_reward
        )

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(self.sim.data.ctrl))
        return control_cost

    @property
    def contact_cost(self):
        contact_forces = self.sim.data.cfrc_ext
        contact_cost = self._contact_cost_weight * np.sum(np.square(contact_forces))
        min_cost, max_cost = self._contact_cost_range
        contact_cost = np.clip(contact_cost, min_cost, max_cost)
        return contact_cost

    @property
    def is_healthy(self):
        min_z, max_z = self._healthy_z_range
        is_healthy = min_z < self.sim.data.qpos[2] < max_z

        return is_healthy

    @property
    def done(self):
        done = (not self.is_healthy) if self._terminate_when_unhealthy else False
        # done = False
        return done

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        com_inertia = self.sim.data.cinert.flat.copy()
        com_velocity = self.sim.data.cvel.flat.copy()

        actuator_forces = self.sim.data.qfrc_actuator.flat.copy()
        external_contact_forces = self.sim.data.cfrc_ext.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        return np.concatenate(
            (
                position,
                velocity,
                com_inertia,
                com_velocity,
                actuator_forces,
                external_contact_forces,
            )
        )

    def step(self, action):
        # set the action of broken joint as 6
        # action = action.copy()
        if self.broken_joint is not None:
            action[self.broken_joint] = 0

        xy_position_before = mass_center(self.model, self.sim)
        self.do_simulation(action, self.frame_skip)
        xy_position_after = mass_center(self.model, self.sim)

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost

        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward
        costs = ctrl_cost + contact_cost

        observation = self._get_obs()
        reward = rewards - costs
        done = self.done
        info = {
            "reward_linvel": forward_reward,
            "reward_quadctrl": -ctrl_cost,
            "reward_alive": healthy_reward,
            "reward_impact": -contact_cost,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
        }

        return observation, reward, done, info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )
        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG_HUMANOID.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

