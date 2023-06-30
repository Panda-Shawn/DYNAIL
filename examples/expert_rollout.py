from imitation.data import types, rollout
from imitation.policies import serialize
from imitation.scripts.common import common
from imitation.util import util

import numpy as np

import os

import dm2gym.envs.dm_suite_env as dm2gym
import realworldrl_suite.environments as rwrl
import functools
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


if __name__=="__main__":
    policy_type = "sac"
    policy_path = r"./output/SAC/Humanoid-v3/final"

    env_name = "Humanoid-v3"
    num_envs = 10
    seed = 2022

    venv = util.make_vec_env(
        env_name=env_name,
        n_envs=num_envs,
        seed=seed,
        parallel=True,
        max_episode_steps=None,
        log_dir=log_dir,
    )
    
    sample_until = rollout.make_sample_until(min_timesteps=None, min_episodes=60)

    policy = serialize.load_policy(policy_type, policy_path)

    rollouts_path = os.path.join("expert", env_name)
    os.makedirs(rollouts_path, exist_ok=True)

    rollouts = rollout.rollout(policy, venv, sample_until, unwrap = False, exclude_infos = False)
    rollouts = sorted(rollouts, key=lambda x:np.sum(x.rews), reverse=True)[:40]
    types.save(os.path.join(rollouts_path, "rollout.pkl"), rollouts)

    with open(os.path.join(rollouts_path, "rollout_info.txt"), "wt") as f:
        rollouts_rew = []
        for i in range(len(rollouts)):
            rollout_rew = np.sum(rollouts[i].rews)
            f.write(f"rollout index: {i}, rollout reward: {rollout_rew}\n")
            rollouts_rew.append(rollout_rew)
        f.write(f"mean: {np.mean(rollouts_rew)}, std: {np.std(rollouts_rew)}")
