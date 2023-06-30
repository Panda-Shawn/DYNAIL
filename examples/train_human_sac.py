from stable_baselines3 import PPO, SAC, TD3
import gym

from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper

from imitation.rewards.reward_nets import BasicRewardNet, BasicShapedRewardNet
from imitation.util.networks import RunningNorm
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from imitation.scripts.common import demonstrations, common

import matplotlib.pyplot as plt
import numpy as np
import os

import logging
from imitation.util import util
from imitation.util import logger as imit_logger

from imitation.policies import serialize, base

import torch as th
import random


if __name__=="__main__":
    logger = logging.getLogger(__name__)


    '''Select environment and algorithm'''

    env_id = "Humanoid-v3"
    algo = SAC
    algo_cls = f"{algo.__name__}"


    '''set random seed'''

    seed = 2022
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


    '''Set path of logger'''

    log_dir = os.path.join(
                "output",
                algo_cls,
                env_id,
            )
    os.makedirs(
        log_dir, exist_ok=True
    )

    class Logger(object):
        def __init__(self, filename="Default.log"):
            self.terminal = sys.stdout
            self.log = open(filename, "a")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass

    sys.stdout = Logger(os.path.join(log_dir, "cout.txt"))

    logging.basicConfig(level=logging.INFO)
    logger.setLevel("INFO") # ensure the logger level in case that some packages set basicConfig before
    para_handler = logging.FileHandler(os.path.join(log_dir, "log.txt"))
    para_handler.setLevel("INFO") # handler level has to be higher than logger level
    logger.addHandler(para_handler)
    logger.info("Logging to %s", log_dir)

    custom_logger = imit_logger.configure(
        folder=os.path.join(log_dir, "log"),
        format_strs=["tensorboard", "stdout"],
    )


    '''Create RL algorithm'''

    num_envs = 10
    venv = util.make_vec_env(
        env_name=env_id,
        n_envs=num_envs,
        seed=seed,
        parallel=True,
        max_episode_steps=None,
        log_dir=log_dir,
    )
    learner = algo(
        env=venv,
        policy="MlpPolicy",
        # policy=base.FeedForward32Policy,
        batch_size=250, # set 16384 for using the entire buffer once update, buffer size is 8 * 2048 = 16384 (8 for num_envs, 2048 for n_steps)
        ent_coef=0.01,
        learning_rate=5e-4,
        train_freq=50,
        gradient_steps=200,
        learning_starts=10000,    
        gamma=0.99,
        buffer_size= int(5e5),
        policy_kwargs=dict(net_arch=dict(qf=[400,300], pi=[400,300]), activation_fn=th.nn.ReLU),
        verbose = 0,
    )
    learner.set_logger(custom_logger)

    '''Record information for log'''

    logger.info("*" * 20 + "Random Seed" + "*" * 20)
    logger.info(f"random seed: {seed}")
    logger.info("*" * 20 + "Learner Information" + "*" * 20)
    logger.info(f"learner: {learner}")
    logger.info(f"learner.policy: {learner.policy}")
    logger.info(f"learner.batch_size: {learner.batch_size}")
    logger.info(f"learner.env.num_envs: {learner.env.num_envs}")
    logger.info(f"learner.learning_rate: {learner.learning_rate}")
    logger.info(f"learner.train_freq: {learner.train_freq}")
    logger.info(f"learner.gradient_steps: {learner.gradient_steps}")
    logger.info(f"learner.learning_starts: {learner.learning_starts}")
    logger.info(f"learner.gamma: {learner.gamma}")
    logger.info(f"learner.buffer_size: {learner.buffer_size}")

    '''Start training and compare the performance'''
    learner_rewards_before_training, _ = evaluate_policy(
        learner, venv, 10, return_episode_rewards=True
    )
    learner.learn(total_timesteps=3000 * 1000) 
    learner_rewards_after_training, _ = evaluate_policy(
        learner, venv, 10, return_episode_rewards=True
    )
    
    print(np.mean(learner_rewards_after_training))
    print(np.mean(learner_rewards_before_training))

    logger.info(f"reward after training: {np.mean(learner_rewards_after_training)}")
    logger.info(f"reward before training: {np.mean(learner_rewards_before_training)}")

    '''Save discriminator and generator'''

    output_dir = os.path.join(log_dir, "final")
    serialize.save_stable_model(output_dir, learner)