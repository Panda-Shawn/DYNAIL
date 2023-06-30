from stable_baselines3 import SAC, PPO
import gym

from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper

from imitation.algorithms.adversarial.airl import AIRL
from imitation.algorithms.adversarial.dynail import DYNAIL
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

import dm2gym.envs.dm_suite_env as dm2gym
import realworldrl_suite.environments as rwrl
import functools


if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    '''Collect expert trajectories'''

    rollout_path = r"./expert/quadruped/rollout.pkl"
    rollouts = demonstrations.load_expert_trajs(
        rollout_path, n_expert_demos=40
    )
    rollouts = rollouts[:40]

    '''Select environment and algorithm'''

    env_id = "quadruped"
    algo = DYNAIL
    algo_cls = f"{algo.__name__}"

    '''set random seed'''

    seed = 2022  # 5234567
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    '''Set path of logger'''

    log_dir = os.path.join(
        "output",
        algo_cls,
        env_id,
        util.make_unique_timestamp(),
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

    '''Create imitation algorithm'''

    num_envs = 10
    fric_mu = 0.1
    class GymEnv(dm2gym.DMSuiteEnv):
        """Wrapper that convert a realworldrl environment to a gym environment."""

        def __init__(self, env):
            """Constructor. We reuse the facilities from dm2gym."""
            self.env = env
            self.metadata = {
                'render.modes': ['human', 'rgb_array'],
                'video.frames_per_second': round(1. / self.env.control_timestep())
            }
            self.observation_space = dm2gym.convert_dm_control_to_gym_space(
                self.env.observation_spec())
            self.action_space = dm2gym.convert_dm_control_to_gym_space(
                self.env.action_spec())
            self.viewer = None

    def _load_rw_env_fric(seed):
        rl_env = rwrl.load(
            domain_name=env_id,
            task_name='realworld_walk',
            safety_spec=dict(enable=False),
            delay_spec=dict(enable=False), 
            perturb_spec=dict(
                enable = True,
                scheduler = 'constant',
                param = 'contact_friction',
                min = fric_mu,
                max = fric_mu,
                std = 0,
                start = fric_mu
                
            ),    
            environment_kwargs=dict(flat_observation=True)
            )
        env = GymEnv(rl_env)
        print("Loading the RWRL environment...")

        return env

    rng = np.random.RandomState(seed)
    env_seeds = rng.randint(0, (1 << 31) - 1, (num_envs,))
    env_fns = [functools.partial(_load_rw_env_fric, s) for i, s in enumerate(env_seeds)]
    venv = SubprocVecEnv(env_fns, start_method="forkserver")
   
    learner = PPO(
        env=venv,
        policy="MlpPolicy",
        # policy=base.FeedForward32Policy,
        # set 16384 for using the entire buffer once update, buffer size is 8 * 2048 = 16384 (8 for num_envs, 2048 for n_steps)
        batch_size=100,
        ent_coef=0.01,
        learning_rate=3e-4,
        n_epochs=10,
        n_steps=100,
        policy_kwargs=dict(activation_fn=th.nn.ReLU),
        device="cuda:1"
    )
    reward_net = BasicRewardNet(  # choose BasicShapedRewardNet if using shaped reward function
        venv.observation_space,
        venv.action_space,
        # use_action=False, # use state-only reward function
        normalize_input_layer=RunningNorm,
        hid_sizes=(256, 256),
    )
    q_sa_net = BasicRewardNet(  # choose BasicShapedRewardNet if using shaped reward function
        venv.observation_space,
        venv.action_space,
        # use_action=False, # use state-only reward function
        normalize_input_layer=RunningNorm,
        hid_sizes=(256, 256),
    )
    q_sas_net = BasicRewardNet(  # choose BasicShapedRewardNet if using shaped reward function
        venv.observation_space,
        venv.action_space,
        use_next_state=True,
        # use_action=False, # use state-only reward function
        normalize_input_layer=RunningNorm,
        hid_sizes=(256, 256),
    )

    adversarial_trainer = algo(
        demonstrations=rollouts,
        demo_batch_size=1000,  # set 1024 for discriminator batch size
        gen_replay_buffer_capacity=10000,  # set 2048 for discriminator buffer size
        n_disc_updates_per_round=5,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
        q_sa_net=q_sa_net,
        q_sas_net=q_sas_net,
        custom_logger=custom_logger,
        disc_opt_cls=th.optim.Adam,
        disc_opt_kwargs=dict(
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,  # weight_decay=0.01,
        ),
        eta=64.0,  # for DAIL
        mod=40,
        log_dir=log_dir,
    )

    '''Record information for log'''

    logger.info("*" * 20 + "Random Seed" + "*" * 20)
    logger.info(f"random seed: {seed}")
    logger.info("*" * 20 + "Learner Information" + "*" * 20)
    logger.info(f"learner: {learner}")
    logger.info(f"learner.policy: {learner.policy}")
    logger.info(f"learner.batch_size: {learner.batch_size}")
    logger.info(f"learner.n_steps: {learner.n_steps}")
    logger.info(f"learner.env.num_envs: {learner.env.num_envs}")
    logger.info(f"learner.n_epochs: {learner.n_epochs}")
    # logger.info(f"learner.train_freq: {learner.train_freq}")
    # logger.info(f"learner.gradient_steps: {learner.gradient_steps}")
    logger.info("*" * 20 + "Adversarial Trainer Information" + "*" * 20)
    logger.info(f"adversarial_trainer: {adversarial_trainer}")
    logger.info(f"adversarial_trainer.demonstration_path: {rollout_path}")
    logger.info(
        f"adversarial_trainer.demo_batch_size: {adversarial_trainer.demo_batch_size}")
    logger.info(
        f"adversarial_trainer.gen_replay_buffer_capacity: {adversarial_trainer._gen_replay_buffer.capacity}")
    logger.info(
        f"adversarial_trainer.gen_train_timesteps: {adversarial_trainer.gen_train_timesteps}")
    logger.info(
        f"adversarial_trainer.n_disc_updates_per_round: {adversarial_trainer.n_disc_updates_per_round}")
    try:
        logger.info(f"adversarial_trainer.reward_net: {adversarial_trainer.reward_net}")
    except:
        logger.info(f"adversarial_trainer.reward_net: {adversarial_trainer._reward_net}")
    logger.info(
        f"adversarial_trainer.entire_reward_net: {adversarial_trainer.venv_wrapped.reward_fn}")
    if "DAIL" in algo_cls:
        logger.info(f"adversarial_trainer.eta: {adversarial_trainer.eta}")
        logger.info(f"adversarial_trainer.q_sa_net: {adversarial_trainer.q_sa_net}")
        logger.info(f"adversarial_trainer.q_sas_net: {adversarial_trainer.q_sas_net}")
        logger.info(f"adversarial_trainer.mod: {adversarial_trainer.mod}")

    '''Start training and compare the performance'''

    learner_rewards_before_training, _ = evaluate_policy(
        learner, venv, 10, return_episode_rewards=True
    )
    adversarial_trainer.train(int(1e6))  # set 3e7 for CustomAnt-v0 or DisabledAnt-v0
    learner_rewards_after_training, _ = evaluate_policy(
        learner, venv, 10, return_episode_rewards=True
    )

    print(np.mean(learner_rewards_after_training))
    print(np.mean(learner_rewards_before_training))

    logger.info(f"reward after training: {np.mean(learner_rewards_after_training)}")
    logger.info(f"reward before training: {np.mean(learner_rewards_before_training)}")

    '''Save discriminator and generator'''

    def save(trainer, save_path):
        os.makedirs(save_path, exist_ok=True)
        th.save(trainer.reward_train, os.path.join(save_path, "reward_train.pt"))
        th.save(trainer.reward_test, os.path.join(save_path, "reward_test.pt"))
        th.save(trainer.q_sa_net, os.path.join(save_path, "q_sa.pt"))
        th.save(trainer.q_sas_net, os.path.join(save_path, "q_sas.pt"))
        serialize.save_stable_model(
            os.path.join(save_path, "gen_policy"),
            trainer.gen_algo,
        )

    save(adversarial_trainer, os.path.join(log_dir, "checkpoints", "final"))
