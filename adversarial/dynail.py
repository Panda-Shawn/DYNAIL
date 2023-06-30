"""Dynamics Adapted Imitation Learning (DYNAIL)."""

from typing import Optional

import torch as th
from torch.nn import functional as F
from stable_baselines3.common import base_class, vec_env

from imitation.algorithms import base
from imitation.algorithms.adversarial import common
from imitation.rewards import reward_nets
from imitation.util import networks


class ClippingRewardNet(reward_nets.RewardNet):
    """Wrapper for reward network that takes log sigmoid of wrapped network."""

    def __init__(self, reward_net, clipping_value):
        """Builds LogSigmoidRewardNet to wrap `reward_net`."""
        # TODO(adam): make an explicit RewardNetWrapper class?
        super().__init__(
            observation_space=reward_net.observation_space,
            action_space=reward_net.action_space,
            normalize_images=reward_net.normalize_images,
        )
        self.reward_net = reward_net
        self.clipping_value = clipping_value

    def forward(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
    ) -> th.Tensor:
        """Computes negative log sigmoid of base reward network."""
        output = th.clip(self.reward_net(state, action, next_state, done), -self.clipping_value, self.clipping_value)
        return output


class DYNAIL(common.AdversarialTrainer):
    """Dynamics Adapted Imitation Learning (DYNAIL).
    DYNAIL: https://openreview.net/forum?id=w36pqfaJ4t
    """

    def __init__(
        self,
        *,
        demonstrations: base.AnyTransitions,
        demo_batch_size: int,
        venv: vec_env.VecEnv,
        gen_algo: base_class.BaseAlgorithm,
        reward_net: Optional[reward_nets.RewardNet] = None,
        q_sa_net: Optional[reward_nets.RewardNet] = None,
        q_sas_net: Optional[reward_nets.RewardNet] = None,
        eta: float = 0.01,
        **kwargs,
    ):
        """Builds an DYNAIL trainer.

        Args:
            demonstrations: Demonstrations from an expert (optional). Transitions
                expressed directly as a `types.TransitionsMinimal` object, a sequence
                of trajectories, or an iterable of transition batches (mappings from
                keywords to arrays containing observations, etc).
            demo_batch_size: The number of samples in each batch of expert data. The
                discriminator batch size is twice this number because each discriminator
                batch contains a generator sample for every expert sample.
            venv: The vectorized environment to train in.
            gen_algo: The generator RL algorithm that is trained to maximize
                discriminator confusion. Environment and logger will be set to
                `venv` and `custom_logger`.
            reward_net: Reward network; used as part of DYNAIL discriminator. Defaults to
                `reward_nets.BasicShapedRewardNet` when unspecified.
            **kwargs: Passed through to `AdversarialTrainer.__init__`.

        Raises:
            TypeError: If `gen_algo.policy` does not have an `evaluate_actions`
                attribute (present in `ActorCriticPolicy`), needed to compute
                log-probability of actions.
        """
        net_kwagrs = {"normalize_input_layer": networks.RunningNorm}

        if reward_net is None:
            reward_net = reward_nets.BasicRewardNet(
                observation_space=venv.observation_space,
                action_space=venv.action_space,
                **net_kwagrs,
            )

        if q_sa_net is None:
            q_sa_net = reward_nets.BasicRewardNet(
                observation_space=venv.observation_space,
                action_space=venv.action_space,
                **net_kwagrs,
            )

        if q_sas_net is None:
            q_sas_net = reward_nets.BasicRewardNet(
                observation_space=venv.observation_space,
                action_space=venv.action_space,
                use_next_state=True,
                **net_kwagrs,
            )

        q_sa_net = q_sa_net.to(gen_algo.device)
        q_sas_net = q_sas_net.to(gen_algo.device)
        reward_net = reward_net.to(gen_algo.device)
        self.eta = eta

        super().__init__(
            demonstrations=demonstrations,
            demo_batch_size=demo_batch_size,
            venv=venv,
            gen_algo=gen_algo,
            reward_net=reward_net,
            q_sas_net=q_sas_net,
            q_sa_net=q_sa_net,
            **kwargs,
        )
        if not hasattr(self.gen_algo.policy, "evaluate_actions") and not hasattr(self.gen_algo.policy.actor, "action_log_prob"):
            raise TypeError(
                "DYNAIL needs a stochastic policy to compute the discriminator output.",
            )

    def logits_gen_is_high(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
        log_policy_act_prob: th.Tensor,
    ) -> th.Tensor:
        """Compute the discriminator's logits for each state-action sample."""
        if log_policy_act_prob is None:
            raise TypeError(
                "Non-None `log_policy_act_prob` is required for this method.",
            )
        reward_output_train = th.clip(self._reward_net(state, action, next_state, done), -20.0, 20.0)
        reward_output_train = self._reward_net(state, action, next_state, done)
        q_sa_logits = self.q_sa_net.forward(state, action, next_state, done)
        q_sas_logits = self.q_sas_net.forward(state, action, next_state, done)
        sig_q_sa_logits = F.sigmoid(q_sa_logits)
        sig_q_sas_logits = F.sigmoid(q_sas_logits + q_sa_logits)
        delta_reward = self.eta * th.clip((th.log(sig_q_sas_logits / (1 - sig_q_sas_logits + 1e-8) + 1e-8) - th.log(
            sig_q_sa_logits / (1 - sig_q_sa_logits + 1e-8) + 1e-8)), -5.0, 5.0)

        d_logits = log_policy_act_prob - (reward_output_train + delta_reward)
        return d_logits, q_sas_logits, q_sa_logits, (reward_output_train, delta_reward, self.eta)

    @property
    def reward_train(self) -> reward_nets.RewardNet:
        return ClippingRewardNet(self._reward_net, 20.0)

    @property
    def reward_test(self) -> reward_nets.RewardNet:
        if isinstance(self._reward_net, reward_nets.ShapedRewardNet):
            return self._reward_net.base
        else:
            return ClippingRewardNet(self._reward_net, 20.0)