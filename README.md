# Dynamics Adapted Imitation Learning [TMLR]

This repo contains code accompaning the paper, [Dynamics Adapted Imitation Learning](https://openreview.net/forum?id=w36pqfaJ4t). DYNAIL is an adversarial imitation learning method dealing with dynamics shift between expert demonstrations and environment.

## Dependencies

The code is based on [Imitation v0.3.0](http://github.com/HumanCompatibleAI/imitation/tree/v0.3.0), which is a library with clean implementations of imitation and reward learning algorithms.

### Install Imitation v0.3.0

```
git clone --depth 1 --branch v0.3.0 https://github.com/HumanCompatibleAI/imitation.git
cd imitation
pip install -e .
```

### Modifications

Then `imitation/examples`, `imitation/src/imitation/algorithms/adversarial` and `imitation/src/imitation/util` can be replaced with the folders provided in this repo. What's more, we use [mujoco-py v2.1.2.14](https://github.com/openai/mujoco-py/tree/v2.1.2.14) instead of mujoco-py v1.5 for all experiments in the paper.

## Usage
For example, we train DYNAIL in BrokenHumanoid-v3 with expert demonstrations from Humanoid-v3.

1. Training expert policy.
   
    We train an expert policy with SAC in Humanoid-v3.
    ```
    cd example
    python train_human_sac.py
    ```
    Then the expert policy is saved in `./output/SAC/Humanoid-v3/final/model.zip`.

2. Generating demonstrations.
   
    We use the expert policy to generate 60 rollouts in Humanoid-v3 and choose the top 40 ones as expert demonstrations.
    ```
    python expert_policy.py
    ```
    Then the demonstration are collected in `./expert/Humanoid-v3/rollout.pkl` along with reward information in `./expert/Humanoid-v3/rollout_info.txt`.

3. Training imitation method.
   
    We train DYNAIL in BrokenHumanoid-v3 with the 40 demonstrations.
    ```
    python train_brohuman_dynail_sac.py
    ```
    For comparision, we can also train GAIFO as a baseline.
    ```
    python train_brohuman_gaifo_sac.py
    ```
    Then, the corresponding results are saved in `./output/DYNAIL/BrokenHumanoid-v3` and `./output/GAIL/BrokenHumanoid-v3`, respectively.

## Results

### Experiments with [realworldrl-suite](https://github.com/google-research/realworldrl_suite)

#### Source Domain: Quadruped

<div align="center">
    <img src="media/source_expert.gif" width="200"/>
    <br/>
    <font>source expert in source domain</font>
</div>

#### Target Domain: Quadruped with Low Friction

##### Direct Transfer
<div align="center">
    <img src="media/source_expert_in_target.gif" width="200"/><img src="media/bc.gif" width="200"/>
    <br/>
    <font>source expert and behavior cloning in target domain</font>
</div>

Both of quadruped robots overturn on the groud with low friction.

##### Our Method: DYNAIL
<div align="center">
    <img src="media/dynail.gif" width="200"/>
    <br/>
    <font>DYNAIL in target domain</font>
</div>

Our method succeeds in quadruped task with low friction.

### Experiments with High-Dimensional Environment Humanoid

#### Source Domain: Humanoid-v3

<div align="center">
    <img src="media/hu_exp_sou.gif" width="200"/>
    <br/>
    <font>source expert in source domain</font>
</div>

#### Target Domain: BrokenHumanoid-v3 (Humanoid-v3 with red broken abdomen joint)

##### Direct Transfer
<div align="center">
    <img src="media/hu_exp_tar.gif" width="200"/><img src="media/hu_bc.gif" width="200"/>
    <br/>
    <font>source expert and behavior cloning in target domain</font>
</div>
Both of the episodes above terminate because of unhealthy conditions.

##### Our Method: DYNAIL
<div align="center">
    <img src="media/hu_dynail.gif" width="200"/>
    <br/>
    <font>DYNAIL in target domain</font>
</div>
Our method succeeds in humanoid task with red broken abdomen.

### Experiments with Maze (breaking assumptions)

#### Source Domain: UMaze-v0

<div align="center">
    <img src="media/maze_source.gif" width="200"/>
    <br/>
    <font>source expert in source domain</font>
</div>


#### Target Domain: IMaze-v0 (Moving the middle wall block to the right)

##### Direct Transfer
<div align="center">
    <img src="media/maze_bc.gif" width="200"/><img src="media/maze_gwil.gif" width="200"/>
    <br/>
    <font>behavior cloning and GWIL in target domain</font>
</div>
Both of the baselines fail to find the way to the goal.

##### Our Method: DYNAIL
<div align="center">
    <img src="media/maze_dynail.gif" width="200"/>
    <br/>
    <font>DYNAIL in target domain</font>
</div>
Our method succeeds in maze task with a moving wall block.

