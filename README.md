# DYNAIL

## Experiments with [realworldrl-suite](https://github.com/google-research/realworldrl_suite)

### Source Domain: Quadruped

<div align="center">
    <img src="media/source_expert.gif" width="200"/>
    <br/>
    <font>source expert in source domain</font>
</div>

### Target Domain: Quadruped with Low Friction

#### Direct Transfer
<div align="center">
    <img src="media/source_expert_in_target.gif" width="200"/><img src="media/bc.gif" width="200"/>
    <br/>
    <font>source expert and behavior cloning in target domain</font>
</div>

#### Our Method: DYNAIL
<div align="center">
    <img src="media/dynail.gif" width="200"/>
    <br/>
    <font>DYNAIL in target domain</font>
</div>

## Additional Experiments with More Demonstrations

We increase the expert demonstrations from 40k to 80k for the existing SOTA methods (SAIL and GWIL). With more expert demonstrations, these baselines improve, yet DYNAIL still has superior performance with only 40k expert samples. We will update these experiments to our final version of the paper.

<div align="center">
    <img src="figures/reb_cus.png" width="400"/><img src="figures/reb_dis1.png" width="400"/><img src="figures/reb_dis2.png" width="400"/>
    <br/>
    <font>SAIL and GWIL with more demonstrations in Ant environments</font>
</div>

<div align="center">
    <img src="figures/reb_brohalf.png" width="400"/><img src="figures/reb_halfobs.png" width="400"/>
    <br/>
    <font>SAIL and GWIL with more demonstrations in HalfCheetah environments</font>
</div>

For computational complexity, it takes a long time for SAIL with 80 expert trajectories (80000 transitions) to train. Thus, the figures above are to be updated until the training of SAIL is finished.
