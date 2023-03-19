# DYNAIL

## Experiments with [realworldrl-suite](https://github.com/google-research/realworldrl_suite)

### Source Domain: Quadruped

<div align="center">
    <img src="media/source_expert.gif" width="200"/>
    <br/>
    <font>source expert in source domain</font>
</div>

### Target Domain: Quadruped with Low Friction

<div align="center">
    <img src="media/source_expert_in_target.gif" width="200"/><img src="media/bc.gif" width="200"/>
    <br/>
    <font>source expert and behavior cloning in target domain</font>
</div>


<div align="center">
    <img src="media/target_expert.gif" width="200"/><img src="media/dynail.gif" width="200"/>
    <br/>
    <font>target expert and dynail in target domain</font>
</div>

## Additional Experiments with More Demonstrations

<div align="center">
    <img src="figure/reb_cus.png" width="200"/><img src="figure/reb_dis1.png" width="200"/><img src="figure/reb_dis2.png" width="200"/>
    <br/>
    <font>SAIL and GWIL with more demonstrations in Ant environments</font>
</div>

<div align="center">
    <img src="figure/reb_brohalf.png" width="200"/><img src="figure/reb_halfobs.png" width="200"/>
    <br/>
    <font>SAIL and GWIL with more demonstrations in HalfCheetah environments</font>
</div>

For computational complexity, it takes a long time for SAIL with 80 expert trajectories (80000 transitions) to train. Thus, the figure is to be updated until the training of SAIL is finished.
