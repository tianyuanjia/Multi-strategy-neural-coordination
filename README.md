# Multi-strategy-neural-coordination

## Abstract
Making intelligent and efficient decisions like humans faces significant challenges in complex and dynamic environments. Traditional single-strategy methods often struggle with task adaptability and are trapped in the exploration-exploitation trade-off dilemma. It is revealed that the prefrontal cortex of the brain possesses the ability to concurrently monitor multiple strategies and engage in concurrent decision reasoning. Motivated by this, we propose a novel scheme, namely Multi-Strategy Neural Coordinator (MSNC), which effectively mimics human reliability inference. Specifically, a general Concurrent Inference Exploration-Exploitation (CIEE) module is designed, which can eliminate the exploration-exploitation dilemma simply and efficiently. In addition, with the designed MSNC, concurrent optimal actions are generated to adapt to tasks in complex environments. To illustrate the optimization performance, we conduct a case study focusing on motion planning tasks in high-dimensional continuous spaces. The results show that MSNC outperforms the state-of-the-art benchmarks in terms of three representative metrics, which confirms the potential of concurrent multi-strategy coordination to enhance the intelligence and adaptability of decision-making in complex environments.

## Installation 
install pytorch
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch  
#install torch geometric, refer to https://github.com/pyg-team/pytorch_geometric  
pip install pybullet jupyterlab transforms3d matplotlib shapely descartes  

## Reference
[gnn motion planning](https://github.com/rainorangelemon/gnn-motion-planning)  
[NEXT learning to plan](https://github.com/NeurEXT/NEXT-learning-to-plan)   
[PyBullet planning](https://github.com/caelan/pybullet-planning)  
[PyBullet](https://github.com/bulletphysics/bullet3)
