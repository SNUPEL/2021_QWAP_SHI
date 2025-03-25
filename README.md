# 2021_QWAP_SHI

|                 Developer                |               Developer               |                
| :--------------------------------------: | :-----------------------------------: | 
| [Young-in Cho](https://github.com/Youngin-Cho) | [Seung-heon Oh](https://github.com/hyunjinei) |
|         🧑‍💻 AI-Development               |       🧑‍💻 AI-Development               |                

<br>

## Project Overview

- Participants
    - Samsung Heavy Industry (SHI)
- History
    - 1st development: 2021.05 ~ 2021.08
    - 2nd development: 2023.06 ~ 2023.12
    - 3rd development: 2024.05 ~ 2024.06

## Project Introduction
We develop a quay wall allocation algorithm for post-stage outfitting processes in shipyards based on deep reinforcement learning
<br>
The quay wall allocation problem (QWAP) is modeled as a flexible job shop scheduling problem (FJSP) with preemption and machine preference. The QWAP is formulated as a sequential decision-making problem based on Markov decision process. The scheduling agent is trained 

<img src="figure/image-1.png"/>

<br>

## Main Function

### 1️⃣ Overall framework
<img src="figure/image-2.png"/>

<br>

### 2️⃣ Markov decision process

#### 2.1 State
- a compound state representation composed of a heterogeneous graph and auxiliary matrix
    - **heterogeneous graph**: a modified disjunctive graph for FJSP
        - nodes representing quay walls and operations
        - edges representing low preference / high preference / precedence constraints
    - **auxiliary matrix**: a matrix for predicting the effects of the scheduling actions

#### 2.2 Action
- a combination of the vessel and quay-wall (machine assignment and job sequencing)
    - **candidate vessels**
        - newly launched vessels from the docks
        - vessels returning from sea trials
        - vessels waiting at sea owing to the shortage of quay-walls
        - vessels that need to be reallocated due to interruption
    - **candidate quay walls**
        - empty quay walls
        - occupied quay walls with preemption allowed

#### 2.3 Reward
- minimization of the total cost in the post-stage outfitting process
- a sum of three cost-related rewards
    - **penalty cost**: the penalty cost for the delay in the delivery of vessels
    - **moving cost**: the cost of moving the vessels
    - **loss cost**: the additional processing cost

<br>

### 3️⃣ DES-based learning environment
- DES model of the post-stage outfitting process in shipyards
- state transition that takes the action of the agent as the input and calculates the next state and reward.

<br>

### 4️⃣ Scheduling agent with PPO algorithm
#### 4.1 Network Structure
- **Representation module**
    - Two types of latent representation are extracted from the heterogeneous graphs and auxiliary matrix, respectively
    - For heterogeneous graphs, the embedding vectors of nodes are generated using the relational information between nodes
    - For an auxiliary matrix, the embedding vectors for combinations of quay-walls and vessels are generated using the MLP layers 
- **Aggregation module**
    - Input vectors for the output model are generated based on the embedding vectors from the representation module
- **Output module**
    - The actor layers calculate the probability distribution over actions $\pi_{\theta} (\cdot|s_t)$
    - The critic layers calculate a approximate state-value function $V_{\pi_{\theta}} (s_t)$, respectively

#### 4.2 Reinforcement Learning Algorithm
- **PPO(proximal policy optimization)**
    - Policy-based reinforcement learning algorithm
