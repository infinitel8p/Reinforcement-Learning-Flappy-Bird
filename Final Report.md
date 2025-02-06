# Reinforcement Learning for Flappy Bird

**Bochum PfMML24/25 Custom Project**  
**German Litvinov**  
Department of Applied Computer Science  
*Ruhr University Bochum*  
[German.Litvinov@ruhr-uni-bochum.de](mailto:German.Litvinov@ruhr-uni-bochum.de)

**Ludovico Ferrara**  
Department of Applied Computer Science  
*Ruhr University Bochum*  
[ludovico.ferrara@ruhr-uni-bochum.de](mailto:ludovico.ferrara@ruhr-uni-bochum.de)


## Abstract

In this project, we explore the application of reinforcement learning, specifically Deep Q-Learning (DQL), to train an agent capable of playing the game Flappy Bird. Flappy Bird is a deceptively simple yet challenging side-scrolling game where the player (or agent) controls a bird navigating through gaps between pipes, with the goal of maximizing the number of pipes passed without collision. The agent faces unique challenges such as sparse rewards, high penalties for failure, and the need for precise control with minimal actions (flap or do nothing).

Through extensive experimentation, we observed that the agent performed significantly better in the pyGame environment, achieving higher scores and longer survival durations compared to the gymnasium environment, where performance plateaued early. These differences highlight the impact of environment dynamics and state representation on reinforcement learning performance. Despite time and computational constraints limiting full hyperparameter optimization and architectural exploration, the project successfully demonstrates the feasibility of applying Deep Q-Learning to dynamic, real-time games like Flappy Bird.

## 1 Key Information to include
- Mentor: Prof. Dr. Nils Jansen
- External Collaborators: None
- Sharing project: Yes

## 2 Introduction
In this project, we aimed to train a reinforcement learning agent to play the game **Flappy Bird**.  
Flappy Bird is a side-scrolling game in which the player controls a bird and attempts to fly between pipes without hitting them. The player is awarded a point for each set of pipes passed. 
As soon as a pipe is hit, the game is over.  

Our approach focuses on Deep Q-Learning (DQL), a reinforcement learning technique where a neural network is used to approximate Q-values. These are the expected future rewards for taking specific actions in given states.
This allows the agent to learn a policy that maximizes the expected cumulative reward over time by polishing it's strategies through trial and error.

The game itself and it's environment are relatively simple, and while it's easy to learn due to its limited mechanics - consisting of just two actions: flap or do nothing and fall - it presents quite the challenge to achieve a high score or even train an agent that can play the game well due to its unforgiving nature where a single mistimed flap can result in a game over.

The difficulty in training a model on this 'simple' game arises from several factors:
- the sparse reward structure (points are only earned after successfully passing a pipe so creating a good reward function is crucial to guide the agent's learning)
- the high penalty for failure (immediate game over upon collision, which makes it hard for the agent to explore the state space)   
- and the need for continuous, precise control with limited available actions (flap or do nothing)

Through this project, we aimed to explore the challenges of applying Deep Q-Learning to such a dynamic environment, analyze the agent's learning process, and evaluate the effectiveness of different strategies to improve performance.

## 3 Related Work
Reinforcement learning has been applied to a variety of games, from simple grid-world environments to complex video games. Since we are focusing on Flappy Bird, we will discuss related work that has applied reinforcement learning to this specific game.

Since our focus is on Flappy Bird, we will concentrate on related work that has specifically applied reinforcement learning techniques to this game. 

1. [DeepLearningFlappyBird](https://github.com/yenchenlin/DeepLearningFlappyBird) by [Yen-Chen Lin](https://github.com/yenchenlin)
    >This project follows the description of the Deep Q Learning algorithm described in Playing Atari with Deep Reinforcement Learning [...] and shows that this learning algorithm can be further generalized to the notorious Flappy Bird.

    This was our first source of information for the project, as it provides a good starting point for understanding how to apply Deep Q-Learning to Flappy Bird. The project uses a simple Deep Q-Network to train an agent to play the game.
    The author provides a detailed explanation DQN and how it can be applied to Flappy Bird, as well as the code to train the agent. 

2. [Using a Deep Neural Network to Play Flappy Bird via Reinforcement Learning](https://nathanbaileyw.medium.com/using-a-deep-neural-network-to-play-flappy-bird-via-reinforcement-learning-cea639053768) by [Nathan Bailey](https://nathanbaileyw.medium.com/)
    >We solve the reinforcement learning problem using a Deep-Q Network. This is a common solution to reinforcement learning problems. To understand where these networks originate from we first must understand Markov Decision Processes (MDPs).

    This article provides a good overview of the theory behind Deep Q-Learning and how it can be applied to Flappy Bird. It also provides a detailed explanation of the code and theory used to train the agent, as well as the results obtained.
    This was a good source of information for understanding the theory behind Deep Q-Learning and how it can be applied to Flappy Bird.

3. [Flappy-bird-deep-Q-learning-pytorch](https://github.com/vietnh1009/Flappy-bird-deep-Q-learning-pytorch) by [Viet Nguyen](https://github.com/vietnh1009)

    This project provides a PyTorch implementation of Deep Q-Learning for Flappy Bird. Anothere good source of reference for our project to double check our implementation and see different approaches to the problem.

4. [Implement Deep Q-Learning with PyTorch and Train Flappy Bird!](https://www.youtube.com/watch?v=arR7KzlYs4w&ab_channel=JohnnyCode) by [Johnny Code](https://www.youtube.com/@johnnycode)

    This video tutorial provides a step-by-step guide on how to implement Deep Q-Learning with PyTorch and train an agent to play Flappy Bird. This resource helped understanding how to implement Deep Q-Learning for Flappy Bird and solidify our understanding of the theory behind it.

## 4 Approach
Our approach utilizes the **Deep Q-Learning (DQL)** algorithm to train an agent capable of playing Flappy Bird. DQL is a reinforcement learning technique that approximates the optimal action-value function (Q-function) using a neural network. This enables the agent to learn the best actions to take in various game states to maximize cumulative rewards over time.

The agent interacts with the game environment by selecting actions, observing the resulting state transitions, and receiving feedback in the form of rewards. Through continuous trial and error, the agent refines its policy to achieve higher scores by avoiding obstacles and surviving longer.

### 4.1 Deep Q-Learning (DQL) Algorithm
At the core of our approach lies the Q-Learning algorithm, which estimates the value of taking a particular action in a given state. The update rule for Q-learning is based on the Bellman equation:

$$
Q(s, a) = r + \gamma \max_{a'} Q(s', a')
$$

Our implementation mirrors the Bellman equation as follows:
```python
expected_state_action_values = (next_state_action_values * self.GAMMA) + reward
```
For which we have:
- **r** = `reward`
- **γ** = `self.GAMMA`
- $\max_{a'} Q(s', a')$ = `next_state_action_values.max(1)[0]`

### 4.2 Neural Network Architecture
- Our DQN consists of the following components:
    - **Input Layer**: Receives the state representation as input (explicit pipe & player features)
    - **Hidden Layers**: Consist of five fully connected layers, each with ReLU6 activation:
        - **Layer 1**: 64 
        - **Layer 2**: 128
        - **Layer 3**: 256
        - **Layer 4**: 512
        - **Layer 5**: 512
    - **Output Layer**: Outputs the Q-values for each available action
        - `1` → Perform a flap
        - `0` → Do nothing

- For the Dueling DQN, the network architecture is split into two streams:
    - **State-Value Stream**: Estimates the value of the current state  
    - **Advantage Stream**: Estimates the advantage of taking a specific action $A(s, a)$  

    The final Q-values are computed as:

$$
Q(s, a) = V(s) + \left( A(s, a) - \max_{a} A(s, a) \right)
$$

```python
# V(s)
state_values = self.state_values(x)
# A(s, a)           
advantages = self.advantages(x)
# Q(s, a)
output = state_values + (advantages - torch.max(advantages, dim=1, keepdim=True)[0])
```

### 4.3 Training Process
To stabilize training and improve sample efficiency, we implemented an experience replay buffer. This buffer stores past experiences in the form of tuples (as shown in [5.1](#51-data)).
During training, mini-batches of 32 experiences are randomly sampled from this buffer to break the correlation between surrounding experiences.

We used a target network to stabilize training by periodically updating the target network weights with the policy network weights. This helps prevent the target Q-values from oscillating during training.

### 4.4 Hyperparameters
- **Learning Rate**: `lr=1e-4`
- **Discount Factor (γ)**: `GAMMA=0.99`
- **Epsilon (ε)**: `EPS_START=1.0`
- **Epsilon Decay**: `EPS_DECAY_VALUE=0.999995`
- **Epsilon Minimum**: `EPS_END=0.05`   
- **Batch Size**: `BATCH_SIZE=32`
- **Memory Size**: `MEMORY_SIZE=100000`
- **Target Update Rate**: `TAU=0.005`

These hyperparameters were chosen after testing different values and observing the agent's performance during training. Additionally, we utilized screen recording tools to capture the agent's gameplay for analysis. The training process also included periodic saving of model checkpoints and performance metrics to evaluate the agent's progress over time.

## 5 Experiments
During our work we unfortunately did not have enough time and computational power to conduct all of planned experiments. Our initial plan was to test different network architectures, such as Double DQN and Double Dueling DQN and tweak their structures, as well as try out different parameters. 

The training was done in two environments - the **gymnasium** and **pyGame** environments - which had different state representations and rewards. This led to different training results (which where ultimately also the reason to switch from gymnasium to pyGame), challenges in each environment and resulted in us not being able to fully explore the potential of the model as we would have liked due to time constraints.

### 5.1 Data

The main source of data is the **memory** of our agent. Every action taken is recorded along with:  
- **Current state**  
- **Next state**  
- **Action**  
- **Reward**  
- **Done flag** (indicating whether the run is finished)  

Each sample is structured as: 
```python
sample = (state, next_state, action, reward, done)
```

### 5.1.1 Environment of Gymnasium  

#### **State Representation**  

The agent's state can be represented in two different ways:  

- **Option 1: LIDAR Sensor Readings**  
    - 180 readings from the **LIDAR sensor**  
    *(Reference Paper: [Playing Flappy Bird Based on Motion Recognition Using a Transformer Model and LIDAR Sensor](https://www.mdpi.com/1424-8220/24/6/1905))*  

- **Option 2: Explicit Pipe & Player Features**  
    **Last Pipe:**  
    - `last_pipe_x` – Horizontal position  
    - `last_pipe_top_y` – Vertical position of the top pipe  
    - `last_pipe_bottom_y` – Vertical position of the bottom pipe  
    
    **Next Pipe:**  
    - `next_pipe_x` – Horizontal position  
    - `next_pipe_top_y` – Vertical position of the top pipe  
    - `next_pipe_bottom_y` – Vertical position of the bottom pipe  
    
    **Next-Next Pipe:**  
    - `next_next_pipe_x` – Horizontal position  
    - `next_next_pipe_top_y` – Vertical position of the top pipe  
    - `next_next_pipe_bottom_y` – Vertical position of the bottom pipe  
    
    **Player Attributes:**  
    - `player_y` – Player’s vertical position  
    - `player_vel` – Player’s vertical velocity  
    - `player_rotation` – Player’s rotation angle  

The **next state** is structured similarly, representing the agent's state after taking an action.  

#### **Action Representation**  
- `1` → Perform a flap  
- `0` → Do nothing  

#### **Reward Function**  
- **`+0.1`** → For every frame the agent stays alive  
- **`+1.0`** → For successfully passing a pipe  
- **`-1.0`** → For dying  
- **`-0.5`** → For touching the top of the screen  

#### **Done Flag**  
- **`True`** → The run is over  
- **`False`** → The run is still active  


--- 
### 5.1.2 Environment of pyGame

#### **State Representation**  
The **state** of the agent is described by the following values:      
**Next Pipe:**  
- `next_pipe_dist_to_player` – Distance to the next pipe  
- `next_pipe_top_y` – Y-position of the top part of the next pipe  
- `next_pipe_bottom_y` – Y-position of the bottom part of the next pipe  

**Next-Next Pipe:**  
- `next_next_pipe_dist_to_player` – Distance to the pipe after the next one  
- `next_next_pipe_top_y` – Y-position of the top part of the next-next pipe  
- `next_next_pipe_bottom_y` – Y-position of the bottom part of the next-next pipe  

**Player Attributes:**  
- `player_y` – Player’s vertical position  
- `player_vel` – Player’s velocity 

The **next state** is defined similarly, but represents the agent's state after taking an action.  

#### **Action Representation**  
- `1` → Perform a jump  
- `0` → Do nothing  

#### **Reward Function**  
- **`-5`** → If the agent loses (falls or hits a pipe)  
- **`0.1`** → For any action taken (to incentivize survival)  
- **`1`** → For successfully passing a pipe  

#### **Done Flag**  
- **`True`** → The run is over  
- **`False`** → The run is still active  

#### **Data Fetching**  
The data is fetched randomly from memory in **batches of 32 samples**. For each sample a **state action value** is calculated with the **policy network** and compared against the **expected state action value** calculated by the **target network**.

### 5.2 Evaluation method
The performance of our model would be evaluated by inspecting the **sum of rewards** for each run or **duration** of each run over the **number of episodes**. For both values there was computed an average value over the last 100 datapoints The corresponding plots were created and updated during the training process and watched upon, to get an idea if the model is progressing.

- **duration** - is a count for how many actions were taken in the run
- **sum of reward** - is the sum of each reward after an action in a given run. Sometimes named 'Score' on plots.

An indicator of good performance would be a plot that showed growth of reward or duration.

### 5.3 Experimental details
We ended up using several upproaches to achieve good model performance. We focused on **Double DQN** and **Dueling Double DQN** architectures and used different environments.

### 5.3.1 Dueling Double DQN with gymnasium
The first environment was a **gymnasium** environment with the flappy bird game.
Both **policy** and **target** networks have the same structure

The training of the agent lasted for about 40 000 episodes and resulted into a **plato** after around 10 000 steps after reaching the average reward of around 3,5. 

![Image](https://github.com/user-attachments/assets/62c09f9b-3a7a-4308-95b5-bbe1230a452c)


### 5.3.2 Double Dueling DQN with pyGame
The second environment was the **pyGame** environment.
Both **policy** and **target** networks have the same structure

This time the training was done for around 11 000 episodes and the agent was able to achieve **high** durations and scores. If the training was not stopped, it seems that the agent would be able to achieve even better results.

![Image](https://github.com/user-attachments/assets/5846c694-de1e-4636-98fb-79993cd24bb0)

### 5.4 Results
#### **Comparison of Training Results**  

| Environment   | Training Duration      | Performance                                             | Observations                           |
| ------------- | ---------------------- | ------------------------------------------------------- | -------------------------------------- |
| **pyGame**    | around 11,000 episodes | High durations & scores                                 | Training shows continuous improvement. |
| **gymnasium** | around 40,000 episodes | Plateau at around 10,000 steps (Avg. reward approx 3.5) | Performance plateaued.                 |

---

#### **Observations**  

- **pyGame Environment**:  
  - The agent achieved high durations and scores after training for around 11,000 episodes.  
  - If training continued, the agent seemed capable of achieving even better results.  

- **gymnasium Environment**:  
  - Training lasted around 40,000 steps, but the agent plateaued after approx. 10,000 steps.  
  - The average reward reached 3.5, and further improvements were not observed.  
  - This suggests possible issues with the environment's dynamics or training hyperparameters.  

---

#### **Conclusion & Future Improvements**  

There could be several potential reasons for the poor **gymnasium** performance:
- More difficult play mode with higher speed, velocity and smaller pipe holes
- Too comlicated state descriprion that imitates lidar
- Bugs in the enironment itself

## 6 Analysis
#### **Agent's behavior**  

- The agent learns to avoid pipes and maximize its survival time, especially in the **pyGame** environment, where performance continuously improves.  
- In **gymnasium**, the agent learns up unitil a certain point but then stops improving.  


#### **When the agent performs good**  

**Pipe avoidance:**  
  - The agent successfully identifies the optimal jump timing to pass through pipes without collisions.  
  - It learns to maintain a stable trajectory.  

**Long survival times:**  
  - The agent tends to closer to the middle of the pipe opening.  
  - In the PyGame environment, it gets better at performing the needed actions, leading to longer survival times.  

---

#### **When the agent performs bad**  

**Early training:**  
  - At the beginning, the agent jumps to often, leading to crashes into pipes and floor.  
  - It deos  not recognize long-term consequences of its actions.

**gymansium**
  - The agent does not perform good in the gymnasium environment

---

#### **Insights**  

**gymnasium:**  
  - The agent could be employing too small networks if the environments game rules are more harsh or the state description too complicated.
  - In case of larger networks longer training times are also required

**Number of episodes**
  - The agent benefits from long training times .

**Reward tuning:**  
  - Rewarding smoother movements or penalizing unnecessary jumps could improve the agent’s efficiency.  

**Improved state representation:**  
  - Using additional environmental cues, such as angle adjustments or future trajectory predictions, could improve decision-making. But could also potentialy increase the need for a larger network.

## 7 Conclusion
We conclude by saying that we were able to successfully train a reinforcement learning agent to play Flappy Bird using the Deep Q-Learning algorithm. The agent was able to learn a policy that allowed it to navigate the game environment and achieve a relatively high score. While the lack of time due to us having to switch environments throughout the project prevented us from fully exploring the potential of the model by finetuningn hyperparameters and testing different network architectures, we were able make a model capable of playing the game.
