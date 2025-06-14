# Robot-Learning-Reinforcement-Learning-for-Control-and-Custom-Environment-Design
Robot Learning: Reinforcement Learning for Control and Custom Environment Design
This repository features practical implementations in Reinforcement Learning (RL), including actor-critic algorithms for continuous control and the design of custom OpenAI Gym environments for various task complexities.

Project Overview
This project focuses on the hands-on application and evaluation of Reinforcement Learning algorithms. Key coding components include:

Actor-Critic Methods: Implementation of an Actor-Critic framework, where a neural network-based critic learns to approximate the value function using Temporal Difference (TD) errors. This robust algorithm is applied to solve the classic Cart-Pole control problem.
Custom Gym Environment Design: Development of a novel OpenAI Gym-compatible environment, meticulously defining its state and action spaces, reward structure, and episode termination conditions.
RL Agent Training and Evaluation: Training a Reinforcement Learning agent within the custom environment, demonstrating learning with sparse binary rewards and subsequently showcasing accelerated learning through the strategic application of reward shaping techniques.
Technical Setup
The development environment for this project was primarily Google Colab. All necessary Python dependencies are managed and installed via the provided code.

To set up the environment:

Clone this repository:
Bash

git clone [git@github.com:hesamazj/Robot-Learning-Reinforcement-Learning-for-Control-and-Custom-Environment-Design.git]
Navigate into the cloned directory. The included Jupyter notebooks or Python scripts will automatically handle the installation of required Python packages upon execution.
Core Implementations and Demonstrations
Actor-Critic Methods (Cart-Pole Control)
This section implements and evaluates an Actor-Critic algorithm to learn a policy for balancing the Cart-Pole.

Actor-Critic Architecture:
The Actor component is a neural network that parametrizes the policy π 
θ
​
 , outputting action probabilities for a given state.
The Critic component is a separate neural network that approximates the state-value function V 
π 
θ,ϕ
​
 
​
 (s). It is trained by minimizing the squared Temporal Difference (TD) error, where the TD error is calculated as TD 
error
​
 :=r(s 
t
​
 ,a 
t
​
 )+γV 
π 
θ
​
 ,ϕ
​
 (s 
t+1
​
 )−V 
π 
θ
​
 ,ϕ
​
 (s 
t
​
 ). The critic's loss function is min 
ϕ
​
 ∑ 
i,t
​
 (V 
π 
θ
​
 ,ϕ
​
 (s 
it
​
 )−y 
it
​
 ) 
2
 , where y 
t
​
 =r(s 
t
​
 ,a 
t
​
 )+γV 
π 
θ
​
 ,ϕ
​
 (s 
t+1
​
 ).
Variance Reduction: The policy gradient update incorporates discounted returns and a learned value function baseline V 
π 
θ,ϕ
​
 
​
 (s 
it
​
 ) to reduce variance in the gradient estimates.
Cart-Pole Environment (OpenAI Gym):
State Space: A 4-dimensional continuous observation space representing cart position, cart velocity, pole angle, and pole angular velocity.
Action Space: A 2-dimensional discrete action space: push cart left (0) or push cart right (1).
Reward Function: A reward of +1 is granted for each timestep the pole remains upright.
Termination Conditions: An episode concludes if the pole angle exceeds ±12 
∘
 , the cart moves beyond ±2.4 units from the center, or the episode length reaches 200 steps.
Implementation Details:
The Critic class within p2_a2c.ipynb requires completion, implementing a neural network to predict state values.
The train_actor_critic function orchestrates the training loop, managing episodes, collecting experiences, and applying updates to both actor and critic networks.
Running the Cart-Pole Demonstration:
Bash

# Ensure you have completed the `Critic` class implementation in `p2_a2c.ipynb`.
# Execute the relevant cells in `p2_a2c.ipynb` to train and test the Actor-Critic network.
Expected Performance: A well-trained agent should consistently achieve episode scores above 100, demonstrating stable control of the Cart-Pole for extended durations.
Custom MDP and RL Agent
This section involves designing a custom OpenAI Gym environment and training an RL agent to solve the defined task.

Custom Gym Environment Design (Chosen Scenario: [Specify Scenario A or B]):
Scenario A (Apple Slicing Gridworld):
Environment: An N×N Gridworld.
State Space: A representation capturing the agent's (x,y) coordinates, the apple's (x,y) coordinates, the plate's (x,y) coordinates, and a binary status indicating whether the apple has been sliced.
Action Space: Discrete actions such as MOVE_NORTH, MOVE_SOUTH, MOVE_EAST, MOVE_WEST, and INTERACT (to slice apple or place slices on plate).
Reward Function: An initial sparse binary reward (+1) is given exclusively upon successful completion of the entire task (apple sliced and all slices on the plate).
Termination Condition: The episode concludes upon task completion or reaching a predefined maximum number of steps.
Scenario B (Partially Observable Goal-Reaching Gridworld):
Environment: An N×N Gridworld where the agent starts at a fixed bottom-left corner.
State Space: The agent's current (x,y) coordinates. The goal cell's (x,y) coordinates are randomly initialized per episode but remain constant throughout that episode.
Action Space: Discrete movement actions (e.g., UP, DOWN, LEFT, RIGHT).
Reward Function: A positive reward is granted when the agent reaches the goal state. Initially, this is a sparse binary reward.
Partial Observability: The agent's observation is limited solely to its current location, without direct knowledge of the goal's location.
Reset Mechanism: The agent is reset to its initial state (bottom-left) after a fixed number of steps within an episode (e.g., every 15 steps in a 60-step rollout), continuing the episode.
RL Algorithm for Training (Chosen Algorithm: [Specify which algorithm you used, e.g., Q-Learning, Deep Q-Network (DQN), Policy Gradient, etc.]):
Implementation details of the chosen RL algorithm, including its core components (e.g., Q-table updates for Q-Learning, neural network architecture for DQN, policy parametrization for Policy Gradient).
Sparse Reward Training: Initial training demonstrates the agent's learning performance solely with the sparse, terminal reward.
Reward Shaping Integration: Implementation of reward shaping techniques (e.g., small positive rewards for moving towards the goal, penalties for illegal moves or steps taken) and a clear demonstration of how these techniques accelerate learning or enable task solution.
Evaluation and Analysis:
Visualizations of RL training metrics (e.g., average episode rewards over time, loss curves) and task success rate curves are provided to illustrate the agent's learning progress and final performance.
References
[1] Alexey Dosovitskiy, German Ros, Felipe Codevilla, Antonio Lopez, and Vladlen Koltun. Carla: An open urban driving simulator. In Conference on Robot Learning, 2017.
[2] Zhangjie Cao, Erdem Bıyık, Woodrow Z Wang, Allan Raventos, Adrien Gaidon, Guy Rosman, and Dorsa Sadigh. Reinforcement learning based control of imitative policies for near-accident driving. Proceedings of Robotics: Science and Systems (RSS), 2020.
[3] Jason Kong, Mark Pfeiffer, Georg Schildbach, and Francesco Borrelli. Kinematic and dynamic vehicle models for autonomous driving control design. In IEEE Intelligent Vehicles Symposium (IV), 2015.
[4] Dean A Pomerleau. Alvinn: An autonomous land vehicle in a neural network. Advances in Neural Information Processing Systems, 1988.
[5] Felipe Codevilla, Matthias Müller, Antonio López, Vladlen Koltun, and Alexey Dosovitskiy. End-to-end driving via conditional imitation learning. In IEEE International Conference on Robotics and Automation (ICRA), 2018.