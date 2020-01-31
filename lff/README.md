How to run an experiment:

python main.py

Run reinforce experiment with inverse propensity scoring:
python main.py --agent reinforce --use\_ips

Run a q learning experiment in an online fashion:
python main.py --agent qlearning --num\_agents 1000 --num\_episodes 1 --online
Note: here num\_agents refers to the number of steps being taken by a single agent, because there is no re-initialization of the agent. num\_episodes represents the batch size.  

Run an actor critic experiment on MountainCar-v0:
python main.py --agent actorcritic --env MountainCar-v0 

Tensorboard graphs are written to 'runs' folder in this directory. Reward and Loss are the two main tracked metrics. The average reward of each generation of agent is also printed to the console.

There is currently no functionality to load model weights and continue training. There is also no batch processing. All neural networks in agents consist of a single fully connected hidden layer.

In offline mode (the default), while the training data is generated offline, the evaluation is still performed online. For the CartPole-v0 environment, an agent is successful if achieves an average reward of 195.0 or higher on 100 newly generated evaluation episodes.

In online mode, there is one newly generated evaluation episode per training step. An online agent in the CartPole-v0 environment is successful if it achieves a running average reward of 195.0 or higher on the last 100 evaluation episodes.

Arguments:
--gamma : discount factor (default: 0.99)
--seed : random seed (default: 543)
--render : render the environment (store\_true)
--record : save records to this location (default: "")
--monitor\_dir : save monitors to this directory (default: "")
--save\_model : save weights to this location (default: "")
--num\_episodes : number of episodes to run per agent (default: 200)
--num\_agents : number of generations of agents to train (default: 10)
--hidden\_size : size of hidden layer (default: 128)
--use\_ips : use ips reweighting (store\_true)
--debug : run in debug mode (store\_true)
--lr : learning rate (default: 1e-2)
--agent : agent type (ActorCritic, Reinforce, QLearning, Random, Fixed) (default: ActorCritic)
--online : train the agent online rather than offline (store\_true)
--env : openAI gym environment (default: CartPole-v0)
--render\_end : render 3 episodes of the final model (store\_true)
