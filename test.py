import matplotlib.pyplot as plt
from models import ComplexPolicyNetwork
from environment import SingleStockTradingEnv
import torch

# Load the trained model
env = SingleStockTradingEnv('NVDA',)
model = ComplexPolicyNetwork(env.observation_space.shape[0], env.action_space.n)
model.load_state_dict(torch.load('./check_points/policy-400.pth'))
model.eval()

# Run the model on an episode and record the actions
state = env.reset()
done = False
actions = []
while not done:
    state_tensor = torch.tensor(state, dtype=torch.float32)
    action_probs = model(state_tensor)
    action = torch.multinomial(action_probs, 1).item()
    actions.append(action)
    state, _, done, _ = env.step(action)

env.render()