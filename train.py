import torch.optim as optim
from models import PolicyNetwork, ComplexPolicyNetwork
from environment import SingleStockTradingEnv
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
import glob
import statistics

envs = []
csv_files = glob.glob('./data/processed_news/*.csv')
for file in csv_files:
    ticker = file.split('/')[-1].split('.')[0]
    envs.append(SingleStockTradingEnv(ticker))

# Initialize the environment and the policy
policy = ComplexPolicyNetwork(envs[0].observation_space.shape[0], envs[0].action_space.n).to('cuda')
optimizer = optim.Adam(policy.parameters(), lr=0.0001)

# Training loop
num_epochs = 500
epochs_to_improve = None

# Initialize tqdm progress bar
pbar = tqdm(range(num_epochs))
losses = []
loss = torch.tensor(0, dtype=torch.float32)
latest_money = [0] * len(envs)
best_money_average = 0

for epoch in pbar:
    for i, env in enumerate(envs):
        log_probs = []
        rewards = []
        state = env.reset()
        done = False
        while not done:
            # Select action
            state_tensor = torch.tensor(state, dtype=torch.float32, device='cuda')
            action_probs = policy(state_tensor)
            #print(action_probs)
            action = torch.multinomial(action_probs, 1).item()
            log_prob = torch.log(action_probs[action])
            log_probs.append(log_prob)
            #print('log prob', log_prob.item())

            # Take a step in the environment
            state, reward, done, _ = env.step(action)
            #print('reward', reward)
            rewards.append(reward)
            pbar.set_description(f"Loss: {loss.item():.4f}, Best Money: {best_money_average:.2f}, Reward: {reward:.4f}")

        # Compute the loss
        loss = -torch.stack([r * log_prob for r, log_prob in zip(rewards, log_probs)]).sum()
        losses.append(abs(loss.item()))
        # Update the progress bar description with the current loss
        
        #print(loss.item())
        
        # Update the policy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if min(losses) == loss.item(): 
            last_best_epoch = epoch
            torch.save(policy.state_dict(), './check_points/best_policy.pth')
            
        latest_money[i] = env.current_money
        
        env.render()
        
    
            
    if statistics.mean(latest_money) > best_money_average:
        best_money_average = statistics.mean(latest_money)
        torch.save(policy.state_dict(), './check_points/best_policy_money.pth')
        
    if epochs_to_improve and last_best_epoch < epoch - epochs_to_improve:
        break
    
    if epoch % 100 == 0:
        torch.save(policy.state_dict(), f'./check_points/policy-{epoch}.pth')
        
# save model
torch.save(policy.state_dict(), './check_points/last_policy.pth')

# After the training loop, plot the losses
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('./figures/loss.png')
plt.close()