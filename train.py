# train.py
# Trains the tabular Q-learning agent on data_train.csv for 500 episodes.
# Saves q_table.npy and training_log.csv.


import numpy as np
import pandas as pd
import sys

# Import our custom files
sys.path.append(".")
from yahoo_env   import StockEnv
from yahoo_qagent import QLearningAgent


# ─────────────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────────────
print("Loading data...")
train_df = pd.read_csv("data_train.csv")
print(f"Training on {len(train_df)} days "
      f"({train_df['Date'].iloc[0]} → {train_df['Date'].iloc[-1]})\n")


# ─────────────────────────────────────────────────────
#  CREATE ENVIRONMENT AND AGENT
# ─────────────────────────────────────────────────────
env   = StockEnv(train_df, initial_cash=10_000)
agent = QLearningAgent(
    n_states  = 18,
    n_actions = 3,
    alpha     = 0.1,    # learning rate
    gamma     = 0.99,   # discount factor
    epsilon   = 1.0     # start fully random
)

# ─────────────────────────────────────────────────────
#  TRAINING LOOP
#  Each episode = agent lives through all 1600 days
#  from scratch with fresh £10,000
# ─────────────────────────────────────────────────────
N_EPISODES   = 500
rewards_log  = []     # total reward per episode
portfolio_log = []    # final portfolio value per episode

print("Starting training...\n")
print(f"{'Episode':>8} | {'Avg Reward':>10} | {'Portfolio':>12} | "
      f"{'Epsilon':>8} | {'Trades':>6}")
print("-" * 60)

for episode in range(1, N_EPISODES + 1):

    # Reset environment — fresh £10,000, back to day 0
    state        = env.reset()
    total_reward = 0
    n_trades     = 0   # count how many buy/sell actions taken

    # ── Run through all 1600 trading days ──────────
    while True:

        # 1. Agent picks an action
        action = agent.choose_action(state)

        # 2. Environment executes it and responds
        next_state, reward, done = env.step(action)

        # 3. Agent learns from what just happened
        agent.update(state, action, reward, next_state, done)

        # 4. Count trades (not holds)
        if action in [1, 2]:
            n_trades += 1

        # 5. Move to next day
        total_reward += reward
        state         = next_state

        if done:
            break

    # ── End of episode ──────────────────────────────
    agent.decay_epsilon()
    agent.snapshot()         # save Q-table state for analysis

    rewards_log.append(total_reward)
    portfolio_log.append(env.portfolio_value)

    # Print progress every 50 episodes
    if episode % 50 == 0 or episode == 1:
        avg_reward = np.mean(rewards_log[-50:])
        print(f"{episode:>8} | "
              f"{avg_reward:>10.2f} | "
              f"£{env.portfolio_value:>10,.2f} | "
              f"{agent.epsilon:>8.3f} | "
              f"{n_trades:>6}")

print("-" * 60)
print("\nTraining complete!")


# ─────────────────────────────────────────────────────
#  SAVE RESULTS
# ─────────────────────────────────────────────────────
# Save Q-table
np.save("q_table.npy", agent.Q)
print("Q-table saved → q_table.npy")

# Save training logs
log_df = pd.DataFrame({
    "episode"   : range(1, N_EPISODES + 1),
    "reward"    : rewards_log,
    "portfolio" : portfolio_log
})
log_df.to_csv("training_log.csv", index=False)
print("Training log saved → training_log.csv")


# ─────────────────────────────────────────────────────
#  PRINT FINAL Q-TABLE
# ─────────────────────────────────────────────────────
print()
agent.print_q_table()


# ─────────────────────────────────────────────────────
#  QUICK SUMMARY
# ─────────────────────────────────────────────────────
print(f"\n── Training Summary ─────────────────────────────────")
print(f"Episodes trained      : {N_EPISODES}")
print(f"Final epsilon         : {agent.epsilon:.4f}")
print(f"Best portfolio value  : £{max(portfolio_log):,.2f}")
print(f"Worst portfolio value : £{min(portfolio_log):,.2f}")
print(f"Final portfolio value : £{portfolio_log[-1]:,.2f}")
print(f"Starting cash         : £10,000.00")
profit = portfolio_log[-1] - 10_000
print(f"Profit / Loss (ep500) : £{profit:+,.2f}  "
      f"({profit/10_000*100:+.1f}%)")
