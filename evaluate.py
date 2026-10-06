# evaluate.py
# Final evaluation of the trained Q-table on unseen test data.
# Compares against a buy-and-hold baseline and saves results.png.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys

sys.path.append(".")
from yahoo_env    import StockEnv
from yahoo_qagent import QLearningAgent


# ─────────────────────────────────────────────────────
#  LOAD DATA AND SAVED Q-TABLE
# ─────────────────────────────────────────────────────
print("Loading data and trained Q-table...")
train_df = pd.read_csv("data_train.csv")
test_df  = pd.read_csv("data_test.csv")
log_df   = pd.read_csv("training_log.csv")

# Recreate agent and load saved Q-table
agent         = QLearningAgent()
agent.Q       = np.load("q_table.npy")
agent.epsilon = 0.0    # NO exploration during evaluation — pure learned policy

print(f"Test period : {test_df['Date'].iloc[0]} → {test_df['Date'].iloc[-1]}")
print(f"Test days   : {len(test_df)}")
print(f"Q-table loaded from q_table.npy\n")


# ─────────────────────────────────────────────────────
#  RUN AGENT ON TEST DATA
# ─────────────────────────────────────────────────────
print("Running agent on unseen test data...")
test_env   = StockEnv(test_df, initial_cash=10_000)
state      = test_env.reset()

agent_portfolio = [10_000]   # track value every day
actions_taken   = []
n_trades        = 0

while True:
    action                   = agent.choose_action(state)
    held_before              = test_env.shares > 0
    next_state, reward, done = test_env.step(action)
    actions_taken.append(action)

    # Count only trades that actually execute:
    # Buy while already holding, or Sell with no shares, does nothing
    if (action == 1 and not held_before) or (action == 2 and held_before):
        n_trades += 1

    agent_portfolio.append(test_env.portfolio_value)
    state = next_state

    if done:
        break

agent_final  = test_env.portfolio_value
agent_return = (agent_final - 10_000) / 10_000 * 100


# ─────────────────────────────────────────────────────
#  BUY AND HOLD BASELINE
#  Buy on day 1, hold until the last day — no strategy
#  This is what a passive investor would achieve
# ─────────────────────────────────────────────────────
print("Computing buy-and-hold baseline...")
first_price   = float(test_df["Close"].iloc[0])
last_price    = float(test_df["Close"].iloc[-1])
shares_bought = 10_000 // first_price
leftover_cash = 10_000 - shares_bought * first_price

# Track buy-and-hold portfolio value every day
bh_portfolio  = []
for price in test_df["Close"]:
    bh_portfolio.append(shares_bought * float(price) + leftover_cash)

bh_final  = bh_portfolio[-1]
bh_return = (bh_final - 10_000) / 10_000 * 100


# ─────────────────────────────────────────────────────
#  PRINT RESULTS
# ─────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  FINAL RESULTS — TEST SET (unseen data)")
print("=" * 55)
print(f"  Period          : {test_df['Date'].iloc[0]} → "
      f"{test_df['Date'].iloc[-1]}")
print(f"  Test days       : {len(test_df)}")
print(f"  Starting cash   : £10,000.00")
print("-" * 55)
print(f"  RL Agent final  : £{agent_final:>10,.2f}  "
      f"({agent_return:+.1f}%)")
print(f"  Buy & Hold final: £{bh_final:>10,.2f}  "
      f"({bh_return:+.1f}%)")
print("-" * 55)
diff = agent_return - bh_return
if diff > 0:
    print(f"  Agent BEAT buy-and-hold by {diff:.1f} percentage points")
else:
    print(f"  Agent UNDERPERFORMED buy-and-hold by {abs(diff):.1f} pp")
print(f"  Total trades made : {n_trades}")
print("=" * 55)

# Action breakdown
action_names  = {0: "Hold", 1: "Buy", 2: "Sell"}
action_counts = {0: 0, 1: 0, 2: 0}
for a in actions_taken:
    action_counts[a] += 1

print("\nAction breakdown on test set:")
for a, name in action_names.items():
    pct = action_counts[a] / len(actions_taken) * 100
    bar = "█" * int(pct / 2)
    print(f"  {name:4s}: {action_counts[a]:4d} days  ({pct:5.1f}%)  {bar}")


# ─────────────────────────────────────────────────────
#  CHART 1 — Portfolio value over test period
# ─────────────────────────────────────────────────────
print("\nGenerating charts...")

fig = plt.figure(figsize=(14, 12))
fig.suptitle("Q-Learning Stock Trading Agent — Results",
             fontsize=15, fontweight="bold", y=0.98)

gs  = gridspec.GridSpec(3, 2, figure=fig,
                        hspace=0.45, wspace=0.35)

# Chart 1 — Test portfolio
ax1 = fig.add_subplot(gs[0, :])   # full width top row
dates = list(test_df["Date"])

ax1.plot(range(len(bh_portfolio)),
         bh_portfolio,
         label=f"Buy & Hold  ({bh_return:+.1f}%)",
         color="#5DCAA5", linewidth=1.8, linestyle="--")

ax1.plot(range(len(agent_portfolio)),
         agent_portfolio,
         label=f"RL Agent    ({agent_return:+.1f}%)",
         color="#D85A30", linewidth=2.0)

ax1.axhline(y=10_000, color="gray",
            linestyle=":", linewidth=1.0, alpha=0.7)
ax1.set_title("Portfolio value — test set (unseen data)",
              fontsize=12, pad=8)
ax1.set_xlabel("Trading days into test period")
ax1.set_ylabel("Portfolio value (£)")
ax1.legend(fontsize=10)
ax1.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"£{x:,.0f}"))
ax1.grid(True, alpha=0.3)


# ─────────────────────────────────────────────────────
#  CHART 2 — Training reward per episode
# ─────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])

ax2.plot(log_df["episode"], log_df["reward"],
         color="#7F77DD", linewidth=0.8, alpha=0.5,
         label="Raw reward")

# Smoothed line (rolling average)
smooth = log_df["reward"].rolling(20, min_periods=1).mean()
ax2.plot(log_df["episode"], smooth,
         color="#3C3489", linewidth=2.0,
         label="20-ep average")

ax2.axhline(y=0, color="gray",
            linestyle="--", linewidth=1.0, alpha=0.7)
ax2.set_title("Training reward per episode", fontsize=11, pad=8)
ax2.set_xlabel("Episode")
ax2.set_ylabel("Total reward")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)


# ─────────────────────────────────────────────────────
#  CHART 3 — Training portfolio per episode
# ─────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])

ax3.plot(log_df["episode"], log_df["portfolio"],
         color="#D85A30", linewidth=1.2, alpha=0.7)
ax3.axhline(y=10_000, color="gray",
            linestyle="--", linewidth=1.0, alpha=0.7,
            label="Starting cash")
ax3.set_title("Portfolio value during training", fontsize=11, pad=8)
ax3.set_xlabel("Episode")
ax3.set_ylabel("Portfolio value (£)")
ax3.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"£{x:,.0f}"))
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)


# ─────────────────────────────────────────────────────
#  CHART 4 — Q-table heatmap
# ─────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[2, :])   # full width bottom row

# Build readable state labels
state_labels = []
rsi_labels   = ["RSI<40", "RSI40-60", "RSI>60"]
ma_labels    = ["MA<0.98", "MA0.98-1.02", "MA>1.02"]
hold_labels  = ["No shares", "Holding"]

for rsi_i in range(3):
    for ma_i in range(3):
        for h_i in range(2):
            state_labels.append(
                f"{rsi_labels[rsi_i]}\n{ma_labels[ma_i]}\n{hold_labels[h_i]}")

q_display = agent.Q.T   # transpose: rows=actions, cols=states

im = ax4.imshow(q_display, cmap="RdYlGn",
                aspect="auto", vmin=-5, vmax=20)

ax4.set_xticks(range(18))
ax4.set_xticklabels(state_labels, fontsize=6.5)
ax4.set_yticks([0, 1, 2])
ax4.set_yticklabels(["Hold", "Buy", "Sell"], fontsize=10)
ax4.set_title("Q-table heatmap — green = agent prefers this action "
              "in this state", fontsize=11, pad=8)

# Add Q values as text inside cells
for i in range(3):
    for j in range(18):
        ax4.text(j, i, f"{q_display[i, j]:.1f}",
                 ha="center", va="center",
                 fontsize=6, color="black")

plt.colorbar(im, ax=ax4, shrink=0.6, label="Q value")


# ─────────────────────────────────────────────────────
#  SAVE CHART
# ─────────────────────────────────────────────────────
plt.savefig("results.png", dpi=150, bbox_inches="tight")
print("Chart saved → results.png")
plt.show()

print("\nFiles produced:")
print("  results.png       ← charts for your report")
print("  q_table.npy       ← trained Q-table")
print("  training_log.csv  ← episode-by-episode training data")