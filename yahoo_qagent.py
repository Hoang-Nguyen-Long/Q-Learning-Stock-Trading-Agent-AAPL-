# agent.py


import numpy as np


class QLearningAgent:
    """
    Q-table shape: (18 states, 3 actions)
    Actions: 0 = Hold, 1 = Buy, 2 = Sell
    """

    def __init__(self,
                 n_states  = 18,    # 3 rsi_bins × 3 ma_bins × 2 holding
                 n_actions = 3,     # Hold / Buy / Sell
                 alpha     = 0.1,   # learning rate — how fast to update
                 gamma     = 0.99,  # discount factor — how much future matters
                 epsilon   = 1.0):  # exploration — starts at 100% random

        # Q-table: all zeros at the start — agent knows nothing
        self.Q         = np.zeros((n_states, n_actions))

        self.alpha     = alpha
        self.gamma     = gamma
        self.epsilon   = epsilon

        self.eps_min   = 0.01    # never go below 1% exploration
        self.eps_decay = 0.995   # multiply epsilon by this each episode

        # Track Q-table history for analysis later
        self.q_history = []

    # ─────────────────────────────────────────────────────
    #  CHOOSE ACTION
    #  Epsilon-greedy: explore randomly OR exploit Q-table
    # ─────────────────────────────────────────────────────
    def choose_action(self, state):
        """
        With probability epsilon  → pick a random action (explore)
        With probability 1-epsilon → pick the best known action (exploit)
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(3)           # random: 0, 1, or 2
        return int(np.argmax(self.Q[state]))      # greedy: highest Q value

    # ─────────────────────────────────────────────────────
    #  UPDATE  — the Bellman equation
    #  Called after every single trading day
    # ─────────────────────────────────────────────────────
    def update(self, state, action, reward, next_state, done):
        """
        Bellman equation:
        Q(s,a) ← Q(s,a) + α × [ r + γ × max Q(s') − Q(s,a) ]

        In plain English:
        Nudge Q(s,a) a little bit toward what we now think it should be.
        What we think it should be = reward we just got
                                   + discounted best future value
        """
        # If episode is done, no future reward exists
        future = 0.0 if done else np.max(self.Q[next_state])

        # What we wish Q(state, action) had been
        target = reward + self.gamma * future

        # Nudge current Q value toward that target by alpha (10%)
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])

    # ─────────────────────────────────────────────────────
    #  DECAY EPSILON
    #  Called once per episode — reduce exploration gradually
    # ─────────────────────────────────────────────────────
    def decay_epsilon(self):
        """
        Early training:  epsilon near 1.0 → mostly random exploration
        Late training:   epsilon near 0.01 → mostly learned strategy
        """
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

    # ─────────────────────────────────────────────────────
    #  SAVE Q-TABLE SNAPSHOT
    #  Call periodically to track how Q values evolve
    # ─────────────────────────────────────────────────────
    def snapshot(self):
        self.q_history.append(self.Q.copy())

    # ─────────────────────────────────────────────────────
    #  PRINT Q-TABLE  — human readable
    # ─────────────────────────────────────────────────────
    def print_q_table(self):
        rsi_labels  = ["RSI-low ", "RSI-mid ", "RSI-high"]
        ma_labels   = ["MA-below", "MA-near ", "MA-above"]
        hold_labels = ["no-shares", "holding  "]

        print("\n── Q-Table (Hold | Buy | Sell) ──────────────────────────")
        print(f"{'State':<35} {'Hold':>7} {'Buy':>7} {'Sell':>7}  Best")
        print("-" * 65)

        for rsi_i in range(3):
            for ma_i in range(3):
                for h_i in range(2):
                    state = rsi_i * 6 + ma_i * 2 + h_i
                    q     = self.Q[state]
                    best  = ["Hold", "Buy ", "Sell"][np.argmax(q)]
                    label = f"{rsi_labels[rsi_i]} + {ma_labels[ma_i]} + {hold_labels[h_i]}"
                    print(f"{label:<35} "
                          f"{q[0]:>7.3f} "
                          f"{q[1]:>7.3f} "
                          f"{q[2]:>7.3f}  {best}")
        print("-" * 65)


# ─────────────────────────────────────────────────────────
#  MANUAL TEST — run this file directly to verify
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":

    agent = QLearningAgent()

    print("Q-table at start (all zeros):")
    print(agent.Q)
    print(f"\nShape: {agent.Q.shape}  ← 18 states × 3 actions")
    print(f"Epsilon: {agent.epsilon}  ← 100% exploration at start")

    # Simulate one learning update manually
    print("\n── Simulating one update ──────────────────────────")
    state      = 4     # some market situation
    action     = 1     # agent chose Buy
    reward     = 2.5   # portfolio grew by 2.5%
    next_state = 9     # new market situation after buying
    done       = False

    print(f"Before: Q[{state}, Buy] = {agent.Q[state, action]:.4f}")
    agent.update(state, action, reward, next_state, done)
    print(f"After:  Q[{state}, Buy] = {agent.Q[state, action]:.4f}")
    print(f"\nExplained:")
    print(f"  target = {reward} + {agent.gamma} × max(Q[{next_state}])")
    print(f"  target = {reward} + {agent.gamma} × {np.max(agent.Q[next_state]):.4f}")
    print(f"  target = {reward + agent.gamma * np.max(agent.Q[next_state]):.4f}")
    print(f"  new Q  = 0 + {agent.alpha} × ({reward + agent.gamma * np.max(agent.Q[next_state]):.4f} - 0)")
    print(f"  new Q  = {agent.Q[state, action]:.4f}")

    # Test epsilon decay
    print(f"\n── Epsilon decay over episodes ────────────────────")
    test_agent = QLearningAgent()
    for ep in [1, 100, 200, 300, 400, 500]:
        while test_agent.epsilon > test_agent.eps_min * 1.001:
            test_agent.decay_epsilon()
            break
        pass

    # Just show what epsilon looks like at key points
    eps_agent = QLearningAgent()
    checkpoints = {}
    for ep in range(1, 501):
        eps_agent.decay_epsilon()
        if ep in [1, 100, 200, 300, 400, 500]:
            checkpoints[ep] = eps_agent.epsilon

    for ep, eps in checkpoints.items():
        bar = "█" * int(eps * 30)
        print(f"  Episode {ep:3d}: ε = {eps:.3f}  {bar}")





