
import numpy as np
import pandas as pd


class StockEnv:
    """
    State:   discretised (rsi_bin, ma_bin, holding) → 18 possible states
    Actions: 0 = Hold, 1 = Buy, 2 = Sell
    Reward:  change in portfolio value minus transaction cost
    """

    N_ACTIONS = 3
    N_STATES  = 18       # 3 rsi bins × 3 ma bins × 2 holding states

    def __init__(self, df, initial_cash=10_000):
        self.df           = df.reset_index(drop=True)
        self.initial_cash = initial_cash
        self.reset()

    # ─────────────────────────────────────────────────────
    #  DISCRETISATION
    #  RSI and MA ratio are continuous numbers.
    #  Q-Learning needs integer state indices.
    #  We bucket them into 3 bins each.
    # ─────────────────────────────────────────────────────
    def _discretise(self, rsi, ma_ratio, holding):

        # RSI → 3 bins
        # 0 = oversold  (rsi < 40)  → stock falling, possible buy
        # 1 = neutral   (40–60)
        # 2 = overbought(rsi > 60)  → stock rising, possible sell
        if rsi < 40:
            rsi_bin = 0
        elif rsi > 60:
            rsi_bin = 2
        else:
            rsi_bin = 1

        # MA ratio → 3 bins
        # 0 = below average (ma_ratio < 0.98) → price looks cheap
        # 1 = near average  (0.98–1.02)
        # 2 = above average (ma_ratio > 1.02) → price looks expensive
        if ma_ratio < 0.98:
            ma_bin = 0
        elif ma_ratio > 1.02:
            ma_bin = 2
        else:
            ma_bin = 1

        # Holding → 2 bins
        # 0 = not holding shares
        # 1 = currently holding shares
        h_bin = 1 if holding else 0

        # Combine into one integer: 3 × 3 × 2 = 18 states
        # Formula: rsi_bin × 6 + ma_bin × 2 + h_bin
        # Example: rsi=0, ma=1, holding=1 → 0×6 + 1×2 + 1 = 3
        return rsi_bin * 6 + ma_bin * 2 + h_bin

    # ─────────────────────────────────────────────────────
    #  RESET  — called at the start of every episode
    # ─────────────────────────────────────────────────────
    def reset(self):
        self.current_step = 0
        self.cash         = self.initial_cash
        self.shares       = 0
        self.prev_worth   = self.initial_cash
        return self._get_state()

    # ─────────────────────────────────────────────────────
    #  GET STATE  — read today's indicators → integer
    # ─────────────────────────────────────────────────────
    def _get_state(self):
        row = self.df.iloc[self.current_step]
        return self._discretise(
            rsi      = float(row["rsi"]),
            ma_ratio = float(row["ma_ratio"]),
            holding  = self.shares > 0
        )

    # ─────────────────────────────────────────────────────
    #  STEP  — agent sends action, environment responds
    #  Returns: (next_state, reward, done)
    # ─────────────────────────────────────────────────────
    def step(self, action):
        row   = self.df.iloc[self.current_step]
        price = float(row["Close"])
        cost  = 0.0

        # ── Execute the action ──────────────────────────
        if action == 1 and self.shares == 0:      # Buy
            self.shares  = self.cash // price     # buy as many as we can
            cost         = self.shares * price * 0.001   # 0.1% broker fee
            self.cash   -= (self.shares * price) + cost

        elif action == 2 and self.shares > 0:     # Sell
            cost         = self.shares * price * 0.001   # 0.1% broker fee
            self.cash   += (self.shares * price) - cost
            self.shares  = 0

        # action == 0 → Hold, do nothing

        # ── Advance time by one trading day ────────────
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        # ── Calculate reward ────────────────────────────
        # Current portfolio value = cash + value of shares held
        curr_worth = self.cash + self.shares * price

        # Reward = % change in portfolio, scaled to small number
        # Minus transaction cost penalty to discourage overtrading
        reward  = (curr_worth - self.prev_worth) / self.initial_cash * 100
        reward -= cost / self.initial_cash * 100

        self.prev_worth = curr_worth

        # ── Get next state ──────────────────────────────
        if not done:
            next_state = self._get_state()
        else:
            next_state = 0   # episode over, state doesn't matter

        return next_state, reward, done

    # ─────────────────────────────────────────────────────
    #  PROPERTIES  — useful info to read after each step
    # ─────────────────────────────────────────────────────
    @property
    def portfolio_value(self):
        price = float(self.df.iloc[self.current_step]["Close"])
        return self.cash + self.shares * price

    @property
    def profit_pct(self):
        return (self.portfolio_value - self.initial_cash) / self.initial_cash * 100


# ─────────────────────────────────────────────────────────
#  MANUAL TEST  — run this file directly to verify it works
#  Simulates 5 trading days with fixed actions
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("Loading training data...")
    train_df = pd.read_csv("data_train.csv")
    print(f"Loaded {len(train_df)} rows\n")

    env   = StockEnv(train_df)
    state = env.reset()

    print(f"Initial state index : {state}")
    print(f"Starting cash       : £{env.cash:,.2f}")
    print(f"Starting portfolio  : £{env.portfolio_value:,.2f}")
    print("-" * 55)

    # Manually test 5 steps with fixed actions
    test_actions = [1, 0, 0, 2, 0]   # Buy, Hold, Hold, Sell, Hold
    action_names = {0: "Hold", 1: "Buy ", 2: "Sell"}

    for i, action in enumerate(test_actions):
        next_state, reward, done = env.step(action)

        print(f"Day {i+1} | Action: {action_names[action]} | "
              f"State: {state:2d} → {next_state:2d} | "
              f"Reward: {reward:+.4f} | "
              f"Portfolio: £{env.portfolio_value:,.2f}")

        state = next_state

    print("-" * 55)
    print(f"Final portfolio value : £{env.portfolio_value:,.2f}")
    print(f"Profit / Loss         : {env.profit_pct:+.2f}%")
