# Q-Learning Stock Trading Agent (AAPL)

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-tabular%20RL-013243?logo=numpy&logoColor=white)
![Data](https://img.shields.io/badge/data-Yahoo%20Finance-6001D2)

A tabular **Q-learning** agent that learns when to buy, sell or hold Apple (AAPL) stock from two technical indicators, RSI and a moving-average ratio. The agent is trained on 2015–2021 prices and tested on unseen 2021–2022 prices against a simple buy-and-hold investor.

---

## Results

Test period: 2 June 2021 – 30 December 2022 (400 trading days), starting capital 10,000.

| Strategy | Final value | Return |
|---|:-:|:-:|
| Buy and hold | 10,475 | **+4.8%** |
| Q-learning agent (greedy policy) | 9,041 | −9.6% |

The agent executed 62 trades. It chose Hold on 37% of days, Buy on 29% and Sell on 34%, but many Buy/Sell choices were no-ops because it was already in that position.

### What the agent learned

Reading the greedy action from the Q-table gives a clear pattern:

| Situation | Learned action |
|---|---|
| Price **above** its 10-day average (MA ratio > 1.02), RSI ≥ 40 | **Buy** / keep holding |
| Price **near or below** its 10-day average while holding | **Sell** |
| RSI < 40 and price below average, no shares | Hold (stay out) |

In other words, the agent learned a **short-term momentum** strategy: buy after the price has risen, sell after it dips. That worked in the long 2015–2021 bull run it was trained on, but the test period was choppy and ended in the 2022 sell-off. A momentum rule in a sideways market keeps buying after rallies and selling after dips, which is exactly the whipsaw pattern visible in the test portfolio chart.

### Why the agent struggles

1. **The reward cannot tell actions apart.** In `yahoo_env.py`, the reward for day *t* is the change in portfolio value from yesterday's close to today's close, using the position held *before* today's action. That position is already part of the state (the "holding" bit), so every action in a given state receives the same immediate reward. Actions only differ through the next state, which is why many Q-table rows have three nearly identical values (e.g. 16.9 / 16.5 / 16.7).
2. **The state is very coarse.** Eighteen states built from two indicators and a holding flag leave little information for predicting next-day returns, which are close to random at this horizon.
3. **Distribution shift.** Training covers a strong bull market (AAPL rose about 4.8× over the training period), while the test period includes a bear market.
4. **It underperforms even in-sample.** Over the last 50 training episodes the agent averaged about 13,700, while buy-and-hold on the same training data would reach about 48,000. (Training episodes still use ε ≈ 0.08 exploration, so this is slightly pessimistic, but the gap is large.)

---

## MDP formulation

**State (18 discrete states)**: `state = rsi_bin × 6 + ma_bin × 2 + holding`

| Component | Bins |
|---|---|
| RSI (14-day) | 0: < 40 (oversold) · 1: 40–60 (neutral) · 2: > 60 (overbought) |
| MA ratio (close ÷ 10-day moving average) | 0: < 0.98 (below) · 1: 0.98–1.02 (near) · 2: > 1.02 (above) |
| Holding | 0: no shares · 1: holding shares |

**Actions**: 0 = Hold, 1 = Buy (invest all cash), 2 = Sell (sell all shares)

**Reward**: daily change in portfolio value as a percentage of starting capital, minus a 0.1% transaction cost on every executed trade

**Update rule**:

```
Q(s,a) ← Q(s,a) + α [ r + γ · max_a' Q(s',a') − Q(s,a) ]
```

| Hyperparameter | Value |
|---|---|
| Learning rate α | 0.1 |
| Discount γ | 0.99 |
| ε start / min / decay per episode | 1.0 / 0.01 / 0.995 |
| Episodes | 500 (each episode = one pass over all 1,600 training days) |
| Starting capital | 10,000 |

**Data split**: chronological 80/20 (never random, to avoid look-ahead leakage). Training: 23 Jan 2015 – 1 Jun 2021 (1,600 days). Test: 2 Jun 2021 – 30 Dec 2022 (400 days).

---

## Repository structure

```
.
├── yahoo_data.py       # Downloads AAPL, computes RSI and MA ratio, saves the train/test split
├── yahoo_env.py        # Trading environment: state discretisation, actions, reward
├── yahoo_qagent.py     # Q-learning agent: epsilon-greedy policy and Bellman update
├── train.py            # Trains for 500 episodes, saves q_table.npy and training_log.csv
├── evaluate.py         # Tests on unseen data vs buy-and-hold, saves results.png
├── data_full.csv       # Full processed dataset (2,000 trading days)
├── data_train.csv      # Training split
├── data_test.csv       # Test split
├── q_table.npy         # Trained Q-table behind the results above
├── training_log.csv    # Reward and portfolio value for every training episode
├── assets/results.png  # Figure used in this README
└── requirements.txt
```

The processed CSVs, Q-table and training log are committed on purpose. Yahoo Finance re-adjusts historical prices over time and training has no fixed random seed, so these files are the only way to reproduce the exact numbers above.

---

## Getting started

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Reproduce the reported results** using the committed Q-table:

```bash
python evaluate.py
```

**Run the full pipeline from scratch** (results will differ slightly between runs):

```bash
python yahoo_data.py     # re-download data (optional; overwrites the CSVs)
python train.py          # ~500 episodes
python evaluate.py
```

`yahoo_env.py` and `yahoo_qagent.py` can also be run directly as quick self-tests.

---

## Future work

- **Fix the reward timing** so each action is rewarded by what happens *after* it: execute at today's close and value the portfolio at tomorrow's close.
- Set a random seed in `train.py` for reproducible training.
- Add a validation period for choosing hyperparameters and indicator thresholds, rather than going straight from training to test.
- Compare against simple rule-based baselines (e.g. a pure moving-average crossover) to check whether the agent learned anything beyond them.
- Train across several tickers or market regimes to reduce dependence on one bull market.
- Report risk-adjusted metrics such as Sharpe ratio and maximum drawdown alongside raw return.

**This project is an educational exercise and is not investment advice.**

---


