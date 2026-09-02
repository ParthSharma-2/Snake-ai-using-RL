# 🐍 Teach AI To Play Snake! — Reinforcement Learning with PyTorch and Pygame

An AI agent that learns to play Snake from scratch using **Deep Q-Learning (DQN)**, built with **Pygame** (game environment) and **PyTorch** (neural network + training). The agent starts out with zero knowledge of the game and, over hundreds of episodes, learns to chase food and avoid crashing into walls or itself.

---

## 📂 Project Structure

```
.
├── game.py               # SnakeGameAI — the RL environment (agent-controlled)
├── snake_game_human.py   # Standalone playable Snake game (arrow keys, no AI)
├── agent.py               # Agent class + training loop (the entry point)
├── model.py                # Linear_QNet (PyTorch model) + QTrainer
├── helper.py                 # Live matplotlib plot of training scores
├── arial.ttf                   # Font used to render the score in-game
├── LICENSE                       # MIT License
└── README.md
```

---

## 🎮 How It Works

### The environment (`game.py`)
`SnakeGameAI` runs the actual Snake game logic. Instead of taking keyboard input, `play_step(action)` takes an action from the agent each frame, moves the snake, checks for collisions, updates the score/food, and returns `(reward, game_over, score)`.

### State (11 values)
`agent.get_state()` builds an 11-value boolean state vector from the snake's head position and current direction:
- Danger straight ahead / danger to the right / danger to the left (relative to current heading)
- Current direction: left, right, up, down
- Food position relative to the head: left, right, up, down

### Action (3 values)
Actions are relative to the snake's current heading, not absolute arrow keys:

| Action | Meaning |
|---|---|
| `[1, 0, 0]` | Continue straight |
| `[0, 1, 0]` | Turn right |
| `[0, 0, 1]` | Turn left |

### Reward
| Event | Reward |
|---|---|
| Eats food | `+10` |
| Collides with wall or itself, or stalls (frame limit hit) | `-10` |
| Any other step | `0` |

### The model (`model.py`)
`Linear_QNet` is a small feed-forward network:

```
Input (11) → Linear → ReLU → Linear → Output (3 Q-values)
```

`QTrainer` implements the Bellman update:

```
Q_new = reward                                    (if game over)
Q_new = reward + gamma * max(Q(next_state))       (otherwise)
```

It computes predicted Q-values, updates the target for the action actually taken, and backpropagates the MSE loss between predicted and target Q-values via the Adam optimizer.

### The agent (`agent.py`)
`Agent` ties it all together:
- **Epsilon-greedy exploration** — `epsilon = 80 - n_games`, so the agent explores randomly early on and exploits its learned policy more as training progresses
- **Short-term memory** — trains immediately on every single step taken
- **Long-term memory (experience replay)** — stores up to 100,000 transitions in a `deque`; after each episode, samples a batch of 1,000 and trains on it
- Saves the model to `./model/model.pth` whenever a new high score (record) is reached
- Prints each game's score/record and updates a live plot via `helper.py`

---

## ⚙️ Installation

**Requirements:** Python 3.8+

```bash
git clone <your-repo-url>
cd <repo-folder>

# (optional) virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install torch pygame numpy matplotlib ipython
```

---

## 🚀 Usage

### Train the AI
```bash
python agent.py
```
This opens a Pygame window where the snake plays (and initially fails) in real time, alongside a live matplotlib plot tracking score and rolling average score per game. The best model is auto-saved to `model/model.pth` whenever a new record is set. Stop anytime with `Ctrl+C`.

### Play it yourself
```bash
python snake_game_human.py
```
A standalone, keyboard-controlled version of the same game (arrow keys) — useful for sanity-checking the game logic independent of the AI.

---

## 🛠️ Key Hyperparameters (`agent.py`)

| Parameter | Value | Description |
|---|---|---|
| `MAX_MEMORY` | 100,000 | Replay buffer capacity |
| `BATCH_SIZE` | 1,000 | Sample size for long-term training |
| `LR` | 0.001 | Learning rate (Adam) |
| `gamma` | 0.9 | Discount factor for future rewards |
| `epsilon` | `80 - n_games` | Decaying exploration rate |
| Hidden layer size | 256 | `Linear_QNet(11, 256, 3)` |

Game constants in `game.py`: `BLOCK_SIZE = 20`, board size `640x480`, game speed `SPEED = 40`.

---

## 📊 What to Expect

Scores are noisy early on (mostly random wall/self collisions) and generally trend upward as the agent accumulates experience — the live plot (score + running mean) makes this trend easy to see across a training session. Exact results depend on how many games you let it play.

---

## 🔮 Possible Extensions

- [ ] Convolutional network trained on raw pixel frames instead of the hand-crafted 11-value state
- [ ] Double DQN / Dueling DQN
- [ ] Prioritized experience replay
- [ ] Resume training from a saved checkpoint instead of starting fresh
- [ ] Configurable board size / speed via CLI args

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.