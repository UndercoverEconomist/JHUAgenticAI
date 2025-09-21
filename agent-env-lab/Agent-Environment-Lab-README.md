# Agent–Environment Lab (Vite + React, Plain CSS)

An interactive **gridworld** teaching tool that demonstrates how **environment properties** (observability, stochasticity, dynamics, horizon, action space, multi‑agent) and **resource limits** (real‑time planning budget) shape what counts as **rational** behavior. The agent uses an **anytime Monte‑Carlo** planner with additional **robustness heuristics** (loop‑avoidance, information‑seeking fallback, no‑progress detection) so it doesn’t get stuck when the environment gets hard.

---

## ✨ What is being demonstrated
- **Observability** → intuition for POMDPs (fog‑of‑war, sensor noise, revealing new cells)
- **Stochasticity** → noisy transition kernel via “slip” probability
- **Dynamics** → moving obstacles (non‑stationarity), need for replanning
- **Horizon** → episodic (achievement) vs. non‑episodic (maintenance) tasks
- **Action space** → (placeholder slider) talk about discrete vs. continuous control
- **Multi‑agent** → adversary as strategic uncertainty (random vs. greedy chaser)
- **Bounded/resource rationality** → time‑budgeted anytime planning

---

## 🚀 Quickstart

> Requires **Node 18+** (or 20+). No Tailwind / PostCSS used.

If you haven’t scaffolded the project yet, do:

```bash
npm create vite@latest agent-env-lab -- --template react
cd agent-env-lab
npm i
```

Replace/add the app files with those from this repository (notably `src/App.jsx` and `src/index.css`). Then run:

```bash
npm run dev
```

Open the URL Vite prints (usually http://localhost:5173).

> If you already have the project, just ensure your `src/App.jsx` and `src/index.css` match the latest version with robustness enabled.

---

## 🧭 Project structure

```
agent-env-lab/
├─ index.html
├─ package.json
└─ src/
   ├─ App.jsx       # simulation + UI
   ├─ main.jsx      # React entry point
   └─ index.css     # plain CSS styles (no Tailwind)
```

---

## 🖥️ UI overview

Left: **Control panel** (toggles & sliders) + **status** + teaching tips.  
Right: **Canvas** showing the gridworld.

### Legend
- **Blue circle** = Agent
- **Green circle** = Goal
- **Dark gray squares** = Obstacles (only drawn if “seen” under partial observability)
- **Red square** = Adversary (if enabled)
- **Light tiles** = Seen cells; **Darker tiles** = Unseen (fog‑of‑war)
- **Blue ring around agent** = Sensing radius (partial observability ON)

---

## 🎛️ Controls (toggles & sliders)

| Control | What it changes | Why it matters (concept) |
|---|---|---|
| **Partial (fog‑of‑war)** | If ON, the agent only “sees” cells within a radius; new cells are added to `seen`. | **Observability** axis; intuition for POMDPs and belief updates. |
| **Radius** | Size of the observation radius. | Larger radius reduces uncertainty (higher VoI). |
| **Sensor noise** | Chance to *miss* revealing a cell within the radius. | Noisy observations complicate state estimation. |
| **Slip probability** | With this probability, the chosen action is replaced by a random *different* move. | **Stochasticity** in the transition model \(T(s,a,s')\). |
| **Dynamic obstacles** | Obstacles drift randomly every few ticks. | **Dynamic/non‑stationary** environments require replanning. |
| **Episodic (end at goal)** | If ON, the episode ends at the goal (achievement). If OFF, it keeps going (maintenance). | **Task horizon**: achievement vs. maintenance. |
| **Granularity** | *Placeholder* slider (currently still 1‑cell steps). | Talk about **discrete vs. continuous** control; can be wired to micro‑steps. |
| **Enable adversary** | Adds a second agent (random or chaser). | **Multi‑agent/strategic uncertainty** preview. |
| **Adversary type** | *Random walker* or *Chaser* (greedy towards agent). | Adversarial dynamics vs. stochastic noise. |
| **Real‑time planning budget (ms)** | Per‑step time for the planner to think. | **Bounded/resource rationality**: more time ⇒ better decisions. |

### Buttons
- **Step** — Run one decision + environment step.
- **Play / Pause** — Start/stop a ~6 FPS loop so students can watch behavior evolve.
- **Reset** — Reset the environment with current settings.
- **Randomize** — Create a new random map (size varies within a range).

---

## 🧠 Agent policy (anytime Monte‑Carlo with robustness)

At each decision step, the agent uses an **anytime Monte‑Carlo lookahead**:
1. **Until the time budget expires**, loop through actions \\( \\{\\text{UP, RIGHT, DOWN, LEFT}\\} \\).  
2. For each action, **rollout** a short horizon (≈10–14 ticks):  
   - First move = the candidate action, subsequent moves = random policy (cheap futures).  
   - Score each rollout with a shaped return:  
     - **Progress**: \\(-0.02\\times\\text{ManhattanDistanceToGoal}\\) (encourage shorter paths)  
     - **Goal**: +5 if goal reached inside the rollout
3. **Average** scores per action and take the best (estimates shown in the status panel).  
4. Tiny \\(\\varepsilon\\)-randomization breaks ties to reduce cycling.

### Robustness upgrades (to avoid loops / getting stuck)
- **Loop‑avoidance memory:** Rollouts get a small **penalty** for visiting **recent agent positions** in the first few rollout steps (tabu‑like).
- **No‑progress detector:** Track the best Manhattan distance to goal; if it hasn’t improved for a while (tunable), consider the agent **stuck**.
- **Information‑seeking fallback:** When stuck (and partial obs ON), pick the move that **reveals the most unseen cells** in the observation radius (a VoI‑style heuristic).

All of this keeps the agent moving and exploring even when the world is partial, stochastic, dynamic, and adversarial.

---

## 🧱 Environment, rewards, & termination

**State elements (tracked in JS):**
- `agent` (x,y), `goal` (x,y)
- `obstacles` (Set of \"x,y\" keys)
- `adversary` (optional) (x,y)
- `seen` (Set of cells revealed so far)
- `tick` (time steps), `reward` (cumulative), `done` (terminal flag)
- **Robustness state:**  
  - `recent` (queue of recent agent positions)  
  - `bestDist` (best Manhattan distance to goal so far)  
  - `lastImprovedTick` (tick when `bestDist` last improved)

**Transition sequence per step:**
1) Apply **slip**: with prob `slipProb`, replace chosen action with a different random action.  
2) If next cell is blocked, the agent stays put.  
3) Update **seen** (if partial obs).  
4) If **dynamic**, obstacles drift one cell (while keeping agent/goal/adversary cells free).  
5) If **multi‑agent**, the adversary moves (random or greedy **chaser**).

**Rewards:**
- Step cost: **−0.05** (encourage efficiency)  
- Goal reached: **+10**  
- Adversary collision: **−5**

**Termination:**
- If **episodic** is ON: episode ends when the agent reaches the goal.  
- Otherwise: no terminal condition; it’s a maintenance task (watch cumulative reward trend).

---

## 🧪 Try this

1. **Bounded rationality**  
   Partial obs ON (radius 2), slip 0.2. Compare behavior at **10 ms vs 200 ms** planning budgets.  
   *Observation:* estimates stabilize and plans become less myopic with more time.

2. **Open environment stress test**  
   Partial obs ON, slip 0.3, **dynamic obstacles ON**, **adversary = chaser**.  
   *Observation:* agent must continuously replan; robustness avoids loops.

3. **Maintenance vs. achievement**  
   Toggle **episodic OFF**. Discuss how policies prioritize steady low cost and safety indefinitely.

4. **Information has value**  
   Partial obs ON with **sensor noise** ≈ 0.2. Increase radius and note fewer surprises, better path quality.

---

## ⚙️ Configuration & tuning knobs (in code)

Look near the top of `src/App.jsx` for these constants to adjust behavior:

```js
const LOOP_PENALTY = 0.10      // penalty for revisiting recent cells in rollouts
const LOOP_PENALTY_STEPS = 6   // apply above only in early rollout steps
const INFO_GAIN_WEIGHT = 0.02  // reward per newly revealed cell in rollout
const NO_PROGRESS_TICKS = 18   // if no distance improvement for this many ticks => stuck
const STUCK_MIN_HISTORY = 6    // need at least this many steps to judge stuck
```

---

## 🛠️ Troubleshooting

- **Blank page** — Check browser console. Common causes: missing `import './index.css'` in `src/main.jsx` or missing `<div id=\"root\">` in `index.html`.
- **Looks jittery / random** — That’s Monte‑Carlo variance. Raise the **planning budget**.
- **Still loops sometimes** — Increase `LOOP_PENALTY` or decrease `NO_PROGRESS_TICKS`. You can also widen the info‑seeking heuristic by increasing `INFO_GAIN_WEIGHT`.
- **Performance** — Reduce grid size or lower dynamic obstacle frequency (in the code, it moves every 3 ticks).

---

## 🧩 Possible extensions

- **Wire up true granularity**: make `granularity > 1` take sub‑steps (quasi‑continuous control).  
- **Belief heatmap**: show a probability grid for obstacles and update it with a Bayes step.  
- **Preset scenarios**: one‑click setups (e.g., *Fully Observable & Deterministic*, *POMDP + Stochastic + Dynamic*, *Open Environment*).  
- **MCTS (UCT) core**: replace simple rollouts with a small search tree while respecting the time budget.  
- **Metrics**: add regret, constraint violations, safety counters to the status panel.

---

