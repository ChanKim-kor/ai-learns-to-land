# AI Learns to Land

Chani Builds Project #003. The project name is a working title.

A learning project exploring rocket landing through Python, physics simulation,
interactive control, and reinforcement learning. The long-term goal is to train
an agent to land a rocket, starting with one-dimensional vertical flight and
later extending to two dimensions.

## Current Status

The one-dimensional physics model, flight playback, landing classification, and
manual engine controls are implemented. The manual flight view supports live
ON/OFF commands, fuel-dependent flame display, landing feedback, and restart.

The `learning/` package contains both introductory neural-network exercises
and a rocket-connected Q-learning experiment with experience replay, a target
network, periodic evaluation, model saving, and policy flight playback.

**Work in progress:** the agent does not land reliably yet. Training behavior
and failure cases are being investigated. This is an educational experiment,
not a finished landing controller or a realistic flight simulator.

Implemented:

- Constant-acceleration motion, free fall, and constant-thrust simulation.
- Finite propellant, changing total mass, and engine cutoff when fuel runs out.
- Step splitting at ignition, fuel depletion, estimated ground contact, and the
  simulation time limit.
- A reusable `step_fueled_flight` function that accepts a thrust command and
  advances to the requested endpoint or an earlier physical event.
- Unit tests for motion, input validation, fuel depletion, and event handling.
- Height and velocity convergence comparisons against analytical references
  during continuous burning.
- Flight playback with a rocket marker, engine flame, state display, and
  synchronized height, velocity, and propellant plots.
- Playback based on wall-clock time: display frames can be skipped when drawing
  falls behind, without changing the calculated flight history.
- Landing classification with a configurable contact-speed threshold.
- Manual ON/OFF engine control, live state display, automatic height-range
  expansion, flight outcome display, and restart after or during a flight.
- Learning exercises covering manual gradients, PyTorch autograd, SGD, linear
  layers, a shared hidden layer, and one Q-learning update.
- Rocket-connected episodes with ON/OFF engine actions, experience replay,
  batch Q-value updates, and a target network.
- Evaluation history, final and best-evaluation model checkpoints, reward
  plots, and saved-policy flight playback with engine-switch timing.

## Physics Model

- SI units are used throughout: meters, seconds, kilograms, and newtons.
- Height is positive above the ground. Upward velocity and acceleration are
  positive; downward velocity is negative.
- Vertical acceleration is `thrust / total_mass - g`.
- Propellant consumption is `thrust / (specific_impulse * g0)`, where
  `g0 = 9.80665 m/s^2` is the reference gravity used for specific impulse.
  The simulation gravity `g` is configured separately.
- Acceleration is held constant within each step using the mass at its start.
  Burning motion is therefore an approximation, not an exact variable-mass
  solution. The current continuous-burn comparison shows approximately
  first-order convergence as the time step is reduced.
- Ground-contact timing is calculated using that step's constant acceleration.
- Fuel depletion turns off thrust; flight continues until ground contact or the
  time limit. Ground contact takes priority if both termination conditions
  coincide.

The model currently covers vertical translation only. It does not include
atmospheric drag, attitude, horizontal motion, or landing-leg impact dynamics.

### Step Function Contract

`step_fueled_flight` receives the current height, velocity, and propellant mass,
rocket and environment parameters, a commanded thrust, and a positive `dt`.
It returns the updated state, the actual time advanced, and contact/fuel flags.

A step can end before `dt` when fuel runs out or ground contact occurs. The
caller is responsible for continuing any remaining time. The returned
`fuel_depleted` flag describes the resulting state: it is also true when the
step started with no fuel. It does not mean that depletion occurred during
that particular call.

## Landing Outcomes

At ground contact, a flight is classified as `landed` when
`abs(vertical_velocity) <= max_landing_speed`, and as `crashed` otherwise.
The default maximum landing speed is `2.0 m/s`, including the boundary value.
This is a configurable simulation criterion.

Before contact, `classify_landing` returns `in_flight`. The manual view displays
`time_limit` when the run ends without contact at the configured time limit.
Fuel depletion alone does not end a flight.

The physical termination reason and landing outcome are separate: a run can
terminate with `ground_contact` and have a landing outcome of `crashed`.

## Run Locally

Development environment: Windows PowerShell, Python 3.14.7, and Visual Studio
Code. Run the following commands from the repository root.

Create a project-local environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If `.venv` already exists, reuse it instead of creating another environment.
In VS Code, select `.venv\Scripts\python.exe` as the Python interpreter.
`requirements.txt` records Matplotlib and the PyTorch CPU build. The first
installation command selects PyTorch's official CPU package index so the pinned
`+cpu` build is available. NumPy is currently installed through Matplotlib and is
also used by the visualization code. The learning exercises run on CPU.

Run a text-based flight example:

```powershell
.\.venv\Scripts\python.exe run_simulation.py
```

Open the flight playback window:

```powershell
.\.venv\Scripts\python.exe visualize_simulation.py
```

The playback script calculates flight history before displaying it.
To control a flight interactively instead, run:

```powershell
.\.venv\Scripts\python.exe manual_control.py
```

Use **Engine ON**, **Engine OFF**, and **RESTART**. Flight starts with the engine
off. Button commands are applied after advancing the existing command up to the
input-processing time. The display distinguishes the engine command from the
flame, which is hidden when fuel is unavailable or the flight has finished.
The timer stops at ground contact or the time limit; restart resets the flight,
engine command, view range, and wall-clock origin.

Initial conditions and rocket settings are currently configured in each script.

Run the unit tests and the convergence experiment:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -v
.\.venv\Scripts\python.exe run_convergence.py
```

The convergence experiment reports height and velocity errors for progressively
smaller time steps. It is separate from the unit test suite.

## Rocket Learning Experiment

Run the following modules from the repository root, after installing the
dependencies above. First, train the current experimental agent:

```powershell
.\.venv\Scripts\python.exe -m learning.train_rocket_transition
```

The script currently runs 50 training episodes, with exploration-free evaluations
before training, every five episodes, and after training. It sets both the Python
and PyTorch random seeds to 0. This is a single-seed experiment, not evidence of
reliable performance across runs or starting conditions.

After completion, it writes these files under `runs/`:

| File | Contents |
| --- | --- |
| `evaluation_history.csv` | Periodic evaluation results, including final vertical velocity, outcome, termination reason, and reward |
| `final_model.pt` | Network weights at the end of training |
| `best_model.pt` | Weights with the highest recorded evaluation reward, including the pre-training evaluation |

Re-running training overwrites these files. Copy results elsewhere before
starting another experiment if you want to keep them. The `runs/` directory is
excluded from Git; a fresh clone does not contain these outputs.

After training, plot the saved evaluation history:

```powershell
.\.venv\Scripts\python.exe -m learning.plot_training
```

Or evaluate and replay the policy in `runs/best_model.pt`:

```powershell
.\.venv\Scripts\python.exe -m learning.evaluate_model
```

Playback prints the flight outcome, final vertical velocity, and engine ON/OFF times.
The plot and playback commands need their respective files from a completed
training run; running them before training results in a missing-file error.
The best recorded reward does not mean the model can land successfully or
reliably. Current experiments still include crashes and time-limit outcomes.

## Learning Exercises

Run these modules from the repository root with the project environment:

```powershell
.\.venv\Scripts\python.exe -m learning.train_q_update
```

Replace the module name to follow the learning sequence:

| Module | Focus |
| --- | --- |
| `learning.train_single_neuron` | Manual gradients and repeated parameter updates |
| `learning.train_single_neuron_torch` | Tensor operations, autograd, gradient clearing, and SGD |
| `learning.train_linear_layer` | Two independent action-value outputs; update the selected action |
| `learning.train_small_network` | A shared hidden layer and its effect on both outputs |
| `learning.train_q_update` | Build a target from reward and next-state values, then update once |

The Q-update exercise uses a manually specified transition and a small network.
It is a calculation exercise, not a complete DQN agent. For a nonterminal
transition, its target is `reward + gamma * max(next_q_values)`; for a terminal
transition, the target is `reward`. The target is computed without gradient
tracking before updating the selected current-state Q value.

## Project Files

| File | Purpose |
| --- | --- |
| `physics.py` | Motion equations, propellant calculations, the reusable flight step, and simulation runners |
| `landing.py` | Configurable landing outcome classification |
| `test_physics.py`, `test_landing.py` | Unit tests for physics, events, and landing classification |
| `run_simulation.py` | Text-based fueled-flight example |
| `run_convergence.py` | Comparison with analytical continuous-burn references |
| `visualize_simulation.py` | Flight playback and synchronized plots |
| `manual_control.py` | Live engine controls, state/flame display, flight outcomes, and restart |
| `learning/` | Python package containing introductory exercises and the rocket-learning experiment |
| `learning/train_rocket_transition.py` | Rocket-connected training, replay, periodic evaluation, and checkpoint saving |
| `learning/plot_training.py` | Reward plot from the saved evaluation history |
| `learning/evaluate_model.py` | Saved-policy evaluation, flight playback, and engine-switch timing |
| `requirements.txt` | Python dependency installation |

## Next Steps

1. **Investigate failed landings.** Inspect engine actions, flight trajectories,
   terminal rewards, and time-limit behavior. Test explanations for failures
   rather than treating one improved flight as a solved task.
2. **Make experiments easier to compare.** Record settings alongside results,
   preserve separate runs, and evaluate multiple seeds and starting conditions.
   Compare landing outcomes and contact speeds, not reward alone.
3. **Improve the learning interface and its checks.** As the experiment develops,
   clarify episode-ending rules, separate reusable environment responsibilities,
   and add focused checks for training and evaluation behavior.
4. **Share progress and limitations.** Keep the setup instructions and development
   notes aligned with the implementation. Improve the visual presentation when
   it helps explain the behavior.
5. **Extend the physics and control later.** Explore drag and other refinements,
   then horizontal motion and attitude in two dimensions. Validate each extension
   before assessing its effect on control and learning.

## Development Approach

This is a learning project developed with substantial AI assistance. The AI
assistant wrote most of the code. The author participates by asking questions,
working through explanations, reviewing behavior, suggesting changes, and
deciding what to investigate next. Understanding the physics, Python tools,
and learning process is an ongoing part of the project.

Code, comments, and project documentation are written in English. Google Drive
holds project plans and decision records; GitHub holds code, tests, and the
repository documentation. The roadmap will evolve as implementation and
experiments reveal what is needed.
