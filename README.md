# AI Learns to Land

Chani Builds Project #003. The project name is a working title.

A learning project exploring rocket landing through Python, physics simulation,
interactive control, and reinforcement learning. The long-term goal is to train
an agent to land a rocket, starting with one-dimensional vertical flight and
later extending to two dimensions.

## Current Status

The first one-dimensional physics model and flight playback are implemented.
The current example uses a scheduled ignition time. Manual controls and AI
training have not been implemented yet.

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

## Run Locally

Development environment: Windows PowerShell, Python 3.14.7, and Visual Studio
Code. Run the following commands from the repository root.

Create a project-local environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If `.venv` already exists, reuse it instead of creating another environment.
In VS Code, select `.venv\Scripts\python.exe` as the Python interpreter.
Matplotlib is listed in `requirements.txt`; NumPy is currently installed as one
of its dependencies and is also used by the visualization code.

Run a text-based flight example:

```powershell
.\.venv\Scripts\python.exe run_simulation.py
```

Open the flight playback window:

```powershell
.\.venv\Scripts\python.exe visualize_simulation.py
```

The current example settings are in each script. The visualization calculates
flight history before playback; it does not yet accept live engine commands.

Run the unit tests and the convergence experiment:

```powershell
.\.venv\Scripts\python.exe -m unittest -v test_physics
.\.venv\Scripts\python.exe run_convergence.py
```

The convergence experiment reports height and velocity errors for progressively
smaller time steps. It is separate from the unit test suite.

## Project Files

| File | Purpose |
| --- | --- |
| `physics.py` | Motion equations, propellant calculations, the reusable flight step, and simulation runners |
| `test_physics.py` | Unit tests for physics and event handling |
| `run_simulation.py` | Text-based fueled-flight example |
| `run_convergence.py` | Comparison with analytical continuous-burn references |
| `visualize_simulation.py` | Flight playback and synchronized plots |
| `requirements.txt` | Python dependency installation |

## Next Steps

1. **Define landing outcomes and control rules.** Start with engine ON/OFF
   commands. The provisional successful-landing threshold is ground contact at
   `abs(vertical_velocity) <= 2.0 m/s`. This is a configurable simulation
   criterion; landing classification is not implemented yet. Keep fuel and
   time limits explicit.
2. **Build manual flight control.** Connect keyboard or button input to the
   shared physics step. Advance the simulation during interaction and display
   the actual engine state instead of relying on a scheduled ignition time.
3. **Create the reinforcement learning environment.** Define observations,
   actions, rewards, and episode termination. Reuse the same physics for manual
   and AI control, with learning independent of screen rendering.
4. **Train and evaluate an initial agent.** Measure landing success, contact
   speed, and fuel use across multiple starting conditions. A rule-based
   controller may be added as a comparison if useful; it is not a prerequisite.
5. **Improve presentation and prepare a public release.** After AI control is
   working, refine rocket graphics, flames, the landing scene, and the display.
   Document reproducible results and limitations.
6. **Extend to two dimensions.** Add horizontal motion and attitude dynamics,
   validate the expanded physics, and then extend control and learning.

## Development Approach

The project author implements the physics and simulation code while learning
Python and software development. AI assists with design discussions, exercises,
hints, code review, verification, and documentation.

Code, comments, and project documentation are written in English. Google Drive
holds project plans and decision records; GitHub holds code, tests, and the
repository documentation. The roadmap will evolve as implementation and
experiments reveal what is needed.
