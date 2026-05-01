# Physics module — references and provenance

This module is a from-scratch Python implementation of a 4-ball Korean carom
billiards simulator. The free-flight dynamics, cue impact, and ball-cushion
collision models are not invented here; they are well-known results in the
billiards-physics literature, and parts of the implementation are direct
ports of [PoolTool](https://github.com/ekiefl/pooltool). This file lists the
sources for each component so the paper, the proposal, and any reviewer
can trace every formula back to its origin.

## At a glance

| Component | File(s) | Primary source | Implementation source |
|---|---|---|---|
| Cue tip → ball impact (level cue, instantaneous point) | `cue_impact.py` | Marlow 1995 | PoolTool, `instantaneous_point` |
| Free-flight dynamics (slip / roll / spin decay) | `dynamics.py` | Standard rigid-sphere result; reproduced in Marlow 1995 and Alciatore (Dr. Dave) | own |
| Ball ↔ cushion collision | `collisions.py` (`resolve_ball_cushion`) | Han 2005 | PoolTool, `han_2005` |
| Ball ↔ ball collision | `collisions.py` (`resolve_ball_ball`) | Standard equal-mass elastic-with-restitution exchange | own |
| Default coefficients (μ_slip, μ_roll, cushion_restitution, …) | `state.py` (`TableSpec`) | PoolTool defaults | — |
| Carom equipment dimensions (ball radius, mass, cue mass, rail height) | `state.py` (`TableSpec`) | UMB / world-carom-federation standards (mirrored in PoolTool's carom defaults) | — |
| 4-ball starting layout (head-string offsets) | `state.py` (`TableState.initial_4ball`) | 3-cushion spotting convention, extended to a 2nd red ball | own |

## Full references

### 1. PoolTool (implementation source)

- **Citation.** Kiefl, E. (2024). *Pooltool: A Python package for realistic
  billiards simulation.* Journal of Open Source Software, 9(101), 7301.
  [doi:10.21105/joss.07301](https://doi.org/10.21105/joss.07301)
- **Repo.** <https://github.com/ekiefl/pooltool> (Apache-2.0)
- **Theory write-up.** <https://ekiefl.github.io/2020/04/24/pooltool-theory/>
- **Used here for.**
  - The cue-impact closed-form (`pooltool.physics.resolve.stick_ball.instantaneous_point`)
    is reimplemented in `cue_impact.py` with the cue tilt fixed at θ = 0.
  - The Han 2005 ball-cushion model (`pooltool.physics.resolve.ball_cushion.han_2005`)
    is reimplemented in `collisions.py::resolve_ball_cushion`.
  - The default friction coefficients in `TableSpec` (`mu_slip = 0.20`,
    `mu_roll = 0.018`, `cushion_restitution = 0.85`) are PoolTool's
    cloth/cushion defaults.

### 2. Marlow 1995 (cue impact and free-flight ground truth)

- **Citation.** Marlow, W. C. (1995). *The Physics of Pocket Billiards.*
  MAST Publications. ISBN 978-0964537002.
- **Used here for.**
  - The Marlow energy-loss factor in `cue_impact.py`:
    `v_eff = 2·V0 / (1 + m/M + (5/2)·(a² + b²))`. Off-center hits reduce
    both linear speed and resulting spin through this single factor.
  - The closed-form slip-decay rate `du/dt = -(7/2)·μ_s·g·û` and the
    rolling-friction relation `dv/dt = -μ_r·g·v̂` used in `dynamics.py`.
    These are standard rigid-sphere-on-cloth results; Marlow's book gives
    a full derivation in pool/billiards units.

### 3. Han 2005 (ball-cushion collision)

- **Citation.** Han, I. (2005). *Dynamics in carom and three cushion
  billiards.* Journal of Mechanical Science and Technology, 19(4), 976–984.
  [doi:10.1007/BF02919180](https://doi.org/10.1007/BF02919180)
- **PDF mirror.** <https://drdavepoolinfo.com/physics_articles/Han_paper.pdf>
- **Used here for.** All ball-cushion collision math in
  `collisions.py::resolve_ball_cushion`. We follow Han's Eqs. (14), (17),
  (20)–(23) verbatim in the rail frame, including the contact-angle
  geometry `θ_a = arcsin(h/R − 1)` from cushion nose height `h`, the
  sliding-and-sticking vs. forward-sliding regime split via the threshold
  `μ·P_zE` on the friction impulse, and the impulse rotation back to the
  table frame. Defaults: `cushion_height = 37 mm` (carom rail nose),
  giving `θ_a ≈ 7.5°`.

### 4. Alciatore ("Dr. Dave") (community physics reference)

- **Resource site.** <https://billiards.colostate.edu/physics/>
  (David G. Alciatore, Professor Emeritus of Mechanical Engineering,
  Colorado State University). Hosts derivations and articles cross-
  referencing Marlow and Han.
- **Used here as.** A secondary cross-check / pedagogical reference for
  the impulse formulas. Dr. Dave is the curator that hosts the Han 2005
  PDF mirror linked above, and several of the spin-conventions used in
  `cue_impact.py` (top/back/draw/right/left english sign rules) follow
  the conventions standardized on his site.

## Equipment defaults (`TableSpec`)

```python
ball_radius   = 0.03275 m  # 65.5 mm diameter — UMB carom standard
ball_mass     = 0.210  kg
cue_mass      = 0.55   kg
cushion_height = 0.0370 m  # carom rail nose height; gives θ_a ≈ 7.5°
mu_slip       = 0.20       # PoolTool default
mu_roll       = 0.018      # PoolTool default
mu_spin       = 50.0       # vertical-axis spin decay (rad/s², linear)
cushion_restitution = 0.85 # PoolTool default; velocity-dependent caps off
cushion_friction    = 0.20 # μ_w for tangential / spin coupling
ball_restitution    = 0.94
```

The carom equipment values (ball radius/mass, cue mass, rail height) follow
the world-carom-federation (UMB) standard for international carom games and
match PoolTool's carom preset. They are not Korean-4-ball-specific — Korean
4-ball uses the same equipment as 3-cushion in practice.

## What is *not* modeled

The simulator deliberately omits several second-order effects so that one
shot resolves in seconds on a CPU:

- **Cue tilt / massé / jump.** `cue_impact.py` fixes θ_tilt = 0. A user-
  facing cue stick would tilt up to 90°; we ignore that.
- **Throw / cling on ball-ball collisions.** `resolve_ball_ball` is normal-
  only with restitution, no tangential impulse and no spin transfer.
  PoolTool exposes throw-enabled models; we don't.
- **Velocity-dependent cushion restitution.** `cushion_restitution` is a
  scalar; PoolTool optionally fits `e(v)` curves.
- **Cloth nap / table tilt / spotted balls drift / temperature.** All
  standard idealizations.

These omissions are noted because (a) they bound the realism of the
simulator and (b) they are uniform across every method we compare, so they
do not bias any of the reward-modeling experiments in this project.

## Sanity checks

Smoke artifacts that exercise the implementation against reference
behavior:

- `project/artifacts/dynamics_check.png` — slip→roll transition,
  rolling-friction stop time vs. analytic `t = v / (μ_r g)`.
- `project/artifacts/collisions_check.png` — ball-cushion incidence vs.
  rebound angles at varying side spin, and ball-ball normal-only exchange.
- `project/tests/test_inning_env.py`, `test_random_start_env.py`,
  `test_wrappers.py` — env-level regression tests.
