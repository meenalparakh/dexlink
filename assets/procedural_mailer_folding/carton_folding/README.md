# Procedural finite-thickness folding carton

A parameterized URDF generator, continuous joint-space folding motion, headless
renderer, native `urdfpy` viewer entry point, and sampled collision validator.
The layout is inspired by the supplied four-wall tuck-carton photograph; it is
not a traced die line or a measured reconstruction of that photograph.

**Build status:** The procedural generator, independent reference FK, geometry
checker, and software renderer were executed. `urdfpy` is not installed in the
build sandbox, and network/DNS restrictions prevented installation. Consequently,
**the supplied previews and reports use `--backend reference`, not `urdfpy`**.
The real `urdfpy` path is provided, uses its documented API, and still requires
execution in an environment where that dependency can be installed. There is no
fake `urdfpy` module and no silent backend substitution.

## 1. Contents

| File | Purpose |
|---|---|
| `build_carton.py` | Procedural generation from dimensions or JSON |
| `carton.urdf`, `carton.json` | Generated default model and resolved dimensions |
| `motion.py` | Continuous folding schedule and analytic insertion kinematics |
| `fold_carton.py` | GIF/MP4/still/section rendering, trajectory export, native viewer |
| `kinematics.py` | Real `urdfpy` FK backend and separately named reference FK |
| `validate_carton.py` | Tree, inertia, limits, velocities, and sampled box intersections |
| `generate_family.py` | Seeded dimension randomization with candidate validation |
| `render_family.py` | Gallery of multiple generated boxes |
| `test_carton.py` | Regression tests, including an optional real-urdfpy cross-check |
| `output/` | Default previews, trajectory, test results, and validation reports |
| `examples/` | Eight deliberately varied presets and twenty seeded random boxes |
| `configs/` | Small editable JSON configuration examples |

No external mesh assets are needed: the URDF uses box primitives for both visual
and collision geometry. All links have mass, center of mass, and inertia. The
compound socket-ear inertia includes parallel-axis terms.

## 2. Run with urdfpy

Use a separate Python 3.10 or 3.11 environment for the compatibility-oriented
installation below. These install commands were not executable in the build
sandbox; they are not presented as an environment that was locally verified.

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-urdfpy.txt
python -m pip install --no-deps urdfpy==0.0.22

# Generate the default carton. All lengths are in metres.
python build_carton.py --out models/default

# Uses actual urdfpy.URDF.load and robot.link_fk for all link transforms.
# Headless output uses a software renderer, so it needs no OpenGL display.
python fold_carton.py --urdf models/default/carton.urdf \
  --backend urdfpy --output-dir previews/default --fps 12

# Native urdfpy/pyrender interactive window; requires a desktop OpenGL context.
python fold_carton.py --urdf models/default/carton.urdf \
  --backend urdfpy --native-viewer

# Validate the same geometry and poses through urdfpy.
python validate_carton.py --urdf models/default/carton.urdf \
  --backend urdfpy --samples 5001 --report previews/default/validation.json
```

The upstream `urdfpy==0.0.22` package declares `networkx==2.2`. This setup
intentionally installs NetworkX 2.8.8 instead and uses `--no-deps` for urdfpy to
avoid reverting it. `pip check` will therefore report that declared-version
mismatch. A local NumPy proxy in `load_urdfpy` supplies legacy scalar aliases to
that module without modifying NumPy globally. The included optional unit test
compares actual urdfpy transforms against the independent reference backend.

`urdfpy.animate` treats samples as uniformly timed and plays the trajectory
forward and backward. The wrapper supplies `loop_time=2*motion.total`, so the
forward assembly takes the intended 24.5 seconds. The exported MP4/GIF plays
only the forward assembly and then loops.

### Explicit offline / dependency-light alternative

```bash
python -m pip install -r requirements.txt
python fold_carton.py --backend reference --output-dir previews/offline --fps 12
python validate_carton.py --backend reference --samples 5001
python -m unittest -v test_carton
```

`--snapshots-only` avoids full animation rendering. All preview footers and
`render_provenance.json` identify the FK backend. MP4 needs imageio's FFmpeg
backend; GIF/stills do not. GIF/MP4 rendering may take a few minutes on a CPU.

## 3. Procedural dimensions

```bash
# Different aspect ratio and board thickness; flap/socket sizes are automatic.
python build_carton.py --width 0.12 --depth 0.08 --height 0.18 \
  --thickness 0.0012 --out models/custom

# Same idea from JSON; explicit CLI values override JSON values.
python build_carton.py --config configs/custom.json --out models/from_json

# Explicit insertion geometry overrides, also checked for feasibility.
python build_carton.py --width 0.10 --depth 0.075 --height 0.16 \
  --thickness 0.001 --face-clearance 0.0004 --lateral-clearance 0.001 \
  --tab-width 0.060 --tab-length 0.015 --out models/overrides

python fold_carton.py --urdf models/custom/carton.urdf \
  --backend urdfpy --output-dir previews/custom
```

`width`, `depth`, `height`, `thickness`, `face_clearance`, `lateral_clearance`,
`tab_width`, `tab_length`, `keeper_length`, `ear_width`, `glue_length`, and
`density` are exposed as dataclass fields, JSON keys, and hyphenated CLI flags.
All geometric lengths use metres; density uses kg/m^3; joint angles use radians.

Omitted or JSON-null detail dimensions are automatically recomputed. This is not
just mesh scaling. Automatic sizing adjusts dust-flap notches and lengths,
receiver ears and keepers, lid span, seam offset, tab dimensions, and the
insertion approach angle. Explicit overrides are not silently shortened: an
infeasible override raises a useful error.

**Dimension convention:** W and D are opposing wall-midplane separations. H is
main-wall fold-line height. They are nominal carton dimensions, not exact outer
dimensions or a guaranteed unobstructed cavity. Lifted lids, finite thickness,
and the lap seam change the exterior envelope. The default nominal dimensions
are 160 x 130 x 280 mm; its assembled collision-geometry envelope is approximately
164.6 x 132.0 x 292.4 mm in W/D/H order. Each validation report records the actual
assembled axis-aligned envelope, so generated boxes can be checked against a
size constraint.

Automatic tab length also respects insertion reachability. Large changes in
aspect ratio can require a shorter tab than simple proportional scaling would
suggest. Arbitrary thin/tall/wide geometries are not all feasible; run validation
after changing parameters. The automatic rules are homogeneous: scaling W/D/H
and thickness together uniformly scales the geometry without changing its
joint angles.

### Python API

```python
from pathlib import Path
from build_carton import Parameters, generate
from motion import Motion
from kinematics import load_urdfpy

params = Parameters(width=0.12, depth=0.08, height=0.18, thickness=0.0012)
urdf_path, metadata_path = generate(params, Path("models/api_box"))
motion = Motion(metadata_path)
robot = load_urdfpy(urdf_path)  # Actual upstream urdfpy package

q, stage_name, stage_index = motion.evaluate(20.0)
poses = robot.link_fk(cfg=q)
# q is {joint_name: angle_radians}; use it as simulator position targets.
# For continuous playback, evaluate at the simulator's own timestamps.
```

### Reproducible families

```bash
python generate_family.py --count 20 --seed 7 --out family/new_run \
  --width-range 0.05 0.16 --depth-range 0.04 0.12 \
  --height-range 0.08 0.24 --thickness-range 0.0006 0.0016 \
  --backend urdfpy --samples 1001

python fold_carton.py --urdf family/new_run/box_0000/carton.urdf \
  --backend urdfpy --output-dir previews/box_0000

python render_family.py examples/small/carton.urdf examples/slim/carton.urdf \
  examples/wide/carton.urdf examples/deep/carton.urdf \
  examples/large/carton.urdf examples/shallow/carton.urdf \
  --backend reference --out previews/gallery.png
```

Each accepted family member gets its own URDF, resolved metadata, input
configuration, and validation report. The family manifest records seed, ranges,
accepted candidates, and rejection reasons. Existing families are not silently
overwritten. The topology and joint names remain the same across all sizes.

## 4. Finite-thickness joints and the physical insertion aperture

The model has **22 links, 21 revolute joints, and 24 collision boxes**. It includes
four walls, two opposite tuck lids, four dust flaps, a lap seam, and two receivers.
The zero configuration is a flat net in root-frame XY, with sheet thickness
along Z. Root `back_wall` is fixed by the preview; root-frame Y is carton height.

Three important chains are:

```text
wall --revolute-- lid-root margin --revolute-- lid
lid  --revolute-- tuck margin     --revolute-- tuck tab
wall --revolute-- socket ears     --revolute-- keeper
```

For board thickness `t` and requested clearance `c` on each tab face, the design
uses:

```text
clear slot aperture g = t + 2*c
wall/keeper midplane separation s = g + t = 2*t + 2*c
receiver and tuck hinge-axis spacing a = s
```

For the default `t=2.0 mm`, `c=0.6 mm`, the clear aperture is **3.2 mm** and the
hinge-axis spacing is **5.2 mm**. Finite thickness must be included: using a
3.2 mm midplane spacing would leave only 1.2 mm of air between 2 mm panels.

**The socket mouth is genuinely empty.** Each socket-margin link contains two
separate rectangular ear collision boxes, one on each side of the tab path.
The keeper returns downward behind the receiving wall. A full-width strip
across the mouth would block insertion, so no such collider is present. Never
replace the two ears with one convex hull during import. The two ears share a
rigid-link motion; this is an idealization of their synchronized crease bending.

There is also small collision relief at crease centers (`0.55*t` per adjacent
edge) to prevent finite-thickness boxes from overlapping as they rotate. Thus
crease-to-crease spacing is not identical to the shorter solid span of a strip.
For a 2.6 mm margin spacing, the solid rectangular span is 1.5 mm. These reliefs
represent unmodeled crease regions; this is not a fabrication-ready cutting
pattern. `flat_net.svg` is a projection of simulation geometry, not a die line.

## 5. Folding and insertion motion

The schedule deploys receivers, raises the side/front walls, folds the offset
lap seam, closes the bottom dust flaps and bottom tuck lid, then closes the top
dust flaps and top tuck lid. Approach, alignment, and insertion are distinct
stages. Quintic ramps start and stop each stage smoothly. Lid-root margin joints
and the seam-margin joint are available articulations but are held at zero in
this particular trajectory.

During insertion the tab remains vertical and centered in its slot. Let `theta`
be the lid angle, `alpha` the absolute tuck-margin angle, `a` the tuck-margin
axis spacing, and `L=D-s/2` the main lid hinge-to-hinge length. The chosen branch
satisfies:

```text
z_tab = L*sin(theta) + a*sin(alpha) = L
alpha = pi - asin((L/a) * (1 - sin(theta)))
q_lid = theta
q_tuck_margin = alpha - theta
q_tuck_tab = pi - alpha
```

The tab-root height above the mouth is:

```text
h_root = a + L*cos(theta) + a*cos(alpha)
```

The entry angle is solved so `h_root - tab_length = entry_tip_clearance`.
As `theta` approaches pi/2, the tab root lowers to the mouth while its lateral
position and orientation remain fixed. This gives a genuine downward insertion
through the aperture using revolute joints alone, not teleportation of a tab
through a solid panel. Bottom insertion uses the mirrored construction.

`output/trajectory.json` stores 120 Hz samples with joint names and stage labels;
`assembled_configuration.json` stores final angles. Use `Motion.evaluate(t)` for
the exact continuous joint targets rather than linearly interpolating a sparse
trajectory. All link positions are computed from joint FK.

## 6. What was tested

See `output/validation_summary.json`, the per-model validation reports, and
`output/test_results.txt` for the exact execution record. The default model was
checked at 5,015 unique configurations (5,001 uniform samples plus distinct stage
boundaries), with no inter-link box penetration detected at a 1e-8 m tolerance.
The twenty seeded random models were each checked at 1,015 configurations.
Eight additional aspect-ratio/thickness presets also passed.

The checker tests the actual URDF collision boxes using a 15-axis oriented-box
separating-axis test after an axis-aligned broad phase. It does **not** disable
neighboring-hinge collisions or tab/keeper/wall pairs. It checks positive mass
and valid inertia, the tree structure, positive dimensions, joint limits,
sampled velocity limits, open socket mouths, and tab clearances. Default seated
clearance is 0.3 mm on both broad faces and 1.0 mm at each lateral mouth edge.
Default tip insertion depth is 14 mm.

Seven test methods passed; the eighth, the actual-urdfpy comparison, was skipped
because the dependency could not be installed. Tests and reports are sampled
rigid-geometry checks, **not continuous-time collision proofs**.

## 7. Simulation boundaries

This is a contact-ready rigid-panel approximation, not a deformable-cardboard
solver or a validated physical assembly experiment. It does not model crease
plasticity, distributed bending, buckling, adhesive bonding, or an undercut
locking mechanism. Density, damping, and effort limits are illustrative inputs,
not identified cardboard properties. The preview directly prescribes joint
angles; it does not establish that specified torques can execute the motion.

In a dynamics simulator, choose which creases are actuated versus passive,
configure hinge stiffness/damping, enable self-collision, and tune the time step,
solver, contact/rest offsets, and position-drive gains at the actual thickness
scale. Preserve the separate ear shapes and all tab/receiver collision pairs.
A nonzero rest separation comparable to the available clearance can make a
nominally open slot unusable. Contact generation distance is not the same thing
as hard geometric inflation; tune it according to the engine rather than
assuming a large contact offset physically fills the aperture.

URDF has a tree topology. The lap flap reaches its final overlap pose but is not
glued to the opposite wall by an extra URDF joint. Add any persistent seam bond
or loop-closure constraint in the simulator. Similarly, a seated tab is not
magically latched: retention would require contact forces, friction, elastic
preload, or a separately modeled locking feature.

## References

The real-package calls follow the official urdfpy API:
- [urdfpy examples: loading, FK, and visualization](https://urdfpy.readthedocs.io/en/latest/examples/)
- [URDF API, including animate timing](https://urdfpy.readthedocs.io/en/latest/generated/urdfpy.URDF.html)
- [Upstream dependency declaration](https://github.com/mmatl/urdfpy/blob/master/setup.py)
- [ROS URDF tree model and joint-frame convention](https://docs.ros.org/en/noetic/api/urdf/html/index.html)
- [PhysX contact and rest offsets](https://nvidia-omniverse.github.io/PhysX/physx/5.8.0/docs/AdvancedCollisionDetection.html)
