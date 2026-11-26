# Procedural hinged-lid mailer: URDF + assembly motion

A rigid-panel interpretation of the supplied hinged-lid cardboard mailer photo.
It includes a rectangular tray, four corner ears, double-fold side walls,
four base locking tongues, a rear-hinged lid with two side wings, and a front tuck.

**The reference photo is not a measured die line.** Hidden creases and locking
cuts cannot be reconstructed uniquely from that view. This is a deliberately
rectangularized, simulation-oriented design with explicit finite-thickness
clearances, not an exact manufacturer/FEFCO design or a fabrication-ready net.

## Execution status

The procedural generator, independent reference forward kinematics, software
renderer, regression tests, and sampled collision validator were executed.
**The delivered previews use `--backend reference`, not urdfpy.** Installing
urdfpy failed in the build environment because external package/DNS access was
unavailable. The `urdfpy` execution path is provided but was not run here.

The default playback backend really imports the upstream package and uses
`URDF.load` and `link_fk`; it never silently falls back to a substitute. The
native interactive viewer additionally uses `URDF.animate`. Backend provenance
is written into every preview footer and `output/render_provenance.json`.

This is **kinematic assembly**, not a contact-dynamics rollout. Contact retention,
cardboard flexibility, corrugation, crease plasticity, and friction are not tested.

## Contents

- `build_mailer.py`: procedural URDF generator and editable `Parameters` class.
- `mailer.urdf`, `mailer.json`: default model and resolved geometry/trajectory metadata.
- `mailer_assembly_states.json`: Genesis-read phase starts/goals and plastic rest states.
- `motion.py`: continuous time-to-joint-angle function, including analytic insertions.
- `fold_mailer.py`: animation, screenshots, sections, native viewer, and trajectory export.
- `kinematics.py`: actual urdfpy adapter and explicitly named reference FK.
- `validate_mailer.py`, `collision.py`: sampled geometry and URDF checks.
- `generate_family.py`, `render_family.py`: seeded families and comparative previews.
- `test_mailer.py`: regression tests and an optional actual-urdfpy FK cross-check.
- `output/`: GIF/MP4, open/closed/flat views, three slot sections, reports and trajectories.
- `examples/`: eight aspect-ratio/board-thickness presets and twelve seeded samples.
- `configs/`: editable dimension inputs; omitted/null detail dimensions are automatic.

The default topology is **27 links, 26 revolute joints, and 38 collision boxes**.
There are no external mesh files. Every link has positive mass and inertia,
including parallel-axis terms for compound links. Density is an illustrative
homogeneous bulk value, not a measured material property.

## 1. Run with urdfpy

Use a separate Python 3.10 or 3.11 environment for these compatibility-oriented
requirements. The installation below was not executable in the build sandbox.

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-urdfpy.txt
python -m pip install --no-deps urdfpy==0.0.22

# All geometric dimensions are metres.
python build_mailer.py --out models/default

# Uses real urdfpy FK, with software image rendering; no OpenGL window needed.
python fold_mailer.py --urdf models/default/mailer.urdf \
  --backend urdfpy --output-dir previews/default --fps 12

# Interactive urdfpy/pyrender viewer; needs a working desktop OpenGL context.
python fold_mailer.py --urdf models/default/mailer.urdf \
  --backend urdfpy --native-viewer

# Test geometry and poses from that same urdfpy backend.
python validate_mailer.py --urdf models/default/mailer.urdf \
  --backend urdfpy --samples 3001 --report previews/default/validation.json
```

The upstream package declares `networkx==2.2`. The requirements deliberately
install NetworkX 2.8.8 instead; `--no-deps` prevents urdfpy from downgrading it.
Consequently, `pip check` reports that declared dependency mismatch. A small
local NumPy proxy in `load_urdfpy` provides legacy scalar aliases to the upstream
URDF module without modifying NumPy globally. The optional regression test
compares real urdfpy poses to independent FK and is skipped when urdfpy is absent.

The native `animate` API treats trajectory samples as uniformly timed and plays
forward and backward. The wrapper passes `loop_time=2*motion.total`. Exported
video/GIF shows forward assembly only, including a final closed hold.

### Explicit dependency-light backend

These are the code paths used to create the supplied artifacts:

```bash
python -m pip install -r requirements.txt
python fold_mailer.py --backend reference --output-dir previews/offline --fps 10
python validate_mailer.py --backend reference --samples 3001
python -m unittest -v test_mailer
```

Use `--snapshots-only` to skip animation or `--no-mp4` to skip FFmpeg. MP4 failure
is reported; a successful GIF does not masquerade as a successful MP4. Full
software rendering can take a few minutes on a CPU.

## 2. Change dimensions

```bash
python build_mailer.py \
  --width 0.280 --depth 0.200 --height 0.060 --thickness 0.002 \
  --out models/custom

python fold_mailer.py --urdf models/custom/mailer.urdf \
  --backend urdfpy --output-dir previews/custom

python validate_mailer.py --urdf models/custom/mailer.urdf \
  --backend urdfpy --samples 3001 --report previews/custom/validation.json

# JSON inputs also work; explicit CLI values override JSON values.
python build_mailer.py --config configs/custom.json --out models/from_json
```

Available fields/flags:

| Field | Meaning | Default |
|---|---|---|
| `width` | Opposing outer-side-wall hinge-midplane span | 400 mm |
| `depth` | Rear-to-front wall hinge-midplane span | 300 mm |
| `height` | Outer-wall fold-line height | 90 mm |
| `thickness` | Rigid panel thickness | 3.0 mm |
| `face_clearance` | Required air clearance on each broad tab face | 0.30 * thickness |
| `lateral_clearance` | End clearance for floor/front tongues | thickness |
| `lock_width` | Width of each of the four floor tongues | 0.12 * depth |
| `lid_wing_depth` | Lid-wing crease-to-tip length | 0.60 * height |
| `front_tab_width` | Width of the front tuck | 0.70 * width |
| `front_tab_length` | Front tuck crease-to-tip length | Reachability-limited |
| `density` | Illustrative homogeneous-sheet bulk density | 450 kg/m^3 |

CLI flags use hyphens, e.g. `--face-clearance`, while JSON uses underscores.
Clearance overrides specify minimum/design geometry, not physics-engine margins.

Omitted or null detail parameters are recalculated. This regenerates panel spans,
clearance strips, slot locations, corner-ear sizes, lid-wing end notches and
insertion poses, rather than scaling a fixed triangulated mesh. Joint names and
topology are identical across the family. Uniformly scaling W/D/H and thickness
also uniformly scales geometry while leaving the joint trajectory unchanged.

**Dimension convention matters:** W/D/H are nominal fold-line dimensions, not
exact exterior dimensions or a guaranteed unobstructed interior cavity. The
default closed collision geometry occupies approximately **403.0 x 303.0 x
102.3 mm**, including thickness, the raised lid, and protruding floor tongues.
The validator measures the actual closed axis-aligned envelope for every model.

The motion is designed for shallow mailers, not arbitrary tall cartons. It
conservatively requires

```text
W > 2 * (H + rollover_spacing + foot_margin + 3*t) + 2*t
H >= 18*t
```

There are additional keeper-fit, wing-window, and insertion-reach constraints.
Explicit infeasible overrides raise errors; the generator does not silently
shrink an explicitly requested flap. Always validate after changing dimensions.

## 3. Make a family

```bash
python generate_family.py --count 20 --seed 7 --backend urdfpy \
  --width-range 0.15 0.32 --depth-range 0.11 0.26 \
  --height-range 0.024 0.070 --thickness-range 0.0007 0.0025 \
  --samples 1001 --out family/new_run

python render_family.py examples/small/mailer.urdf examples/wide/mailer.urdf \
  examples/square/mailer.urdf examples/shallow/mailer.urdf \
  examples/large/mailer.urdf examples/long/mailer.urdf \
  --backend reference --out previews/variants.png
```

A family only accepts candidates that pass the requested sampled validation.
The manifest records all attempted dimensions, rejection reasons, seed and
accepted paths. Existing family folders are not overwritten. The illustrated
variants are independently fitted in their views, not shown at a common scale.

## 4. Real openings and extra articulated strips

The all-zero configuration is a flat net in root-frame XY. `base` is the root,
world Z is up, the rear wall is at Y=0, the front wall at Y=D, and side walls are
at X=+/-W/2. The preview fixes the base frame; it never teleports other links.

Three insertion mechanisms are modeled.

### A. Side-wall rollovers and four floor tongues

```text
base --R-- outer side wall --R-- rollover margin --R-- inner liner
                                                    |
                                                    R
                                                    |
                                                foot margin
                                                 /       \
                                                R         R
                                                |         |
                                             tongue 0  tongue 1
```

The rollover margin puts the liner inside the outer wall with sufficient room
for the corner ears. The lower foot margin is a real rectangular link with a
revolute joint on each relevant side. Its two independently articulated tongues
are guided into four true rectangular holes in the base.

The base is partitioned into non-overlapping rectangular solid collision pieces.
The hole regions contain **no visual or collision solid**, and their removed
volume is checked. Do not replace that compound shape with one convex hull.

### B. Lid wings and open side channels

Each side rollover margin consists of **two separated bridge ears**, near the
rear and front corners. There is no continuous bridge across the center. That
leaves a long open mouth between the outer wall and the returning liner. The
lid wings enter those channels as the lid lowers. Their end notches account for
the swept motion, not only the final closed pose. Corner ears occupy the end
regions, not the wing insertion windows.

### C. Front tuck and returning keeper

```text
front wall --R-- two separated spacer ears --R-- front keeper
lid        --R-- front margin strip         --R-- front tuck
```

The front keeper folds down inside the tray. Its support ears are outside the
tuck path, so the central receiving mouth stays open in both visual and collision
geometry. The moving margin coordinates with the lid and flap so the tuck stays
vertical and centered throughout the final insertion.

For thickness `t` and face clearance `c`:

```text
clear aperture                        g = t + 2*c
outer/inner panel midplane spacing    s = 2*t + 2*c
rollover and front receiver spacing     = s
lid spine and front margin spacing     = s
lower foot-margin hinge spacing      A = 4*t + 2*c
```

Default values are `g=2.4 mm`, `s=3.9 mm`, and `A=6.9 mm`.
The longer lower margin supplies the lift needed to guide the floor tongue
through a narrow slot without swinging it through the solid floor.

There is collision relief around crease axes: `0.65*t` at the ordinary creases,
and `t` at the more sharply bending foot/tongue creases. Thus the solid strip
span is shorter than its hinge-to-hinge spacing. For example, the default
3.9 mm rollover spacing has a 1.95 mm solid strip span. These small unsupported
geometric regions idealize the unmodeled flexible crease, not missing hinges.
They are another reason this is not a manufacturing cutting pattern.

## 5. Assembly and insertion kinematics

The complete schedule is 31.4 seconds: raise front/rear walls, fold corner ears,
raise sides, deploy rollovers, pre-bend lower margins, align and insert four
floor tongues, deploy the front receiver, prepare lid wings/front tuck, lower
the lid, and insert the front flap. Each interpolation uses quintic easing.
No collision pair is disabled to make an insertion possible.

### Floor insertion

Use an inward horizontal coordinate `u` from the outer side wall and let
`L=H-2*t`, `A=foot_margin`, and `alpha` be the foot strip's angle above horizontal.
Set

```text
delta = asin(A * (1 - cos(alpha)) / L)
u_tab = s + L*sin(delta) + A*cos(alpha) = s + A
z_tab_root = H - L*cos(delta) + A*sin(alpha)
```

The inner liner, lower margin and tongue angles are coordinated as

```text
roll_return = pi/2 - delta
foot_fold   = -pi/2 - alpha + delta
lock_fold   = pi/2 + alpha
```

The tongue stays vertical, centered on its floor slot, while its root moves down.
The entry `alpha` is solved from the required floor-tip clearance. At the end,
`alpha=0`, the root is at `z=2*t`, and the crease-to-tip length is `3*t`; the tip
is therefore at `z=-t`. It protrudes `t/2` below the base underside. A support
plane must leave space below the slots; see simulation notes below.

### Front insertion

Let `theta` be the lid angle above horizontal, `Llid=D-s/2`, and `a=s`. Set

```text
beta = acos(Llid * (1 - cos(theta)) / a)
y_tab = Llid*cos(theta) + a*cos(beta) = Llid
z_tab_root = H + s + Llid*sin(theta) - a*sin(beta)

lid_fold              = pi/2 - theta
lid_front_margin_fold = theta + beta
lid_front_tab_fold    = pi/2 - beta
```

The flap stays vertical and centered at `Y=D-s/2`. The final root is at the
nominal mouth height H. An entry angle is numerically bracketed/solved so the
flap tip starts above the receiving mouth. Only revolute joints move.

## 6. Python API / simulator targets

```python
from pathlib import Path
from build_mailer import Parameters, generate
from kinematics import load_urdfpy
from motion import Motion

urdf_path, metadata_path = generate(
    Parameters(width=0.28, depth=0.20, height=0.06, thickness=0.002),
    Path('models/api_mailer'),
)
motion = Motion(metadata_path)
robot = load_urdfpy(urdf_path)  # The actual installed upstream package.

q, stage_name, stage_index = motion.evaluate(29.0)
poses = robot.link_fk(cfg=q)
# q maps joint names to radians. Evaluate at your simulator's own time steps
# and use these values as reference position targets, not link teleports.
```

`output/trajectory.npz` contains `time`, `joint_names`, and a position array in
radians, sampled at approximately 120 Hz. JSON is supplied as well. Keep the
URDF and its matching `mailer.json` together: trajectory geometry is parameterized
by that metadata. Open and closed angle maps are separately exported.

## 7. Validation

The default model was checked at **3,016 configurations** (3,001 uniform samples
plus distinct stage boundaries), at a `1e-8 m` geometric overlap tolerance.
There were no detected inter-link box penetrations, invalid inertias, joint-limit
violations, or sampled velocity-limit violations. The minimum sampled face
clearance in the coordinated insertions is approximately **0.90 mm**.

Eight additional presets and twelve seeded random models were also generated
and checked. `output/validation_summary.json` reports the actual total model and
configuration counts; individual reports are included beside each example.
Nine regression tests passed; the actual-urdfpy test was skipped because the
package was unavailable. These claims apply to the explicit reference backend.

The checker uses an AABB broad phase and 15-axis separating-axis tests on the
actual transformed URDF collision boxes. All different-link pairs are enabled,
including neighboring hinges, all receiver components, and all moving tabs.
Same-link primitives are one rigid body and are not tested against themselves.

**Sampling is not a continuous collision proof.** No contact solver, gravity
settling, crease torque, motor stability, glue, frictional retention, force limit,
or cardboard elasticity was simulated or validated.

## 8. Import into a dynamics simulator

Preserve compound collision geometry. In particular, convex-hulling the entire
base would fill its four slots; convex-hulling either pair of socket ears would
bridge the receiving mouth. Keep those boxes separate, or use a suitable
hole-preserving representation supported by your engine.

Anchor the `base` link for replay. If a table is present, do not place the base
underside flush on that table: the floor tabs must pass through it. A default
root Z translation of 6 mm gives room for the seated 3 mm-deep tips, but
contact settings and the support arrangement still require testing.

The TrackArt assembly task builds a directed tree from this URDF with `base` as
the root. Consecutive atomic hinge motions are assigned to both hands only when
their child links descend from different direct children of `base`; otherwise
they remain serial. Every phase stores the full 26-joint start/goal vectors and
the full plastic crease-rest vectors. `mailer_assembly_states.json` records the
same states after loading and reading them back in Genesis above the task's
platform and support block.

Tune collision/contact offsets and time steps for millimetre-scale cardboard
and 0.90 mm face clearances. Physics-engine contact margins are not the same as
the geometric air gaps specified here. The tiny strip inertias can make stiff
position control numerically difficult; use appropriate solver settings and
crease/actuator parameters for your experiment.

Tongues are rectangular and clearanced. Their insertion does not imply a snap
lock, undercut, adhesive joint, or experimentally validated frictional retention.
The kinematic script holds angles. Passive retention, manipulation forces, and
stability after releasing motors are separate modeling/control tasks.

## Upstream API references

- https://urdfpy.readthedocs.io/en/latest/examples/
- https://urdfpy.readthedocs.io/en/latest/generated/urdfpy.URDF.html
- https://github.com/mmatl/urdfpy/blob/master/setup.py

No third-party font files or upstream urdfpy source are bundled.
