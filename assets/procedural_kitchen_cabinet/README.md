# Procedural kitchen cabinet

Countertop kitchen cabinets built entirely from axis-aligned boxes (every
visual/collision element is convex, so Genesis needs no decomposition), plus a
counter and a pan trivet. Regenerate and validate from the repository root:

```bash
conda run -n genesis python assets/procedural_kitchen_cabinet/generate.py            # all presets
conda run -n genesis python assets/procedural_kitchen_cabinet/generate.py \
    --preset single_door --door_open_limit 1.9 --handle_grip_height 0.34       # field overrides
conda run -n genesis python assets/procedural_kitchen_cabinet/validate.py      # sweep checks
```

`KitchenCabinetConfig` (dataclass in `generate.py`) holds every dimension;
every field is a `--<field>` CLI override on top of a `--preset`.

## Frames

- **Cabinet frame**: origin at the center of the carcass bottom, front opening
  toward **-x**, z up. All dimensions below use it.
- **URDF root frame** (`carcass` link): the cabinet frame translated to the
  closed primary door's grip center, `root_origin_in_cabinet_frame` in
  `layout.json` = (-0.2795, -0.1735, 0.36) m for `single_door`. This puts the
  retargeting BPS target (the bar pull) at the origin of the human source
  object (the C25001 toothbrush long axis is also the z axis), so retargeting
  starts near the solution. Place the cabinet with
  `root_position = cabinet_origin + root_origin_in_cabinet_frame`.

## `kitchen_cabinet_single_door` (used by V2_17)

| Part | Dimensions |
| --- | --- |
| Carcass (outer) | 0.44 (y) x 0.40 (x, depth) x 0.50 (z) m; 18 mm panels, 10 mm back |
| Interior | x [-0.200, 0.190], y [-0.202, 0.202], z [0.018, 0.482] |
| Shelf | top at z = 0.22, full width, from x = -0.053 to the back (recessed 0.147 m from the front so the pan handle and the front 2 cm of the pan overhang it: the xHand fingers wrapped under the handle, and their downward swing when the hand opens, stay in free space) |
| Door (`door`) | overlay, 0.437 x 0.497 x 0.018 m, 1.5 mm gaps; hinge on the +y edge of its inner face, joint `door_hinge`, axis -z, limits [0, 115 deg], damping 0.3, friction 0.05; positive angles swing the free edge outward |
| Pull (`door_handle`, fixed to `door`) | vertical bar 10 x 18 x 180 mm, axis 60 mm in front of the door face (55 mm finger gap, sized so the flat-opened xHand fingers clear the door), 45 mm from the free edge, grip center 0.36 m above the carcass bottom; two 16 x 12 mm posts |

Masses/inertias are computed from the boxes (wood 600 kg/m^3, pull 2700 kg/m^3).

## `kitchen_cabinet_double_door_drawer` (optional variant)

0.60 x 0.40 x 0.66 m carcass, shelf at 0.40 m, two doors (`door_hinge` at +y,
`door_2_hinge` at -y, 3 mm center gap), and a bottom drawer bay (0.14 m) with
drawer `drawer` on prismatic joint `drawer_slide` (axis -x, 0-0.26 m) and a
horizontal bar pull `drawer_handle`. Not used by a task yet.

## Supports (`support/`)

- `counter.urdf`: 1.4 x 1.6 x 0.06 m slab, frame on its **top** surface (z = 0).
- `pan_trivet.urdf`: 0.17 x 0.16 x 0.08 m block, frame on its **bottom** face.
  The pan rests on it with the handle overhanging its front edge, again so the
  wrapped fingers under the handle do not hit a support.

## Height choice

The xHands are floating-base (no arm), so absolute height is irrelevant for
reachability. The counter top is the world z = 0 plane and the cabinet stands
on it (a countertop "appliance garage" rather than a wall cabinet at 1.4 m).
The geometry otherwise has kitchen scale: 0.40 m deep carcass, 18 mm panels,
10 mm bar pull (finger clearance 55 mm, a deep appliance-style pull).

## Generated files

`<name>/<name>.urdf`, `<name>/layout.json` (all derived quantities consumed by
the V2_17 spec generator), and `<name>/preview_{closed,open}.obj` (combined
meshes at q = 0 and q = upper limits). `validation_report.json` holds the last
validation (kinematic sweep penetration between each moving link and the rest;
Genesis PD sweep 0 -> upper -> 0 under gravity).
