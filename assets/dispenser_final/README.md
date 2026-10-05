# Dispenser Variation Prismatic v5

Changes from v4:
- Base radius reduced by 1 cm:
  - previous: 0.045 m
  - new: 0.035 m

Retained:
- 2 cm disc transition.
- Head height: 0.015 m.
- Prismatic travel: 0.010 m.
- Downward nozzle bend.

Runtime pump mechanics:
- The URDF robot name `dispenser_variation_prismatic_v5` and prismatic joint
  `neck_to_head_prismatic` form the strict asset identity used by the Genesis RL
  loader, including generated/scaled copies of this URDF.
- The RL loader applies a position spring at the upper limit (`0.0 m`) with
  `kp=2500 N/m`, `kv=100 N s/m`, and a `10 N` restoring-force limit.
- The task-specific simulator metric measures unloaded settling,
  constant-force compression, and recovery after release.
