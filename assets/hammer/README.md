# Hammer URDF

Files:
- hammer.urdf

Geometry:
- handle_link: cylinder, radius 0.015 m, length 0.3 m, mass 0.1 kg, brown
- head_link: cylinder, radius 0.035 m, length 0.12 m, mass 0.5 kg, dark grey
- fixed joint: handle_to_head_fixed

The handle is vertical along z.
The head is cylindrical and rotated so its long axis is along x.
The head center is placed at the top of the handle.
