# active-inference-self-formation

## Overview
This project implements a hierarchical Active Inference model 
to study self-formation and policy conflict resolution.

The key idea is that conflicts between competing actions cannot 
always be resolved at the level of immediate action selection, 
but instead require higher-level inference that gives rise to 
self-related representations.

## Reference
This code is based in part on the open-source `pymdp` framework, with small task-specific modifications (particularly related to D-learning) for the present simulations.

## Files
- `Hierarchical.py` : hierarchical model with higher-level self inference
- `Non-hierarchical.py` : baseline model without top-down control of a higher agent
