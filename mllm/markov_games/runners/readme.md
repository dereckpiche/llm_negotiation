Every method (except linear runner) allows non-linear trajectory generation. Trajectories may
branch out into sub-trajectories at different states according to different schemes.
This type of generation is required for AdAlign* and VinePPO, for instance.

These runners use `asyncio` with `SgLang` to generate files asynchronously
and leveraging automatic `SGLang` server API batching of generations.

At a high level, these methods should export a bunch of trajectory json files and a
tree structure. The trajectory files should not overlap/repeat themselves. For instance,
if a sub-trajectory `st` branches out from `t` at time step `i`, then the file of
`st` should only contain historical information from `i` to `T` (terminal time step).
The implicit standardization of the book keeping of step information in `markov_game.py`
helps following this constraint.
