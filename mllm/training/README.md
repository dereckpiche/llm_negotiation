Suppose we have a trajectory with 3 timesteps.
token:          "0 1 2 3 4 5 6 7 8 9 . . . . ."
string:         "A B C a b c A a A a b c A B C" (Capitalized = User, Lowercased = Assistant)
action_mask:    "x x x ✓ ✓ ✓ x ✓ x ✓ ✓ ✓ x x x" (F = False, T = True)
rewards:        "r r r r r r R R R R R R r r r"
timestep:       "0 0 0 0 0 0 1 1 1 1 1 1 2 2 2"
state_ends:     "x x ✓ x x x ✓ x x x x x x x ✓"

There must be one baseline flag per timestep!

Then, we might have

A naive way to interpret this is to think of the number of assistant messages as the number of
steps in the environment. However, this is not the case in practice. Indeed, in a
single simulation step,




A subtlety arises with credit assignment. In the multi-agent case, we might
