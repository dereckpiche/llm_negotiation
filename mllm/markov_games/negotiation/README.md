## Negotiation games: core mechanics and variants

These games feature two agents who, in each round, may briefly communicate and then simultaneously propose how to split a fixed resource (most commonly 10 coins). Rewards are the amount kept multiplied by an agent’s per-unit value. The starting speaker alternates deterministically across rounds.

Communication is optional and variant-dependent: some settings encourage rich messaging to share private information, while others remove messaging entirely to focus on allocation behavior.

Proportional splitting is used when the two proposals exceed the available total: allocations are scaled proportionally rather than discarded. This preserves a useful learning signal even when agents over-claim.

### Variants

- Classic
  - Single item type (coins); each round, each agent’s per-coin value is independently sampled in a broad range (e.g., 1–20).
  - Each agent observes only their own value; they may use short messages to share and negotiate.
  - Motivation: a simple blend that tests whether agents learn to exchange private information and coordinate proportional, value-aware splits.

- Trust-and-Split (TAS)
  - Each round, a rock–paper–scissors hand draw creates a strong asymmetry: the winner’s per-coin value is 10, the loser’s is 1.
  - Each agent initially sees only their own hand and must communicate to coordinate a fair split.
  - Motivation: enforce large value disparity so one’s own value reveals little about the other’s (avoiding ceiling effects) and incentivize meaningful communication.

- Deal-or-No-Deal (DOND)
  - Multiple item types (typically "books", "hats" and "balls") with limited stocks; each agent has its own per-type values.
  - A deal pays out only if both proposals exactly agree and respect the stock; otherwise no deal (zero reward) that round.
  - Motivation: a known benchmark closer to real-world bargaining, where both parties must explicitly agree.

- Deterministic No‑Press
  - Single item type (coins); values are fixed and public: one agent values coins at 10, the other at 1.
  - No communication; agents go straight to making split proposals, with the starting player alternating deterministically.
  - Motivation: mirrors no‑communication setups (e.g., Advantage Alignment) while keeping the split decision nontrivial.





