=================
MARL Environments
=================

This section provides detailed documentation for the multi-agent negotiation environments included in the library.

Each environment follows the standard interface described in :doc:`../environments` but has its own unique game rules,
dynamics, and implementation details.

.. toctree::
   :maxdepth: 2
   :caption: Available Environments:

   environments/ipd
   environments/diplomacy
   environments/dond

Overview
--------

The library currently includes the following environments:

1. **Iterated Prisoner's Dilemma (IPD)**: A classic game theory problem where two agents repeatedly decide whether to cooperate or defect, with different payoffs based on their joint actions.

2. **Diplomacy**: An adaptation of the board game Diplomacy, where seven European powers compete for control of supply centers through strategic moves and alliances.

3. **Deal or No Deal (DOND)**: A negotiation environment based on `the paper Deal or No Deal? End-to-End Learning for Negotiation Dialogues <https://arxiv.org/pdf/1706.05125>`_ in which agents negotiate over the distribution of a set of prizes.

Each environment documentation includes:

- Game rules and background
- Implementation details
- API reference
- Example usage
- Advanced features and customization options