#pragma once

/*
 * Constant parameters used throughout the program.
 */

// ===========  MCTS parameters ===========

#define MCTS_PUCT             4.0  // magnitude of policy + exploration terms
#define MCTS_TARGET_NODES     300  // Target tree size before making a decision
#define MCTS_PRESERVE_SUBTREE true // Preserve subtree after making action?