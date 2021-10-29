#pragma once

/*
 * Constant parameters used throughout the program.
 */

// =========== config ===========
#define MODEL_PATH "models/model.pt"
#define TAG_SIZE 32

// ===========  Environment     ===========
#define SELECTED_ENV Connect4
#define WIDTH SELECTED_ENV::width
#define HEIGHT SELECTED_ENV::height
#define FEATURES SELECTED_ENV::features
#define PSIZE SELECTED_ENV::policy_size

// ===========  MCTS parameters ===========

#define MCTS_PUCT             4.0  // magnitude of policy + exploration terms
#define MCTS_TARGET_NODES     300  // Target tree size before making a decision
#define MCTS_PRESERVE_SUBTREE true // Preserve subtree after making action?

// ===========  Concurrency parameters ===========

#define ENVS_PER_ACTOR  32 // Number of parallel simulations per actor node
#define INFERENCE_BSIZE 32 // Inference batchsize

// ===========  Message types  ============

#define MSG_REQUEST_INFERENCE 0
