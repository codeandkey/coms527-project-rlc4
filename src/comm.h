#pragma once

#include <string>

/*
 * This file contains methods for communicating between nodes in the cluster.
 */

namespace cluster {
    /**
     * Node identity enum.
     *
     * Nodes are assigned one of three tasks:
     *  (1) Training (GPU only)  : Processes saved experiences and trains model.
     *  (2) Inference (GPU only) : Receives observations from actors and responds with
     *                             policy/value information from the model.
     *  (3) Actor (GPU or CPU)   : Runs simulations on an environment, sends
     *                             observations to Inference nodes and waits
     *                             for responses before making decisions.
     */
    enum Identity {
        TRAINING,
        INFERENCE,
        ACTOR 
    };

    /**
     * Establishes node identity and cluster architecture.
     * Should be called at startup by each node.
     */
    void init(int* argc, char*** argv);

    /**
     * Cleans up the MPI context.
     * Should be called at the end of the program.
     */
    void destroy();

    /**
     * Aborts running processes and outputs a descriptive error message.
     *
     * @param msg Message to display on stderr before aborting
     */
    void abort(std::string msg);

    /**
     * Retrieves the identity of this node.
     *
     * @return Calling node identity
     */
    Identity identity();

    /**
     * Writes an input batch to an inference node. Returns the ID of the node
     * it is sent to. Used by actor nodes.
     *
     * @return Selected inference node
     */
    int request_inference(float* batch, char* tags);
}
