#pragma once

#include <string>

/**
 * Generic environment class. This exposes an environment in which an agent can observe
 * its current state and make a decision.
 */
class Environment {
    public:
        /**
         * Returns the name of this environment type.
         * 
         * @return Environment name
         */
        virtual const char* getName() = 0;

        /**
         * Gets the current environment state in printable form.
         */
        virtual std::string getString() = 0;

        /**
         * Returns the size of the policy array. This is effectively the number
         * of possible decisions which can be made at any time. This method should
         * ALWAYS return a constant value.
         * 
         * @return Policy length
         */
        virtual int policySize() = 0;

        /**
         * Queries the dimensions of the environment. This is important for generating
         * network architecture; this should be constant per environment.
         * 
         * @param width [out] Width of the playing area
         * @param height [out] Height of the playing area
         * @param features [out] Input features per cell
         */
        virtual void getDimensions(int* width, int* height, int* features) = 0;

        /**
         * Loads a mask of legal actions into <dst>. This method should write '1'
         * into indices of legal actions, and '0' into indices of illegal actions.
         * 
         * 'dst' is expected to be of exactly 'policySize()' length.
         * 
         * @param dst [out] Destination mask.
         */
        virtual void legalMask(int* dst) = 0;

        /**
         * Performs an action in the environment. The action must be a legal action
         * as specified by legalMask().
         * 
         * @param ind [in] Action to perform. Must be between 0 and (policySize() - 1) inclusive.
         */
        virtual void push(int ind) = 0;

        /**
         * Unperforms the most recent action. There must be at least one action to unperform.
         */
        virtual void pop() = 0;

        /**
         * Queries the terminal status of the environment.
         * 
         * @param value [out] Terminal value, from the point-of-view of the player to move.
         * @return true if state is terminal, false otherwise
         */
        virtual bool terminal(float* value) = 0;

        /**
         * Generates an input layer from the environment. The output is in row-major order, such that
         * feature i on cell (x, y) is indexed by (y * (w * f) + x * f + i).
         * 
         * @param layer [out] Destination input layer.
         */
        virtual void input(float* layer) = 0;
};