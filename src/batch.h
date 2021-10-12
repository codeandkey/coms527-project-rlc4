#pragma once

/*
 * Basic input batch type. Manages buffers for input and legal mask layers.
 */

#include <vector>

#include "environment.h"

class Batch {
    public:
        /**
         * Initializes an empty input batch.
         * 
         * @param maxsize [in] Maximum frames in the batch.
         */
        Batch(Environment* env, int maxsize);

        /**
         * Adds an input frame to the batch.
         * 
         * @param env Environment state to add.
         */
        void add(Environment* env);

        /**
         * Gets the number of input frames in this batch.
         *
         * @return Input frame count
         */
        int getSize();

        /**
         * Gets a pointer to the environment input data.
         *
         * @return pointer to input data
         */
        float* getInputData();

        /**
         * Gets a pointer to the legal move mask data.
         *
         * @return pointer to lmm data
         */
        float* getMaskData();

    private:
        std::vector<float> batch_input_data, batch_mask_data;
        int frames, frame_size;
};

