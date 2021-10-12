#include "batch.h"

Batch::Batch(Environment* env, int maxsize) {
    frames = 0;

    int w, h, f;
    env->getDimensions(&w, &h, &f);

    frame_size = w * h * f;

    batch_input_data.reserve(maxsize * frame_size);
    batch_mask_data.reserve(maxsize * env->policySize());
}

void Batch::add(Environment* env) {
    batch_input_data.resize(batch_input_data.size() + frame_size);
    batch_mask_data.resize(batch_mask_data.size() + env->policySize());
    env->input(&batch_input_data[frames * frame_size]);
    env->legalMask(&batch_mask_data[frames * env->policySize()]);

    ++frames;
}

int Batch::getSize() {
    return frames;
}

float* Batch::getInputData() {
    return &batch_input_data[0];
}

float* Batch::getMaskData() {
    return &batch_mask_data[0];
}