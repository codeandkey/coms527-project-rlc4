#include "actor.h"
#include "environment.h"
#include "connect4.h"
#include "mcts.h"
#include "params.h"

#include <iomanip>
#include <random>
#include <sstream>
#include <vector>

using namespace std;

int actor::run()
{
    // Initialize environment trees
    vector<Tree> trees(ENVS_PER_ACTOR);

    // Initialize environment tags
    default_random_engine dev;
    uniform_int_distribution<char> dist(0, 0xff);
    vector<char> tags(ENVS_PER_ACTOR * TAG_SIZE, 0);

    for (int i = 0; i < ENVS_PER_ACTOR * TAG_SIZE; ++i)
        tags[i] = dist(dev);

    // Init input batch
    vector<float> batch(WIDTH * HEIGHT * FEATURES * ENVS_PER_ACTOR, 0.0f);

    // Start simulation loop
    while (1)
    {
        // Perform tree selections

        // Build observation buffer
        for (int i = 0; i < ENVS_PER_ACTOR; ++i) {
            shared_ptr<Environment> result = trees[i].simulate();

            if (result == nullptr) {
                // Target n reached. Make a decision and advance the environment
                int action = trees[i].chooseAction();
                trees[i].advance(action);

                // Send tagged trajectory to training thread
            }
        }
    }

    return 0;
}
