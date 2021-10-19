#include "environment_c4.h"
#include "environment.h"

using namespace std;

#include <stdexcept>

vector<int> Environment::getLegalActions() {
    vector<int> output;
    int psize = policySize();
    int* mask = new int[psize];

    for (int i = 0; i < psize; ++i) {
        if (fabs(mask[i]) > 0.5f) {
            output.push_back(i);
        }
    }

    delete[] mask;
    return output;
}

Environment* Environment::initFromName(string name) {
    if (name == "Connect4") {
        return new C4Environment();
    }

    throw std::runtime_error("Unknown environment type " + name);
}