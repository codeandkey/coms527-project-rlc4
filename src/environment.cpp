#include "connect4.h"
#include "environment.h"
#include "params.h"

using namespace std;

#include <cmath>
#include <stdexcept>

vector<int> Environment::getLegalActions() {
    vector<int> output;
    int* mask = new int[PSIZE];

    for (int i = 0; i < PSIZE; ++i) {
        if (fabs(mask[i]) > 0.5f) {
            output.push_back(i);
        }
    }

    delete[] mask;
    return output;
}
