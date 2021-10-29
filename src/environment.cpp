#include "environment.h"
#include "params.h"

using namespace std;

#include <cmath>
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
