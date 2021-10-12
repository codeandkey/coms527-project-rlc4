#include <mpi.h>
#include <omp.h>
#include <torch/torch.h>

#include <iostream>

#include "environment.h"
#include "environment_c4.h"

using namespace std;

int main(int argc, char** argv) {
    Environment* env = new C4Environment();

    cout << "Environment type " << env->getName() << endl;

    float tval;

    while (!env->terminal(&tval)) {
        cout << env->getString();

        int next;
        cout << "Next action > ";
        cin >> next;

        float* mask = new float[env->policySize()];
        env->legalMask(mask);

        if (fabs(mask[next]) < 0.01f) {
            cout << "Illegal action\n";
            delete[] mask;
            continue;
        }

        delete[] mask;

        env->push(next);
    }

    cout << "Terminal state reached, value " << tval << endl;

    return 0;
}