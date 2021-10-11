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

        cout << "Next action > ";
        int next;
        cin >> next;

        int* mask = new int[env->policySize()];
        env->legalMask(mask);

        if (!mask[next]) {
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