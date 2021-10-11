#include <mpi.h>
#include <omp.h>

#include <iostream>

using namespace std;

int main(int argc, char** argv) {
    #pragma omp parallel
    {
        #pragma omp critical
        {
            cout << "Hello from " << omp_get_thread_num() << " of " << omp_get_num_threads() << "!" << endl;
        }
    }

    return 0;
}