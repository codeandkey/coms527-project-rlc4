#include <mpi.h>
#include <omp.h>
#include <torch/torch.h>

#include <fstream>
#include <iostream>
#include <mutex>
#include <thread>

#include "environment.h"
#include "environment_c4.h"
#include "params.h"

using namespace std;

// Message types sent between nodes
static int MSG_READY = 0;

int main(int argc, char** argv) {
    int rank, nprocs;
    int loaded_generation;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (rank == 0) {
        cout << "MPI ready, querying GPUs.." << endl;
    }

    // Identify GPU availability
    int node_has_gpu = torch::cuda::is_available(), *node_gpu_status;
    
    if (rank == 0) {
        node_gpu_status = new int[nprocs];
    }

    MPI_Gather(&node_has_gpu, 1, MPI_INTEGER, node_gpu_status, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);

    // Decide node architecture
    int num_inference_nodes, num_training_nodes, num_actor_nodes;
    vector<int> actor_nodes, training_nodes, inference_nodes;

    if (rank == 0) {
        vector<int> gpu_nodes, cpu_nodes;

        for (int i = 0; i < nprocs; ++i) {
            if (node_gpu_status[i]) {
                gpu_nodes.push_back(i);
            } else {
                cpu_nodes.push_back(i);
            }
        }

        cout << "== Initializing node architecture ==" << endl;

        // Write accerlerated node info
        cout << "Accelerated nodes:";
        for (auto i : gpu_nodes) {
            cout << " " << i;
        }
        cout << endl << "CPU nodes:";
        for (auto i : cpu_nodes) {
            cout << " " << i;
        }
        if (!cpu_nodes.size()) cout << " (none)";
        cout << endl;

        // Check there are enough GPUs
        if (gpu_nodes.size() < 2) {
            cout << "ERROR: Only " << gpu_nodes.size() << " GPU nodes available, at least 2 required for training+inference" << endl;
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        // Decide on node tasks
        
        // If only GPU nodes available, perform manual allocation
        if (!cpu_nodes.size()) {
            cout << "NOTE: Performing GPU-only allocation" << endl;

            if (gpu_nodes.size() < 3) {
                cout << "ERROR: Only " << gpu_nodes.size() << " total nodes available (only GPU), at least 3 required for training+inference+actor" << endl;
                MPI_Abort(MPI_COMM_WORLD, -1);
            }

            training_nodes.push_back(gpu_nodes[0]);
            inference_nodes.push_back(gpu_nodes[1]);

            actor_nodes = vector<int>(gpu_nodes.begin() + 2, gpu_nodes.end());
        } else {
            // Allocate 1 acclerated training thread, the rest of the GPU nodes as inference
            // All CPU nodes will be used as actors

            training_nodes.push_back(gpu_nodes[0]);
            inference_nodes = vector<int>(gpu_nodes.begin() + 1, gpu_nodes.end());
            actor_nodes = cpu_nodes;
        }

        cout << "Training nodes:";
        for (auto i : training_nodes) {
            cout << " " << i;
        }
        cout << endl << "Inference nodes:";
        for (auto i : inference_nodes) {
            cout << " " << i;
        }
        cout << endl << "Actor nodes:";
        for (auto i : actor_nodes) {
            cout << " " << i;
        }
        cout << endl;

        num_actor_nodes = actor_nodes.size();
        num_training_nodes = training_nodes.size();
        num_inference_nodes = inference_nodes.size();

        cout << "Total parallel simulations: " << ENVS_PER_ACTOR << " by " << actor_nodes.size() << " => " << ENVS_PER_ACTOR * actor_nodes.size() << endl;
        cout << "== Finalized architecture ==" << endl;
    }

    // Wait for node architecture counts from root
    MPI_Bcast(&num_actor_nodes, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_inference_nodes, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_training_nodes, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);

    // Resize buffers and receive node identity
    if (rank != 0) {
        actor_nodes.resize(num_actor_nodes);
        inference_nodes.resize(num_inference_nodes);
        training_nodes.resize(num_training_nodes);
    }

    MPI_Bcast(&actor_nodes[0], num_actor_nodes, MPI_INTEGER, 0, MPI_COMM_WORLD);
    MPI_Bcast(&training_nodes[0], num_training_nodes, MPI_INTEGER, 0, MPI_COMM_WORLD);
    MPI_Bcast(&inference_nodes[0], num_inference_nodes, MPI_INTEGER, 0, MPI_COMM_WORLD);

    for (auto i : actor_nodes) {
        if (i == rank) {
            cout << "Rank " << i << " : I am an actor node" << endl;
        }
    }
    for (auto i : training_nodes) {
        if (i == rank) {
            cout << "Rank " << i << " : I am a training node" << endl;
        }
    }
    for (auto i : inference_nodes) {
        if (i == rank) {
            cout << "Rank " << i << " : I am an inference node" << endl;
        }
    }
    /*
    // If master, spawn a training process thread
    if (rank == 0) {
        // Initialize model
        if (!fstream(MODEL_PATH)) {
            if (generate_model()) {
                MPI_Abort(MPI_COMM_WORLD, -1);
            }

            cout << "(stub) Generating new model at " << MODEL_PATH << endl;

            ofstream f(MODEL_PATH);
            f << "1" << endl;
            f.close();
        }

        // (TODO)
        auto t = thread([=]() {
            // Build training set

            MPI_File f;
            MPI_File_open(MPI_COMM_WORLD, "generation", MPI_MODE_WRONLY, MPI_INFO_NULL, &f);

            MPI_Status s;
            MPI_File_write(f, &loaded_generation, 1, MPI_INTEGER, &s);
        });
    }*/

    // Wait for root node
    //MPI_Bcast(NULL, 0, MPI_INTEGER, 0, MPI_COMM_WORLD);

    // Start generating games!
    MPI_Finalize();
    return 0;
}