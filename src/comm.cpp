#include "comm.h"

#include <iostream>
#include <vector>

#include <mpi.h>
#include <torch/torch.h>

using namespace std;

static int our_rank, nprocs;
static vector<bool> node_cuda_support;
static vector<int> cpu_nodes, gpu_nodes;
static map<int, cluster::Identity> node_identity;

void cluster::init(int* argc, char*** argv) {
    // Initialize MPI
    MPI_Init(argc, argv);

    // Query rank, total nodes
    MPI_Comm_rank(MPI_COMM_WORLD, &our_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Query CUDA status
    int gpu_enabled = torch::cuda::is_available();

    // Collect CUDA status of all nodes
    int* gpu_status = new int[nprocs];

    MPI_Allgather(&gpu_enabled, 1, MPI_INT, gpu_status, 1, MPI_INT, MPI_COMM_WORLD);

    node_cuda_support.empty();

    for (int i = 0; i < nprocs; ++i) {
        if (gpu_status[i]) {
            gpu_nodes.push_back(i);
        } else {
            cpu_nodes.push_back(i);
        }

        node_cuda_support.push_back(gpu_status[i]);
    }

    // Deterministically decide node identity

    // We shoot for 1 training node, and 4 actors per inference node.
    if (gpu_nodes.size() < 2) {
        if (!our_rank) {
            cerr << "Need at least 2 GPU nodes for training+inference, only detected " << gpu_nodes.size() << endl;
        }

        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    // Initialize all nodes to actor
    for (int i = 0; i < nprocs; ++i) {
        node_identity[i] = ACTOR;
    }

    // First GPU node becomes a trainer
    node_identity[gpu_nodes[0]] = TRAINING;

    // Second GPU node becomes inference
    node_identity[gpu_nodes[1]] = INFERENCE;

    // Determine number of additional inference nodes
    int inference_count = (((int) gpu_nodes.size() - 2) + (int) cpu_nodes.size() - 4) / 4;

    for (int i = 0; i < inference_count; ++i) {
        if (i + 2 >= gpu_nodes.size()) break;

        node_identity[gpu_nodes[i + 2]] = INFERENCE;
    }
}

void cluster::destroy() {
    MPI_Finalize();
}

void cluster::abort(string msg) {
    cerr << "Node " << our_rank << " aborting program: " << msg << endl;
    MPI_Abort(MPI_COMM_WORLD, -1);
}

cluster::Identity cluster::identity() {
    return node_identity[our_rank];
}