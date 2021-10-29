#include <string>

#include <mpi.h>

#include "actor.h"
#include "comm.h"

using namespace std;

int main(int argc, char** argv) {
    cluster::init(&argc, &argv);

    int retcode = 0;

    try {
        switch (cluster::identity()) {
            case cluster::TRAINING:
                // retcode = training::start();
                break;
            case cluster::INFERENCE:
                // retcode = inference::start();
                break;
            case cluster::ACTOR:
                retcode = actor::run();
                break;
        }
    } catch (std::exception& e) {
        cluster::abort(e.what());
    }

    if (retcode) {
        cluster::abort(string("nodetask returned errorcode ") + to_string(retcode));
    }

    cluster::destroy();
    return 0;
}
