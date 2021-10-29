#include "mcts.h"
#include "params.h"

#include <random>
#include <stdexcept>

using namespace std;

void Node::backprop(float v, int depth) {
    w += v;
    n += 1;

    if (depth > this->depth) {
        this->depth = depth;
    }

    if (parent) {
        parent->backprop(-v, depth + 1);
    }
}

float Node::valueAverage() {
    if (!n) {
        return 0.0f;
    } else {
        return (w / (float) n);
    }
}

void Node::expand(vector<int> actions, float* policy, float value) {
    children.reserve(actions.size());

    for (auto& a : actions) {
        shared_ptr<Node> child = make_shared<Node>();

        child->parent = this;
        child->action = a;
        child->p = policy[a];

        children.push_back(child);
    }

    backprop(value);
}

Tree::Tree() {
    env = make_shared<SELECTED_ENV>();
    root = make_shared<Node>();
}

shared_ptr<Environment> Tree::simulate() {
    if (target) {
        throw std::runtime_error("Tree::select() called in waiting mode");
    }

    if (root->n >= MCTS_TARGET_NODES) {
        return nullptr;
    }

    shared_ptr<Node> current = root;

    while (!current->children.empty()) {
        // Maximize PUCT

        float max_puct = std::numeric_limits<float>::min();
        shared_ptr<Node> selecting;

        for (auto& c : current->children) {
            float val = -c->valueAverage();
            float exploration = sqrtf((float) current->n) / (1.0f + (float) c->n);

            float puct = val + MCTS_PUCT * c->p * exploration;

            if (puct > max_puct) {
                selecting = c;
                max_puct = puct;
            }
        }

        current = selecting;
        env->push(selecting->action);
    }

    // Are we at a terminal? If so, unwind now.
    float tval;
    if (env->terminal(&tval)) {
        current->backprop(tval);

        while(current != root) {
            env->pop();
            current = current->parent;
        }

        return simulate();
    }

    // No terminal, we don't unwind and instead enter waiting state.
    target = current;

    return env;
}

vector<float> Tree::getProbabilities() {
    vector<float> out(PSIZE, 0.0f);

    // Compute pvec for actions
    for (auto& c : root->children) {
        out[c->action] = (float) c->n / fmax(1, root->n);
    }

    return out;
}

float Tree::valueAverage() {
    return root->valueAverage();
}

int Tree::chooseAction() {
    vector<float> pvec = getProbabilities();

    if (!root->n) {
        throw std::runtime_error("chooseAction() called with root n=0");
    }

    std::random_device device;
    std::mt19937 engine(device());
    std::discrete_distribution<> dist(pvec.begin(), pvec.end());

    return dist(engine);
}

void Tree::expand(float* policy, float value) {
    // Get list of actions

    target->expand(env->getLegalActions(), policy, value);
    target = nullptr;

    // Unwind actions
    while (target != root) {
        env->pop();
        target = target->parent;
    }
}

void Tree::advance(int action) {
    Node* nextroot = nullptr;

    if (MCTS_PRESERVE_SUBTREE) {
        for (auto& c : root->children) {
            if (c->action == action) {
                nextroot = c;
                break;
            }
        }
    } else {
        nextroot = new Node();
    }

    if (!nextroot) {
        throw std::runtime_error("advance(): action not found in children");
    }

    delete root;
    root = nextroot;
    root->parent = nullptr;
}
