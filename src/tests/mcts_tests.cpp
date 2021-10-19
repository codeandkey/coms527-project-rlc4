#include "../mcts.h"
#include "../environment_c4.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

using namespace std;

TEST(MCTSTest, TreeCanInit) {
    Environment* env = new C4Environment();
    Tree t(env);
    delete env;
}

TEST(MCTSTest, TreeCanSimulate) {
    Environment* env = new C4Environment();
    Tree* t = new Tree(env);

    EXPECT_NE(t->simulate(), nullptr);
    EXPECT_THROW(t->simulate(), std::runtime_error);

    delete t;
    delete env;
}