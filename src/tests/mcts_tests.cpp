#include "../mcts.h"
#include "../connect4.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

using namespace std;

TEST(MCTSTest, TreeCanInit) {
    Tree t;
}

TEST(MCTSTest, TreeCanSimulate) {
    Tree t;

    EXPECT_NE(t.simulate(), nullptr);
    EXPECT_THROW(t.simulate(), std::runtime_error);
}
