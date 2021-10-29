#include "../connect4.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

using namespace std;

TEST(Connect4Test, Dimensions) {
    EXPECT_EQ(Connect4::width, 7);
    EXPECT_EQ(Connect4::height, 6);
    EXPECT_EQ(Connect4::features, 2);
    EXPECT_EQ(Connect4::policy_size, 7);
}

TEST(Connect4Test, Name) {
    Environment* env = new Connect4();

    EXPECT_EQ(string(env->getName()), string("Connect4"));
}

TEST(Connect4Test, GetString) {
    Environment* env = new Connect4();

    EXPECT_EQ(string(env->getString()), string("X to move\n.......\n.......\n.......\n.......\n.......\n.......\n"));
}

TEST(Connect4Test, InitialLegalMask) {
    constexpr int len = 7;

    Environment* env = new Connect4();
    vector<float> expected(len, 1), actual(len, 0);

    ASSERT_EQ(len, Connect4::policy_size);
    env->legalMask(&actual[0]);

    for (int i = 0; i < 7; ++i) {
        EXPECT_EQ(expected[i], actual[i]) << "expected, result differ at index " << i;
    }
}

TEST(Connect4Test, InitialGetLegalMoves) {
    Environment* env = new Connect4();
    vector<int> expected = {0, 1, 2, 3, 4, 5, 6};
    vector<int> actual = env->getLegalActions();

    for (int i = 0; i < 7; ++i) {
        EXPECT_EQ(expected[i], actual[i]) << "expected, result differ at index " << i;
    }
}
