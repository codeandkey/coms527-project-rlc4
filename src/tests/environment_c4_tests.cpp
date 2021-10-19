#include "../environment_c4.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

using namespace std;

TEST(C4EnvironmentTest, Dimensions) {
    Environment* env = new C4Environment();

    int w, h, f;
    env->getDimensions(&w, &h, &f);

    EXPECT_EQ(w, 7);
    EXPECT_EQ(h, 6);
    EXPECT_EQ(f, 2);
}

TEST(C4EnvironmentTest, Name) {
    Environment* env = new C4Environment();

    EXPECT_EQ(string(env->getName()), string("Connect4"));
}

TEST(C4EnvironmentTest, GetString) {
    Environment* env = new C4Environment();

    EXPECT_EQ(string(env->getString()), string("X to move\n.......\n.......\n.......\n.......\n.......\n.......\n"));
}

TEST(C4EnvironmentTest, InitialLegalMask) {
    constexpr int len = 7;

    Environment* env = new C4Environment();
    vector<float> expected(len, 1), actual(len, 0);

    ASSERT_EQ(len, env->policySize());
    env->legalMask(&actual[0]);

    for (int i = 0; i < 7; ++i) {
        EXPECT_EQ(expected[i], actual[i]) << "expected, result differ at index " << i;
    }
}

TEST(C4EnvironmentTest, InitialGetLegalMoves) {
    Environment* env = new C4Environment();
    vector<int> expected = {0, 1, 2, 3, 4, 5, 6};
    vector<int> actual = env->getLegalActions();

    for (int i = 0; i < 7; ++i) {
        EXPECT_EQ(expected[i], actual[i]) << "expected, result differ at index " << i;
    }
}