#include "environment.h"

#include <string>
#include <vector>

class Connect4 : public Environment {
    public:
        /**
         * Initializes a new connect-4 environment.
         */
        Connect4();

        // Inherited methods, implemented in environment_c4.cpp
        const char* getName();
        std::string getString();
        int policySize();
        void getDimensions(int* width, int* height, int* features);
        void legalMask(float* dst);
        void push(int ind);
        void pop();
        bool terminal(float* value);
        void input(float* layer);

        static const int width = 7;
        static const int height = 6;
        static const int policy_size = 7;
        static const int features = 2;

    private:
        int cells[42], turn;
        std::vector<int> history;
};
