#include "environment.h"

#include <string>
#include <vector>

class C4Environment : public Environment {
    public:
        C4Environment();

        // Inherited methods, implemented in environment_c4.cpp
        const char* getName();
        std::string getString();
        int policySize();
        void getDimensions(int* width, int* height, int* features);
        void legalMask(int* dst);
        void push(int ind);
        void pop();
        bool terminal(float* value);
        void input(float* layer);

    private:
        int cells[42], turn;
        std::vector<int> history;
};