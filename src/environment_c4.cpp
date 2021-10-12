#include "environment_c4.h"

using namespace std;

C4Environment::C4Environment() {
    turn = 1;

    for(int i = 0; i < sizeof cells / sizeof cells[0]; ++i) {
        cells[i] = 0;
    }
}

const char* C4Environment::getName() {
    return "Connect4";
}

string C4Environment::getString() {
    string output = "O.X"[turn + 1] + string(" to move\n");

    for (int y = 5; y >= 0; --y) {
        for (int x = 0; x < 7; ++x) {
            int v = cells[y * 7 + x];
            output += "O.X"[v+1];
        }

        output += "\n";
    }

    return output;
}

int C4Environment::policySize() {
    return 7;
}

void C4Environment::getDimensions(int* width, int* height, int* features) {
    *width = 7;
    *height = 6;
    *features = 2;
}

void C4Environment::legalMask(float* dst) {
    for (int x = 0; x < 7; ++x) {
        dst[x] = (cells[7 * 5 + x] == 0) ? 1.0f : 0.0f;
    }
}

void C4Environment::push(int ind) {
    for (int y = 0; y < 6; ++y) {
        int dst = y * 7 + ind;
        if (!cells[dst]) {
            cells[dst] = turn;
            turn = -turn;
            history.push_back(ind);
            return;
        }
    }
}

void C4Environment::pop() {
    int last = history.back();
    history.pop_back();

    for (int y = 5; y >= 0; --y) {
        int dst = y * 7 + last;

        if (cells[dst]) {
            cells[dst] = 0;
            return;
        }
    }
}

bool C4Environment::terminal(float* value) {
    // Check to see if the game is over. If there is a C-4 on the board then
    // the player to move must have lost.

    static auto check_safe = [&](int x, int y, int* cells) -> bool {
        if (x < 0 || x >= 7) return false;
        if (y < 0 || y >= 6) return false;
        if (cells[y * 7 + x] != -turn) return false;

        return true;
    };

    // Check horizontal matches

    for (int x = 0; x < 4; ++x) {
        for (int y = 0; y < 6; ++y) {
            bool ok = true;

            for (int dx = 0; dx < 4; ++dx) {
                ok &= check_safe(x + dx, y, cells);
            }

            if (ok) {
                *value = -1.0f;
                return true;
            }
        }
    }

    // Check vertical matches

    for (int x = 0; x < 7; ++x) {
        for (int y = 0; y < 3; ++y) {
            bool ok = true;

            for (int dy = 0; dy < 4; ++dy) {
                ok &= check_safe(x, y + dy, cells);
            }

            if (ok) {
                *value = -1.0f;
                return true;
            }
        }
    }

    // Check NE diagonal matches

    for (int x = 0; x < 4; ++x) {
        for (int y = 0; y < 3; ++y) {
            bool ok = true;

            for (int dxy = 0; dxy < 4; ++dxy) {
                ok &= check_safe(x + dxy, y + dxy, cells);
            }

            if (ok) {
                *value = -1.0f;
                return true;
            }
        }
    }

    // Check NW diagonal matches

    for (int x = 3; x < 7; ++x) {
        for (int y = 0; y < 3; ++y) {
            bool ok = true;

            for (int dxy = 0; dxy < 4; ++dxy) {
                ok &= check_safe(x - dxy, y + dxy, cells);
            }

            if (ok) {
                *value = -1.0f;
                return true;
            }
        }
    }

    // Check for draw

    bool draw = true;

    for (int i = 0; i < sizeof cells / sizeof cells[0]; ++i) {
        if (!cells[i]) {
            draw = false;
            break;
        }
    }

    if (draw) {
        *value = 0.0f;
    }

    return draw;
}

void C4Environment::input(float* layer) {
    for (int x = 0; x < 7; ++x) {
        for (int y = 0; y < 6; ++y) {
            int val = cells[y * 7 + x];
            float* dst = &layer[y * (7 * 2) + x * 2];

            switch (val * turn) {
                case 1:
                    dst[0] = 1.0f;
                    dst[1] = 0.0f;
                    break;
                case -1:
                    dst[0] = 0.0f;
                    dst[1] = 1.0f;
                    break;
                default:
                    dst[0] = 0.0f;
                    dst[1] = 0.0f;
            }
        }
    }
}