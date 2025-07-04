#include "BitMLP.hpp"

BitMLP::BitMLP(int dim, int hidden_dim, float threshold)
    : fc1(dim, hidden_dim, threshold), fc2(hidden_dim, dim, threshold) {}

float BitMLP::relu(float x) const {
    return x > 0 ? x : 0;
}

vector<float> BitMLP::forward(const vector<float>& x) const {
    vector<float> hidden = fc1.forward(x);
    for (float& v : hidden)
        v = relu(v);
    return fc2.forward(hidden);
}
