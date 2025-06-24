#include "BitAttention.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

BitAttention::BitAttention(int dim, float threshold)
    : dim(dim),
      W_q(dim, dim, threshold),
      W_k(dim, dim, threshold),
      W_v(dim, dim, threshold),
      W_o(dim, dim, threshold) {}

vector<vector<float>> BitAttention::forward(const vector<vector<float>>& X) {
    int n_tokens = X.size();

    // Calcular Q, K, V
    vector<vector<float>> Q(n_tokens), K(n_tokens), V(n_tokens);
    for (int i = 0; i < n_tokens; ++i) {
        Q[i] = W_q.forward(X[i]);
        K[i] = W_k.forward(X[i]);
        V[i] = W_v.forward(X[i]);
    }

    // Calcular la atención (Q × Kᵗ / sqrt(d))
    vector<vector<float>> attention_scores(n_tokens, vector<float>(n_tokens));
    for (int i = 0; i < n_tokens; ++i) {
        for (int j = 0; j < n_tokens; ++j) {
            float dot = 0.0f;
            for (int k = 0; k < dim; ++k)
                dot += Q[i][k] * K[j][k];
            attention_scores[i][j] = dot / sqrt((float)dim);
        }
    }

    // Aplicar softmax por fila
    vector<vector<float>> A(n_tokens);
    for (int i = 0; i < n_tokens; ++i)
        A[i] = softmax(attention_scores[i]);

    // Atención final: A × V
    vector<vector<float>> output(n_tokens, vector<float>(dim, 0.0f));
    for (int i = 0; i < n_tokens; ++i) {
        for (int j = 0; j < n_tokens; ++j) {
            for (int k = 0; k < dim; ++k)
                output[i][k] += A[i][j] * V[j][k];
        }
    }

    // Proyección de salida
    for (int i = 0; i < n_tokens; ++i)
        output[i] = W_o.forward(output[i]);

    return output;
}

vector<float> BitAttention::softmax(const vector<float>& input) const {
    vector<float> output(input.size());
    float max_val = *max_element(input.begin(), input.end());

    float sum = 0.0f;
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = exp(input[i] - max_val);
        sum += output[i];
    }

    for (float& val : output)
        val /= sum;

    return output;
}
