#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "PatchEmbedder.hpp"
#include "PositionalEmbedding.hpp"
#include "BitLinear.hpp"



using namespace std;


bool load_image_from_bin(const string& path, int index, float* image, int image_size = 48) {
    const int pixels = image_size * image_size;
    ifstream in(path, ios::binary);
    if (!in) {
        cerr << "Error al abrir: " << path << "\n";
        return false;
    }

    in.seekg(index * pixels * sizeof(float), ios::beg);
    in.read(reinterpret_cast<char*>(image), pixels * sizeof(float));

    if (!in) {
        cerr << "Error al leer la imagen #" << index << " desde " << path << "\n";
        return false;
    }

    return true;
}

int load_label_from_bin(const string& path, int index) {
    ifstream in(path, ios::binary);
    if (!in) {
        cerr << "Error al abrir: " << path << "\n";
        return -1;
    }
    in.seekg(index * sizeof(uint8_t), ios::beg);
    uint8_t label;
    in.read(reinterpret_cast<char*>(&label), sizeof(uint8_t));
    return static_cast<int>(label);
}

int main() {
    const string x_path = "prePros/X_train.bin";
    const string y_path = "prePros/Y_train.bin";

    float image[48 * 48];
    int index = 0;

    if (!load_image_from_bin(x_path, index, image)) return 1;
    int label = load_label_from_bin(y_path, index);

    PatchEmbedder patcher(6, 64);
    auto tokens = patcher.process(image);

    cout << "Etiqueta: " << label << "\n";
    cout << "Primer token (embedding del primer parche):\n";
    for (float val : tokens[0]) {
        cout << val << " ";
    }
    cout << "\n";

    PositionalEmbedding pos_embed(64, 64);
    pos_embed.apply(tokens);

    cout << "Positional Embedding: \n" << endl; 
    cout << "Etiqueta: " << label << "\n";
    cout << "Primer token (embedding + posicion del primer parche):\n";
    for (float val : tokens[0]) {
        cout << val << " ";
    }
    cout << "\n";

    vector<float> token = tokens[0];

    BitLinear linear(64, 64);  // Capa binarizada 64 → 64
    vector<float> output = linear.forward(token);

    cout << "Output BitLinear:\n";
    for (float val : output) cout << val << " ";
    cout << "\n";

    return 0;
}
