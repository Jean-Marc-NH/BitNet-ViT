#include "BitVisionTransformer.hpp"

BitVisionTransformer::BitVisionTransformer(int patch_size, int d_model, int num_classes, int num_layers, float threshold)
    : d_model(d_model),
      num_layers(num_layers),
      patch_embedder(patch_size, d_model),
      pos_embedding(64, d_model, threshold),
      classifier(d_model, num_classes, threshold) {
    
    for (int i = 0; i < num_layers; ++i)
        encoders.emplace_back(BitTransformerEncoderLayer(d_model, d_model * 2, threshold));
}

int BitVisionTransformer::predict(const float* image_data) {
    // Paso 1: patching y embedding
    vector<vector<float>> tokens = patch_embedder.process(image_data);

    // Paso 2: agregar positional embedding
    pos_embedding.apply(tokens);

    // Paso 3: encoder layers
    for (int i = 0; i < num_layers; ++i)
        tokens = encoders[i].forward(tokens);

    // Paso 4: Global Average Pooling
    vector<float> pooled = pooling.forward(tokens);

    // Paso 5: Clasificación
    vector<float> logits = classifier.forward(pooled);

    // Paso 6: Predicción final
    return argmax(logits);
}
