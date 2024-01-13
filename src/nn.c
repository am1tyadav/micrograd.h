#include <stdio.h>
#include <time.h>

#include "arena.h"
#include "micrograd.h"

float compute_y(float x1, float x2, float x3) {
    float noise = float_create_random() / (float) 10;
    float true_w1 = 3;
    float true_w2 = -1;
    float true_w3 = 5;
    float true_b = -2;

    return true_w1 * x1 + true_w2 * x2 + true_w3 * x3 + true_b + noise;
}

int main(void) {
    srand(time(NULL));

    Arena *arena = arena_create(8192);

    Value **inputs = inputs_create(arena, 3);
    Value *y = value_create_constant(arena, 0);

    NetworkConfig config = {
        .num_inputs = 3,
        .num_layers = 3,
        .num_neurons = (size_t[]) { 3, 3, 1 },
        .hidden_activation = ACT_RELU,
        .output_activation = ACT_LINEAR
    };

    Value **outputs = network_create(arena, inputs, config);
    Value *y_pred = outputs[0];
    Value *loss = loss_mean_squared_error(arena, y, y_pred);

    Graph *graph = graph_create(arena, loss, 1000);

    size_t num_iterations = 5000;
    size_t log_interval = 200;
    float learning_rate = 0.3;

    for (size_t i = 0; i < num_iterations; i++) {
        inputs[0]->data = float_create_random();
        inputs[1]->data = float_create_random();
        inputs[2]->data = float_create_random();

        y->data = compute_y(inputs[0]->data, inputs[1]->data, inputs[2]->data);

        graph_optimisation_step(graph, learning_rate);

        if ((i + 1) % log_interval == 0) {
            printf("Iter: %5zu, Loss: %f\n", i, graph->values[0]->data);
        }
    }

    arena_destroy(arena);
    return 0;
}
