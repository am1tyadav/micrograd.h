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

    float input_data[3] = { };
    Value **inputs = inputs_create(arena, input_data, 3);
    Value *y = value_create_constant(arena, 0);
    Value *minus_one = value_create_constant(arena, -1);

    NetworkConfig config = {
        .num_inputs = 3,
        .num_hidden_layers = 2,
        .num_neurons = (size_t[]) { 2, 2 },
        .use_activation = true
    };

    Value *y_pred = network_create(arena, inputs, config);
    Value *diff = op_add(arena, op_mul(arena, minus_one, y), y_pred);
    Value *loss = op_mul(arena, diff, diff);

    Graph *graph = graph_create(arena, loss, 1000);

    size_t num_iterations = 2000;
    size_t log_interval = 200;
    float learning_rate = 0.003;

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

    return 0;
}
