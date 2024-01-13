#include <stdio.h>
#include <time.h>

#include "arena.h"
#include "micrograd.h"

float compute_y(float x1, float x2) {
    float noise = float_create_random() / (float) 10;
    float true_w1 = 3;
    float true_w2 = -1;
    float true_b = -2;

    return true_w1 * x1 + true_w2 * x2 + true_b + noise;
}

int main(void) {
    srand(time(NULL));

    Arena *arena = arena_create(2048);

    Value *x1 = value_create_constant(arena, 0);
    Value *x2 = value_create_constant(arena, 0);
    Value *y = value_create_constant(arena, 0);
    Value *minus_one = value_create_constant(arena, -1);

    Value *w1 = value_create_random(arena);
    Value *w2 = value_create_random(arena);
    Value *b = value_create_random(arena);

    Value *y_pred = op_add(arena, op_add(arena, op_mul(arena, w1, x1), op_mul(arena, w2, x2)), b);
    Value *diff = op_add(arena, op_mul(arena, minus_one, y), y_pred);
    Value *loss = op_mul(arena, diff, diff);

    w1->repr = 'w';
    w2->repr = 'w';
    b->repr = 'b';
    loss->repr = 'l';

    Graph *graph = graph_create(arena, loss, 20);

    size_t num_iterations = 10000;
    float learning_rate = 0.003;

    for (size_t i = 0; i < num_iterations; i++) {
        x1->data = float_create_random();
        x2->data = float_create_random();
        y->data = compute_y(x1->data, x2->data);

        graph_optimisation_step(graph, learning_rate);
    }

    graph_print(graph);

    printf("Learned w1 or w2: %f, True w1: %f\n", w1->data, 3.f);
    printf("Learned w1 or w2: %f, True w2: %f\n", w2->data, -1.f);
    printf("Learned b: %f, True b: %f\n", b->data, -2.f);

    return 0;
}
