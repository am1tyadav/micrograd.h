#ifndef MICROGRAD_H
#define MICROGRAD_H

#include <stdio.h>
#include <stdbool.h>

#include "arena.h"

typedef struct Value Value;

struct Value {
    char    repr;

    float   data;
    float   grad;
    size_t  num_children;
    Value   **children;
    bool    not_trainable;

    void    (*forward)  (Value *self);
    void    (*backward) (Value *self);
};

typedef struct {
    Value **values;
    size_t num_values;
} Graph;

// Header

Value *value_create_constant(Arena *arena, float data);
Value *value_create_random(Arena *arena);

void op_add_forward(Value *self);
void op_add_backward(Value *self);
void op_mul_forward(Value *self);
void op_mul_backward(Value *self);

Value *op_add(Arena *arena, Value *a, Value *b);
Value *op_mul(Arena *arena, Value *a, Value *b);

void _graph_create(Arena *arena, Value *root, Value **visited, size_t *count);
Graph *graph_create(Arena *arena, Value *root, size_t max_values);
void graph_forward(Graph *graph);
void graph_backward(Graph *graph);
void graph_update(Graph *graph, float learning_rate);
void graph_zero_grad(Graph *graph);
void graph_optimisation_step(Graph *graph, float learning_rate);

void value_print(Value *value);
void graph_print(Graph *graph);
float float_create_random(void);

// Implementation

Value *value_create_constant(Arena *arena, float data) {
    Value *value = (Value *) arena_allocate(arena, sizeof(Value));

    *value = (Value) {
        .repr = 'v',
        .data = data,
        .not_trainable = true
    };

    return value;
}

Value *value_create_random(Arena *arena) {
    Value *value = (Value *) arena_allocate(arena, sizeof(Value));

    *value = (Value) {
        .repr = 'v',
        .data = float_create_random()
    };

    return value;
}

void op_add_forward(Value *self) {
    self->data = self->children[0]->data + self->children[1]->data;
}

void op_add_backward(Value *self) {
    self->children[0]->grad += self->grad;
    self->children[1]->grad += self->grad;
}

void op_mul_forward(Value *self) {
    self->data = self->children[0]->data * self->children[1]->data;
}

void op_mul_backward(Value *self) {
    self->children[0]->grad += self->children[1]->data * self->grad;
    self->children[1]->grad += self->children[0]->data * self->grad;
}

Value *op_add(Arena *arena, Value *a, Value *b) {
    Value *value = (Value *) arena_allocate(arena, sizeof(Value));
    Value **children = (Value **) arena_allocate(arena, sizeof(Value *) * 2);

    children[0] = a;
    children[1] = b;

    *value = (Value) {
        .repr = '+',
        .num_children = 2,
        .children = children,
        .forward = op_add_forward,
        .backward = op_add_backward
    };

    return value;
}

Value *op_mul(Arena *arena, Value *a, Value *b) {
    Value *value = (Value *) arena_allocate(arena, sizeof(Value));
    Value **children = (Value **) arena_allocate(arena, sizeof(Value *) * 2);

    children[0] = a;
    children[1] = b;

    *value = (Value) {
        .repr = '*',
        .num_children = 2,
        .children = children,
        .forward = op_mul_forward,
        .backward = op_mul_backward
    };

    return value;
}

void _graph_create(Arena *arena, Value *root, Value **visited, size_t *count) {
    if (!root) return;

    for (size_t i = 0; i < *count; i++) {
        if (root == visited[i]) return;
    }

    visited[*count] = root;
    *count += 1;

    for (size_t i = 0; i < root->num_children; i++) {
        _graph_create(arena, root->children[i], visited, count);
    }
}

Graph *graph_create(Arena *arena, Value *root, size_t max_values) {
    Value **visited = (Value **) calloc(max_values, sizeof(Value *));
    size_t count = 0;

    _graph_create(arena, root, visited, &count);

    Graph *value_graph = (Graph *) arena_allocate(arena, sizeof(Graph));
    Value **values = (Value **) arena_allocate(arena, sizeof(Value *) * count);

    for (size_t i = 0; i < count; i++) {
        values[i] = visited[i];
    }

    *value_graph = (Graph) {
        .values = values,
        .num_values = count
    };

    free(visited);
    visited = NULL;

    return value_graph;
}

void graph_forward(Graph *graph) {
    for (size_t i = graph->num_values; i > 0; i--) {
        if (graph->values[i - 1]->forward) {
            graph->values[i - 1]->forward(graph->values[i - 1]);
        }
    }
}

void graph_backward(Graph *graph) {
    graph->values[0]->grad = 1;

    for (size_t i = 0; i < graph->num_values; i++) {
        if (graph->values[i]->backward) {
            graph->values[i]->backward(graph->values[i]);
        }
    }
}

void graph_update(Graph *graph, float learning_rate) {
    for (size_t i = 0; i < graph->num_values; i++) {
        Value *value = graph->values[i];

        if (!value->not_trainable) {
            value->data -= learning_rate * value->grad;
        }
    }
}

void graph_zero_grad(Graph *graph) {
    for (size_t i = 0; i < graph->num_values; i++) {
        graph->values[i]->grad = 0;
    }
}

void graph_optimisation_step(Graph *graph, float learning_rate) {
    assert(graph->num_values > 0);

    graph_zero_grad(graph);
    graph_forward(graph);
    graph_backward(graph);
    graph_update(graph, learning_rate);
}

void value_print(Value *value) {
    printf("%c(data=%f, grad=%f, trainable=%s)\n", value->repr, value->data, value->grad, value->not_trainable ? "false" : "true");
}

void graph_print(Graph *graph) {
    printf("===== Graph(%lu values) =====\n", graph->num_values);

    for (size_t i = 0; i < graph->num_values; i++) {
        value_print(graph->values[i]);
    }

    printf("===========================\n");
}

float float_create_random(void) {
    return (float) rand() / (float) RAND_MAX;
}

#endif // MICROGRAD_H