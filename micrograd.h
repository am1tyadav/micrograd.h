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
    Value   **values;
    size_t  num_values;
} Graph;

typedef struct {
    size_t  num_inputs;
    size_t  num_hidden_layers;
    size_t  *num_neurons;
    bool    use_activation;
    // bool    use_output_activation;
} NetworkConfig;

// Header

Value *value_create_constant(Arena *arena, float data);
Value *value_create_random(Arena *arena);

void op_add_forward(Value *self);
void op_add_backward(Value *self);
void op_mul_forward(Value *self);
void op_mul_backward(Value *self);
void op_relu_forward(Value *self);
void op_relu_backward(Value *self);

Value *op_add(Arena *arena, Value *a, Value *b);
Value *op_mul(Arena *arena, Value *a, Value *b);
Value *op_relu(Arena *arena, Value *a);

void _graph_create(Arena *arena, Value *root, Value **visited, size_t *count);
Graph *graph_create(Arena *arena, Value *root, size_t max_values);
void graph_forward(Graph *graph);
void graph_backward(Graph *graph);
void graph_update(Graph *graph, float learning_rate);
void graph_zero_grad(Graph *graph);
void graph_optimisation_step(Graph *graph, float learning_rate);

Value **inputs_create(Arena *arena, float *data, size_t num_inputs);
Value *neuron_create(Arena *arena, Value **inputs, size_t num_inputs, bool use_activation);
Value **layer_create(Arena *arena, Value **inputs, size_t num_inputs, size_t num_neurons, bool use_activation);
Value *network_create(Arena *arena, Value **inputs, NetworkConfig config);

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

void op_relu_forward(Value *self) {
    self->data = self->children[0]->data > 0 ? self->children[0]->data : 0;
}

void op_relu_backward(Value *self) {
    self->children[0]->grad += self->data > 0 ? self->grad : 0;
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

Value *op_relu(Arena *arena, Value *a) {
    Value *value = (Value *) arena_allocate(arena, sizeof(Value));
    Value **children = (Value **) arena_allocate(arena, sizeof(Value *) * 1);

    children[0] = a;

    *value = (Value) {
        .repr = 'r',
        .num_children = 1,
        .children = children,
        .forward = op_relu_forward,
        .backward = op_relu_backward
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

Value **inputs_create(Arena *arena, float *data, size_t num_inputs) {
    Value **inputs = (Value **) arena_allocate(arena, sizeof(Value *) * num_inputs);

    for (size_t i = 0; i < num_inputs; i++) {
        inputs[i] = value_create_constant(arena, data[i]);
    }

    return inputs;
}

Value *neuron_create(Arena *arena, Value **inputs, size_t num_inputs, bool use_activation) {
    Value *bias = value_create_random(arena);
    bias->repr = 'b';

    for (size_t i = 0; i < num_inputs; i++) {
        Value *weight = value_create_random(arena);
        weight->repr = 'w';
        bias = op_add(arena, bias, op_mul(arena, weight, inputs[i]));
    }

    if (use_activation) {
        bias = op_relu(arena, bias);
    }

    return bias;
}

Value **layer_create(Arena *arena, Value **inputs, size_t num_inputs, size_t num_neurons, bool use_activation) {
    Value **neurons = (Value **) arena_allocate(arena, sizeof(Value *) * num_neurons);

    for (size_t i = 0; i < num_neurons; i++) {
        neurons[i] = neuron_create(arena, inputs, num_inputs, use_activation);
    }

    return neurons;
}

Value *network_create(Arena *arena, Value **inputs, NetworkConfig config) {
    Value **outputs = inputs;
    size_t num_inputs = config.num_inputs;

    for (size_t i = 0; i < config.num_hidden_layers; i++) {
        outputs = layer_create(arena, outputs, num_inputs, config.num_neurons[i], config.use_activation);
        num_inputs = config.num_neurons[i];
    }

    // Output layer
    outputs = layer_create(arena, outputs, num_inputs, 1, false);

    return outputs[0];
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
