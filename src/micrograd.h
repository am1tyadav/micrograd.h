#ifndef MICROGRAD_H
#define MICROGRAD_H

#include <stdio.h>
#include <stdbool.h>
#include <math.h>

#include "arena.h"

#define EPSILON     0.01

typedef enum {
    ACT_LINEAR,
    ACT_RELU,
    ACT_SIGMOID,
    ACT_SOFTMAX
} ACTIVATION;

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
    size_t      num_inputs;
    size_t      num_layers;
    size_t      *num_neurons;
    ACTIVATION  hidden_activation;
    ACTIVATION  output_activation;
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
void op_sigmoid_forward(Value *self);
void op_sigmoid_backward(Value *self);
void op_clip_forward(Value *self);
void op_clip_backward(Value *self);

Value *op_add(Arena *arena, Value *a, Value *b);
Value *op_mul(Arena *arena, Value *a, Value *b);
Value *op_relu(Arena *arena, Value *a);
Value *op_sigmoid(Arena *arena, Value *a);
Value *op_negate(Arena *arena, Value *a);
Value *op_clip(Arena *arena, Value *a);

Value *loss_mean_squared_error(Arena *arena, Value *y_true, Value *y_pred);

void _graph_create(Arena *arena, Value *root, Value **visited, size_t *count);
Graph *graph_create(Arena *arena, Value *root, size_t max_values);
void graph_forward(Graph *graph);
void graph_backward(Graph *graph);
void graph_update(Graph *graph, float learning_rate);
void graph_zero_grad(Graph *graph);
void graph_optimisation_step(Graph *graph, float learning_rate);

Value **inputs_create(Arena *arena, size_t num_inputs);
Value *neuron_create(Arena *arena, Value **inputs, size_t num_inputs, ACTIVATION activation);
Value **layer_create(Arena *arena, Value **inputs, size_t num_inputs, size_t num_neurons, ACTIVATION activation);
Value **network_create(Arena *arena, Value **inputs, NetworkConfig config);

void value_print(Value *value);
void graph_print(Graph *graph);
float float_create_random(void);
float float_sigmoid(float x);

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

void op_sigmoid_forward(Value *self) {
    self->data = float_sigmoid(self->children[0]->data);
}

void op_sigmoid_backward(Value *self) {
    float x = self->children[0]->data;

    self->children[0]->grad += (self->grad * float_sigmoid(x) / (1.0f - float_sigmoid(x) + EPSILON));
}

void op_clip_forward(Value *self) {
    if (self->data > (1 - EPSILON)) self->data = 1 - EPSILON;
    else if (self->data < EPSILON) self->data = EPSILON;
}

void op_clip_backward(Value *self) {
    self->children[0]->grad += self->grad;
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

Value *op_sigmoid(Arena *arena, Value *a) {
    Value *value = (Value *) arena_allocate(arena, sizeof(Value));
    Value **children = (Value **) arena_allocate(arena, sizeof(Value *) * 1);

    children[0] = a;

    *value = (Value) {
        .repr = 's',
        .num_children = 1,
        .children = children,
        .forward = op_sigmoid_forward,
        .backward = op_sigmoid_backward
    };

    return value;
}

Value *op_negate(Arena *arena, Value *a) {
    Value *minus_one = value_create_constant(arena, -1);

    return op_mul(arena, a, minus_one);
}

Value *op_clip(Arena *arena, Value *a) {
    Value *value = (Value *) arena_allocate(arena, sizeof(Value));
    Value **children = (Value **) arena_allocate(arena, sizeof(Value *) * 1);

    children[0] = a;

    *value = (Value) {
        .repr = 'c',
        .num_children = 1,
        .children = children,
        .forward = op_clip_forward,
        .backward = op_clip_backward
    };

    return value;
}

Value *loss_mean_squared_error(Arena *arena, Value *y_true, Value *y_pred) {
    Value *half = value_create_constant(arena, 0.5);

    Value *diff = op_add(arena, y_pred, op_negate(arena, y_true));
    Value *loss = op_mul(arena, op_mul(arena, diff, diff), half);

    loss->repr = 'L';
    loss->not_trainable = true;

    return loss;
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
            value->data -= value->grad * learning_rate;
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

Value **inputs_create(Arena *arena, size_t num_inputs) {
    Value **inputs = (Value **) arena_allocate(arena, sizeof(Value *) * num_inputs);

    for (size_t i = 0; i < num_inputs; i++) {
        inputs[i] = value_create_constant(arena, 0);
    }

    return inputs;
}

Value *neuron_create(Arena *arena, Value **inputs, size_t num_inputs, ACTIVATION activation) {
    Value *bias = value_create_random(arena);
    bias->repr = 'b';

    for (size_t i = 0; i < num_inputs; i++) {
        Value *weight = value_create_random(arena);
        weight->repr = 'w';
        bias = op_add(arena, bias, op_mul(arena, weight, inputs[i]));
    }

    if (activation == ACT_RELU) {
        bias = op_relu(arena, bias);
    }
    else if (activation == ACT_SIGMOID) {
        bias = op_sigmoid(arena, bias);
    }

    return bias;
}

Value **layer_create(Arena *arena, Value **inputs, size_t num_inputs, size_t num_neurons, ACTIVATION activation) {
    Value **neurons = (Value **) arena_allocate(arena, sizeof(Value *) * num_neurons);

    for (size_t i = 0; i < num_neurons; i++) {
        neurons[i] = neuron_create(arena, inputs, num_inputs, activation);
    }

    return neurons;
}

Value **network_create(Arena *arena, Value **inputs, NetworkConfig config) {
    Value **outputs = inputs;
    size_t num_inputs = config.num_inputs;

    for (size_t i = 0; i < config.num_layers; i++) {
        bool is_output_layer = i == config.num_layers - 1;
        ACTIVATION activation = is_output_layer ? config.output_activation : config.hidden_activation;

        printf("Creating layer with %zu inputs and %zu outputs\n", num_inputs, config.num_neurons[i]);

        outputs = layer_create(arena, outputs, num_inputs, config.num_neurons[i], activation);
        num_inputs = config.num_neurons[i];
    }

    return outputs;
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
    return (float) rand() * 0.2f / (float) RAND_MAX;
}

float float_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-1.0f * x));
}

#endif // MICROGRAD_H
