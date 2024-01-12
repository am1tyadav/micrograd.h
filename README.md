# micrograd.h

Similar to Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) but in C

## Requirements

- C compiler
- [Task](https://taskfile.dev) or user-defined build system

## Examples

__Linear Regression:__ `task linreg`
__Neural Network:__ `task nn`

```c
// Provide network config
NetworkConfig config = {
    .num_inputs = 3,
    .num_hidden_layers = 2,
    .num_neurons = (size_t[]) { 2, 2 },
    .use_activation = true
};

// Get network output
Value *y_pred = network_create(arena, inputs, config);

// Create computation graph
Graph *graph = graph_create(arena, loss, MAX_VALUES);

// Perform optimisation
graph_optimisation_step(graph, learning_rate);
```
