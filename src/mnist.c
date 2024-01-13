#include <stdio.h>
#include <time.h>

#include "arena.h"
#include "micrograd.h"
#include "mnist.h"
#include "raylib.h"

#define WINDOW_W    896
#define WINDOW_H    448
#define TARGET_FPS  60
#define PIXEL_SIZE  16

uint8_t inference_image[IMAGE_HEIGHT][IMAGE_WIDTH];

void initialise_image() {
    for (uint8_t i = 0; i < IMAGE_HEIGHT; i++) {
        for (uint8_t j = 0; j < IMAGE_WIDTH; j++) {
            inference_image[i][j] = 0;
        }
    }
}

int main(void) {
    srand(time(NULL));

    Arena *arena = arena_create(100000000);
    size_t input_dim = IMAGE_HEIGHT * IMAGE_WIDTH;

    printf("Loading data\n");

    MNISTData *test_data = load_dataset(arena, NUM_TEST_EXAMPLES, TEST_IMAGES_FILEPATH, TEST_LABELS_FILEPATH);
    MNISTData *data = get_zeros_and_ones(arena, test_data);

    printf("Creating network\n");

    Value **inputs = inputs_create(arena, input_dim);
    Value *y = value_create_constant(arena, 0);
    // Value *minus_one = value_create_constant(arena, -1);

    NetworkConfig config = {
        .num_inputs = input_dim,
        .num_layers = 2,
        .num_neurons = (size_t[]) { 16, 1 },
        .hidden_activation = ACT_RELU,
        .output_activation = ACT_SIGMOID
    };

    Value **outputs = network_create(arena, inputs, config);
    Value *y_pred = outputs[0];
    Value *loss = loss_mean_squared_error(arena, y, y_pred);

    printf("Creating graph.. this may take a bit\n");

    Graph *graph = graph_create(arena, loss, 300000);

    printf("Final value count = %zu\n", graph->num_values);

    size_t num_iterations = 100 * data->num_items;
    float learning_rate = 0.0003;
    float epoch_loss = 0;
    bool do_lr_scheduling = false;
    size_t lr_schedule = 5 * data->num_items;
    float lr_factor = 0.5;
    float lr_minimum = 0.0003;

    printf("Starting training.. each epoch will have %u iterations\n", data->num_items);

    for (size_t i = 0; i < num_iterations; i++) {
        // Load example
        size_t index = (size_t) (data->num_items * (float) rand() / (float) RAND_MAX);
        size_t start_index = index * data->num_rows * data->num_cols;

        for (size_t row = 0; row < data->num_rows; row++) {
            for (size_t col = 0; col < data->num_cols; col++) {
                size_t pixel_index = start_index + row * data->num_cols + col;
                uint8_t pixel = data->images[pixel_index];

                inputs[row * data->num_cols + col]->data = (float) pixel / (float) 255;
            }
        }

        y->data = (float) data->labels[index];

        graph_optimisation_step(graph, learning_rate);

        epoch_loss += graph->values[0]->data;

        if ((i + 1) % data->num_items == 0) {
            printf("Epoch: %4zu, Loss: %f\n", (i + 1) / data->num_items, epoch_loss / data->num_items);
            epoch_loss = 0;
        }

        if ((i + 1) % lr_schedule == 0 && learning_rate > lr_minimum && do_lr_scheduling) {
            float new_lr = lr_factor * learning_rate;

            if (new_lr < lr_minimum) {
                new_lr = lr_minimum;
            }

            printf("Reducing learning rate from %f to %f\n", learning_rate, new_lr);

            learning_rate = new_lr;
        }
    }

    // Inference starts here
    InitWindow(WINDOW_W, WINDOW_H, "MNIST Inference");
    SetTargetFPS(TARGET_FPS);

    // initialise_image();

    while (!WindowShouldClose())
    {
        // Inputs
        Vector2 mouse_position = GetMousePosition();

        uint8_t col = (uint8_t) (mouse_position.x / PIXEL_SIZE);
        uint8_t row = (uint8_t) (mouse_position.y / PIXEL_SIZE);

        if (col < IMAGE_WIDTH && row < IMAGE_HEIGHT) {
            if (IsMouseButtonDown(0)) {
                inference_image[row][col] = 250;
            }
            if (IsMouseButtonDown(1)) {
                inference_image[row][col] = 0;
            }
        }

        if (IsKeyPressed(KEY_R)) {
            initialise_image();
        }

        // Updates

        for (uint8_t i = 0; i < IMAGE_HEIGHT; i++) {
            for (uint8_t j = 0; j < IMAGE_WIDTH; j++) {
                uint8_t pixel = inference_image[i][j];
                size_t input_index = i * data->num_cols + j;

                inputs[input_index]->data = (float) pixel / (float) 255;
            }
        }

        graph_forward(graph); // Inference

        // Draw
        BeginDrawing();

        ClearBackground(DARKGRAY);

        for (uint8_t i = 0; i < IMAGE_HEIGHT; i++) {
            for (uint8_t j = 0; j < IMAGE_WIDTH; j++) {
                uint8_t pixel = inference_image[i][j];
                DrawRectangle(j * 16, i * 16, 16, 16, (Color) {pixel, pixel, pixel, 255});
            }
        }

        DrawText("Draw a digit on the canvas on the left", 462, 40, 20, LIGHTGRAY);
        DrawText("Press [R] to reset canvas", 462, 80, 20, LIGHTGRAY);
        DrawText("Press [ESC] to exit", 462, 120, 20, LIGHTGRAY);

        char fps_label[10];
        char label[40];

        sprintf(fps_label, "FPS: %d", GetFPS());
        sprintf(label, "Prediction: %f", y_pred->data);

        DrawText(fps_label, 800, 400, 20, LIGHTGRAY);
        DrawText(label, 462, 200, 20, LIGHTGRAY);

        EndDrawing();
    }

    CloseWindow();
    arena_destroy(arena);
    return 0;
}
