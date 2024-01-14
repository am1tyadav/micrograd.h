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

uint8_t inference_image[IMAGE_HEIGHT][IMAGE_WIDTH] = { };

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

    MNISTData *train_data = load_dataset(arena, NUM_TRAIN_EXAMPLES, TRAIN_IMAGES_FILEPATH, TRAIN_LABELS_FILEPATH);
    MNISTData *data = get_zeros_and_ones(arena, train_data);

    printf("Creating model\n");

    Value **inputs = inputs_create(arena, input_dim);
    Value *y = value_create_constant(arena, 0);

    NetworkConfig config = {
        .num_inputs = input_dim,
        .num_layers = 1,
        .num_neurons = (size_t[]) { 1 },
        .output_activation = ACT_SIGMOID
    };

    Value **outputs = network_create(arena, inputs, config);
    Value *y_pred = outputs[0];
    Value *loss = loss_mean_squared_error(arena, y, y_pred);

    printf("Creating graph.. this may take a bit\n");

    Graph *graph = graph_create(arena, loss, 400000);

    printf("Final value count = %zu\n", graph->num_values);

    size_t num_iterations = 2 * data->num_items;
    float learning_rate = 0.0003;
    float epoch_loss = 0;

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
    }

    // Inference starts here
    InitWindow(WINDOW_W, WINDOW_H, "MNIST Inference");
    SetTargetFPS(TARGET_FPS);

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
        char predicted_label[80];

        sprintf(fps_label, "FPS: %d", GetFPS());
        sprintf(label, "Prediction: %f", y_pred->data);
        sprintf(predicted_label, "Predicted Label: %s", y_pred->data < 0.5 ? "0" : "1");

        DrawText(fps_label, 800, 400, 20, LIGHTGRAY);
        DrawText(label, 462, 200, 20, LIGHTGRAY);
        DrawText(predicted_label, 462, 240, 20, LIGHTGRAY);

        EndDrawing();
    }

    CloseWindow();
    arena_destroy(arena);
    return 0;
}
