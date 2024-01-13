/*

Load MNIST data and display random examples

*/

#include <time.h>
#include "mnist.h"
#include "raylib.h"

#define WINDOW_W    448
#define WINDOW_H    484
#define TARGET_FPS  60
#define PIXEL_SIZE  16

typedef struct {
    size_t index;
    size_t start_index;
    uint8_t label;
    char text_label[10];
    uint8_t image[IMAGE_HEIGHT][IMAGE_WIDTH];
} DisplayData;

int main(void) {
    srand(time(NULL));

    InitWindow(WINDOW_W, WINDOW_H, "MNIST Data");
    SetTargetFPS(TARGET_FPS);

    Arena *arena = arena_create(100000000);
    MNISTData *data = load_dataset(arena, NUM_TRAIN_EXAMPLES, TRAIN_IMAGES_FILEPATH, TRAIN_LABELS_FILEPATH);
    // MNISTData *data = load_dataset(arena, NUM_TEST_EXAMPLES, TEST_IMAGES_FILEPATH, TEST_LABELS_FILEPATH);
    DisplayData display_data = { };

    bool first_frame = true;

    while (!WindowShouldClose())
    {
        if (IsKeyPressed(KEY_SPACE) || first_frame) {
            first_frame = false;
            display_data.index = (size_t) (data->num_items * (float) rand() / (float) RAND_MAX);
            display_data.start_index = display_data.index * data->num_rows * data->num_cols;
            display_data.label = data->labels[display_data.index];
            sprintf(display_data.text_label, "Label: %hhu", display_data.label);

            for (size_t row = 0; row < data->num_rows; row++) {
                for (size_t col = 0; col < data->num_cols; col++) {
                    size_t pixel_index = display_data.start_index + row * data->num_cols + col;
                    uint8_t pixel = data->images[pixel_index];

                    display_data.image[row][col] = pixel;
                }
            }
        }

        BeginDrawing();

        ClearBackground(RAYWHITE);

        for (size_t row = 0; row < data->num_rows; row++) {
            for (size_t col = 0; col < data->num_cols; col++) {
                uint8_t pixel = display_data.image[row][col];
                Color pixel_color = (Color) { pixel, pixel, pixel, 255 };
                DrawRectangle(col * PIXEL_SIZE, row * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE, pixel_color);
            }
        }

        DrawText(display_data.text_label, 180, 460, 20, DARKGRAY);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
