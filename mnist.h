#ifndef MNIST_H
#define MNIST_H

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "arena.h"

#define IMAGE_WIDTH             28
#define IMAGE_HEIGHT            28
#define NUM_TRAIN_EXAMPLES      60000
#define NUM_TEST_EXAMPLES       10000
#define TRAIN_IMAGES_FILEPATH   "../data/train-images-idx3-ubyte"
#define TRAIN_LABELS_FILEPATH   "../data/train-labels-idx1-ubyte"
#define TEST_IMAGES_FILEPATH    "../data/t10k-images-idx3-ubyte"
#define TEST_LABELS_FILEPATH    "../data/t10k-labels-idx1-ubyte"

typedef struct {
    uint8_t *images;
    uint8_t *labels;
    uint32_t magic_number_images;
    uint32_t magic_number_labels;
    uint32_t num_images;
    uint32_t num_items;
    uint32_t num_rows;
    uint32_t num_cols;
} MNISTData;

uint32_t reverse_int (uint32_t i) 
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((uint32_t)c1 << 24) + ((uint32_t)c2 << 16) + ((uint32_t)c3 << 8) + c4;
}

void read_images_file(const char *filepath, MNISTData *data) {
    FILE *file = fopen(filepath, "rb");

    assert(file);

    uint32_t magic_number;
    uint32_t num_images;
    uint32_t num_rows;
    uint32_t num_cols;

    fread((char *) &magic_number, 4, 1, file);
    fread((char *) &num_images, 4, 1, file);
    fread((char *) &num_rows, 4, 1, file);
    fread((char *) &num_cols, 4, 1, file);

    uint32_t magic_number_i = reverse_int(magic_number);
    uint32_t num_images_i = reverse_int(num_images);
    uint32_t num_rows_i = reverse_int(num_rows);
    uint32_t num_cols_i = reverse_int(num_cols);

    data->magic_number_images = magic_number_i;
    data->num_images = num_images_i;
    data->num_rows = num_rows_i;
    data->num_cols = num_cols_i;

    for (uint32_t i = 0; i < num_images_i; i++) {
        for (uint32_t row = 0; row < num_rows_i; row++) {
            for (uint32_t col = 0; col < num_cols_i; col++) {
                unsigned char pixel = 0;
                fread((char *) &pixel, sizeof(pixel), 1, file);
                data->images[i * num_cols_i * num_rows_i + row * num_cols_i + col] = (u_int8_t) pixel;
            }
        }
    }

    fclose(file);
}

void read_labels_file(const char *filepath, MNISTData *data) {
    FILE *file = fopen(filepath, "rb");

    assert(file);

    uint32_t magic_number;
    uint32_t num_items;

    fread((char *) &magic_number, 4, 1, file);
    fread((char *) &num_items, 4, 1, file);

    uint32_t magic_number_i = reverse_int(magic_number);
    uint32_t num_items_i = reverse_int(num_items);

    data->magic_number_labels = magic_number_i;
    data->num_items = num_items_i;

    for (size_t i = 0; i < num_items_i; i++) {
        unsigned char label = 0;
        fread((char *) &label, sizeof(label), 1, file);
        data->labels[i] = (u_int8_t) label;
    }

    fclose(file);
}

MNISTData *load_dataset(Arena *arena, size_t num_examples, const char *images_filepath, const char *labels_filepath) {
    MNISTData *data = (MNISTData *) arena_allocate(arena, sizeof(MNISTData));
    uint32_t images_size = num_examples * IMAGE_HEIGHT * IMAGE_WIDTH;

    printf("Loading data requires %u bytes\n", images_size);

    data->images = (uint8_t *) arena_allocate(arena, sizeof(uint8_t) * images_size);
    data->labels = (uint8_t *) arena_allocate(arena, sizeof(uint8_t) * num_examples);

    read_images_file(images_filepath, data);
    read_labels_file(labels_filepath, data);

    return data;
}

#endif
