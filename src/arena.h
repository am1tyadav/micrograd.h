#ifndef ARENA_H
#define ARENA_H

#include <stdlib.h>
#include <assert.h>

typedef struct {
    char *data;
    size_t size;
    size_t position;
} Arena;

// Header

Arena *arena_create(size_t size);
void arena_destroy(Arena *arena);
void *arena_allocate(Arena *arena, size_t size);

// Implementation

Arena *arena_create(size_t size) {
    Arena *arena = (Arena *) calloc(1, sizeof(Arena));

    arena->data = (char *) calloc(size, sizeof(char));
    arena->size = size;
    arena->position = 0;

    return arena;
}

void arena_destroy(Arena *arena) {
    free(arena->data);
    arena->data = NULL;
    free(arena);
    arena = NULL;
}

void *arena_allocate(Arena *arena, size_t size) {
    assert(arena->position + size <= arena->size);

    void *ptr = (void *) (arena->data + arena->position);
    arena->position += size;

    return ptr;
}

#endif // ARENA_H
