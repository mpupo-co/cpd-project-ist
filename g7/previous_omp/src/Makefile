CC     = gcc
CFLAGS = -O2 -fopenmp -g

TARGET = docs

all: $(TARGET)

docs-omp: docs-omp.c
	$(CC) $(CFLAGS) -o docs docs-omp.c

clean:
	rm -f $(TARGET)
