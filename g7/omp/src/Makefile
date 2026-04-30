CC     = gcc
CFLAGS = -O2 -fopenmp -g

TARGET = docs

all: $(TARGET)

$(TARGET): docs-omp.c
	$(CC) $(CFLAGS) -o $(TARGET) docs-omp.c

clean:
	rm -f $(TARGET)