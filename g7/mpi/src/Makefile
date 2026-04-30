CC     = mpicc
CFLAGS = -O2 -fopenmp -g

TARGET = docs

all: $(TARGET)

$(TARGET): docs-mpi.c
	$(CC) $(CFLAGS) -o $(TARGET) docs-mpi.c

clean:
	rm -f $(TARGET)