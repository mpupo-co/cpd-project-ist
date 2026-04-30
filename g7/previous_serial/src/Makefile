CC = gcc
CFLAGS = -O2 -fopenmp
TARGET = docs

all: $(TARGET)

$(TARGET): docs.c
	$(CC) $(CFLAGS) docs.c -o $(TARGET)

clean:
	rm -f $(TARGET)