CC      = /usr/bin/g++
CFLAGS  = -Wall -Wextra -std=c++20
OPTFLAGS = -O3
DFLAGS = -g
LDFLAGS = -lm -lrt -lz -L./libs/armnn -larmnn -larmnnOnnxParser  -Wl,-rpath=$(realpath ./libs/armnn)

OBJDIR = obj
SOURCEDIR = src
HEADERDIR = src
ARMNN = ./libs/armnn

SOURCES = $(wildcard $(SOURCEDIR)/*.cpp)
OBJECTS = $(patsubst $(SOURCEDIR)/%.cpp, $(OBJDIR)/%.o, $(SOURCES))

$(OBJDIR)/%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

BINARY = AcousticBubble

.PHONY: all clean

all: $(BINARY)

$(info    DEPS is $(OBJDIR)/$(OBJECTS))
$(info $(SOURCES))

$(BINARY): $(OBJDIR)/$(OBJECTS)
	$(CC) $(CFLAGS) $(OPTFLAGS) -I$(HEADERDIR) -I$(SOURCEDIR) -I$(ARMNN)/include $(OBJECTS) -o $(BINARY) $(LDFLAGS)

$(OBJDIR)/%.o: $(SOURCEDIR)/%.cpp $(HEADERDIR)/%.h
	$(CC) $(CFLAGS) -I$(HEADERDIR) -I$(SOURCEDIR) -I$(ARMNN)/include -c $< -o $@ $(LDFLAGS)

clean:
	rm -f $(OBJDIR)/*.o $(BINARY)
