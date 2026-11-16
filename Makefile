# Keep CMake-generated files in build/ and forward make's parallel job settings.
.DEFAULT_GOAL := all

.PHONY: all trt configure clean

all: trt

configure:
	cmake -S . -B build -G "Unix Makefiles"

trt: configure
	+$(MAKE) -C build trt
	cmake -E copy_if_different build/trt trt

clean:
	+$(MAKE) -C build clean
	cmake -E rm -f trt
