# --------------------------------------------------------------------
# This file is part of mpFlow.
#
# mpFlow is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# mpFlow is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with mpFlow. If not, see <http:#www.gnu.org/licenses/>.
#
# Copyright (C) 2015 Patrik Gebhardt
# Contact: patrik.gebhardt@rub.de
# --------------------------------------------------------------------

##############################
# Load build configuration
##############################
CONFIG_FILE ?= Makefile.config
include $(CONFIG_FILE)

##############################
# Main output directories
##############################
prefix ?= /usr/local
ROOT_BUILD_DIR := build

# adjust build dir for debug configuration
DEBUG ?= 0
ifeq ($(DEBUG), 1)
	BUILD_DIR := $(ROOT_BUILD_DIR)/debug
else
	BUILD_DIR := $(ROOT_BUILD_DIR)/release
endif

##############################
# Compiler
##############################
AR := ar rcs
CXX ?= /usr/bin/g++

# use of custom compiler
ifdef CUSTOM_CXX
	CXX := $(CUSTOM_CXX)
endif

# Target build architecture
TARGET_ARCH_NAME ?= $(shell $(CXX) -dumpmachine)
BUILD_DIR := $(BUILD_DIR)/$(TARGET_ARCH_NAME)

# get cuda directory for target architecture
CUDA_DIR := $(CUDA_TOOLKIT_ROOT)
ifdef CUDA_TARGET_DIR
	CUDA_DIR := $(CUDA_DIR)/targets/$(CUDA_TARGET_DIR)
endif

##############################
# Includes and libraries
##############################
LIBRARIES := mpflow_static distmesh_static qhullstatic cudart_static cublas_static culibos pthread dl
LIBRARY_DIRS +=
INCLUDE_DIRS += $(CUDA_DIR)/include ./utils/include

# link aganinst librt, only if it exists
ifeq ($(shell echo "int main() {}" | $(CXX) -o /dev/null -x c - -lrt 2>&1),)
	LIBRARIES += rt
endif

# add (CUDA_DIR)/lib64 only if it exists
ifeq ("$(wildcard $(CUDA_DIR)/lib64)", "")
	LIBRARY_DIRS += $(CUDA_DIR)/lib
else
	LIBRARY_DIRS += $(CUDA_DIR)/lib64
endif

##############################
# Compiler Flags
##############################
COMMON_FLAGS := $(addprefix -I, $(INCLUDE_DIRS))
CXXFLAGS := -std=c++11 -fPIC
LINKFLAGS := -fPIC -static-libstdc++
LDFLAGS := $(addprefix -l, $(LIBRARIES)) $(addprefix -L, $(LIBRARY_DIRS)) $(addprefix -Xlinker -rpath , $(LIBRARY_DIRS))

# Use double precision floating points
ifdef DOUBLE_PRECISION
	COMMON_FLAGS += -DUSE_DOUBLE
endif

# Set compiler flags for debug configuration
ifeq ($(DEBUG), 1)
	COMMON_FLAGS += -g -O0 -DDEBUG
else
	COMMON_FLAGS += -O3 -DNDEBUG
endif

##############################
# Source Files
##############################
CXX_SRCS := $(shell find src -name "*.cpp")
UTILS_SRCS := $(shell find utils/src -name "*.cpp")

# Object files
CXX_OBJS := $(addprefix $(BUILD_DIR)/objs/, $(CXX_SRCS:.cpp=.o))
UTILS_OBJS := $(addprefix $(BUILD_DIR)/objs/, $(UTILS_SRCS:.cpp=.o))
BINS := $(patsubst src%.cpp, $(BUILD_DIR)/bin%, $(CXX_SRCS))

##############################
# Build targets
##############################
.PHONY: all clean

all: $(BINS)

$(BINS): $(BUILD_DIR)/bin/% : $(BUILD_DIR)/objs/src/%.o $(UTILS_OBJS)
	@echo [ Linking ] $@
	@mkdir -p $(BUILD_DIR)/bin
	@$(CXX) -o $@ $< $(UTILS_OBJS) $(COMMON_FLAGS) $(LDFLAGS) $(LINKFLAGS)

$(BUILD_DIR)/objs/%.o: %.cpp
	@echo [ CXX ] $<
	@$(foreach d, $(subst /, ,${@D}), mkdir -p $d && cd $d && ):
	@$(CXX) $(CXXFLAGS) $(COMMON_FLAGS) -c -o $@ $<

clean:
	@rm -rf $(ROOT_BUILD_DIR)
