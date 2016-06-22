FROM nvidia/cuda:8.0-ubuntu16.04
MAINTAINER Patrik Gebhardt <patrik.gebhardt@rub.de>

RUN apt-get update && apt-get install -y \
    make \
    git-core \
    gcc \
    g++ \
    libeigen3-dev \
    libqhull-dev

# configure build environment
RUN mkdir /code
WORKDIR /code

# build libdistmesh
RUN git clone https://github.com/pgebhardt/libdistmesh.git --recursive
WORKDIR /code/libdistmesh
RUN cp Makefile.config.example Makefile.config
RUN echo "INCLUDE_DIRS += /usr/include/eigen3" >> Makefile.config
RUN make -j4 && make install

# build mpflow
WORKDIR /code
RUN git clone https://github.com/pgebhardt/mpflow.git --recursive
WORKDIR /code/mpflow
RUN cp Makefile.config.example Makefile.config
RUN echo "INCLUDE_DIRS += /usr/include/eigen3" >> Makefile.config
RUN echo "CUDA_ARCH += -gencode arch=compute_50,code=sm_50" >> Makefile.config
RUN echo "CUDA_ARCH += -gencode arch=compute_52,code=sm_52" >> Makefile.config
RUN make -j4 && make install

# build mpflow-tools
RUN mkdir /code/mpflow-tools
WORKDIR /code/mpflow-tools
COPY . ./
RUN cp Makefile.config.example Makefile.config
RUN echo "INCLUDE_DIRS += /usr/include/eigen3" >> Makefile.config
RUN make -j4 && make install

# Volumes for config files
VOLUMES /config
