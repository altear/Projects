FROM ubuntu:18.04

MAINTAINER Andre Telfer <telfer006@gmail.com>

ENV HOME=/root
ENV FFMPEG_SOURCES=/root/ffmpeg_sources
ENV FFMPEG_BIN=/root/bin
ENV OPENCV_VERSION="3.4.2"
ENV NASM_VERSION="2.13.03"
ENV YASM_VERSION="1.3.0"
ENV MINICONDA_VERSION="4.5.11"

USER root 

# Install FFmpeg (and general) compilation dependencies
RUN apt-get update && apt-get -y install \
    autoconf \
    automake \
    build-essential \
    cmake \
    git-core \
    libass-dev \
    libfreetype6-dev \
    libsdl2-dev \
    libtool \
    libva-dev \
    libvdpau-dev \
    libvorbis-dev \
    libxcb1-dev \
    libxcb-shm0-dev \
    libxcb-xfixes0-dev \
    pkg-config \
    texinfo \
    wget \
    zlib1g-dev

# Install Conda
RUN wget https://repo.continuum.io/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh -O $HOME/miniconda.sh
RUN bash $HOME/miniconda.sh -b -p $HOME/miniconda
ENV PATH="$HOME/miniconda/bin:${PATH}"

# Install FFmpeg third party dependencies
RUN mkdir -p $FFMPEG_SOURCES $FFMPEG_BIN

# NASM
WORKDIR ${FFMPEG_SOURCES}
ENV PATH="$HOME/ffmpeg_bin:$HOME/bin:${PATH}"
RUN wget https://www.nasm.us/pub/nasm/releasebuilds/${NASM_VERSION}/nasm-${NASM_VERSION}.tar.bz2 && \
    tar xjvf nasm-${NASM_VERSION}.tar.bz2 && \
    cd nasm-${NASM_VERSION} && \
    ./autogen.sh && \
    ./configure --prefix="$HOME/ffmpeg_build" --bindir="$HOME/bin" && \
    make -j"$(nproc)" && \
    make install

# Yasm
RUN wget -O yasm-${YASM_VERSION}.tar.gz https://www.tortall.net/projects/yasm/releases/yasm-${YASM_VERSION}.tar.gz && \
    tar xzvf yasm-${YASM_VERSION}.tar.gz && \
    cd yasm-${YASM_VERSION} && \
    ./configure --prefix="$HOME/ffmpeg_build" --bindir="$HOME/bin" && \
    make -j"$(nproc)" && \
    make install

# libx264
RUN git -C x264 pull 2> /dev/null || git clone --depth 1 https://git.videolan.org/git/x264 && \
    cd x264 && \
    PKG_CONFIG_PATH="$HOME/ffmpeg_build/lib/pkgconfig" ./configure --prefix="$HOME/ffmpeg_build" --bindir="$HOME/bin" --enable-static --enable-pic && \
    make -j"$(nproc)" && \
    make install

# libx265
RUN apt-get install -y mercurial libnuma-dev && \
    if cd x265 2> /dev/null; then hg pull && hg update; else hg clone https://bitbucket.org/multicoreware/x265; fi && \
    cd x265/build/linux && \
    cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX="$HOME/ffmpeg_build" -DENABLE_SHARED=off ../../source && \
    make -j"$(nproc)" && \
    make install

# Install FFmpeg
RUN wget -O ffmpeg-snapshot.tar.bz2 https://ffmpeg.org/releases/ffmpeg-snapshot.tar.bz2 && \
    tar xjvf ffmpeg-snapshot.tar.bz2 && \
    cd ffmpeg && \
    PKG_CONFIG_PATH="$HOME/ffmpeg_build/lib/pkgconfig" \
    ./configure \
        --prefix="$HOME/ffmpeg_build" \
        --pkg-config-flags="--static" \
        --extra-cflags="-I$HOME/ffmpeg_build/include" \
        --extra-ldflags="-L$HOME/ffmpeg_build/lib" \
        --extra-libs="-lpthread -lm" \
        --bindir="$HOME/bin" \
        --enable-gpl \
        --enable-libass \
        --enable-libfreetype \
        --enable-libvorbis \
        --enable-libx264 \
        --enable-libx265 \
        --enable-nonfree \
        --enable-pic \
        --enable-shared && \
    PATH="$HOME/bin:$PATH" && \
    make -j"$(nproc)" && \
    make install && \
    hash -r
ENV PKG_CONFIG_PATH="$HOME/ffmpeg_build/lib/pkgconfig"

# Install OpenCV Dependencies
RUN apt-get install -y software-properties-common && \
    add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main" && \
    apt-get update && \
    apt-get install -y libjasper1 libjasper-dev

RUN apt-get -y update && apt-get -y install \
    python3-dev \
    git \
    wget \
    unzip \
    cmake \
    build-essential \
    pkg-config \
    libatlas-base-dev \
    gfortran \
    libgtk2.0-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libv4l-dev 

RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py && \
    pip3 install numpy

# Move the ffmpeg installation to a global location so that a non-root user can access them 
RUN cp -R $FFMPEG_BIN /usr/local
RUN cp -R $HOME/ffmpeg_build/lib /usr/local
ENV LD_LIBRARY_PATH=/usr/local/lib


# Install OpenCV
WORKDIR / 
ENV OPENCV_VERSION="3.4.2"

RUN apt-get install -y unzip
RUN wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip -O opencv3.zip && \
    unzip -q opencv3.zip && \
    mv /opencv-${OPENCV_VERSION} /opencv && \
    rm opencv3.zip && \
    wget https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip -O opencv_contrib3.zip && \
    unzip -q opencv_contrib3.zip && \
    mv /opencv_contrib-${OPENCV_VERSION} /opencv_contrib && \
    rm opencv_contrib3.zip 

RUN mkdir /opencv/build && cd /opencv/build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -DOPENCV_EXTRA_MODULES_PATH=/opencv_contrib/modules \
    -DBUILD_EXAMPLES=OFF \
    -DWITH_IPP=OFF \
    -DWITH_FFMPEG=ON \
    -DWITH_V4L=ON .. \
    -DBUILD_opencv_java=OFF \
    -DWITH_CUDA=OFF \
    -DWITH_OPENGL=ON \
    -DWITH_OPENCL=ON \
    -DWITH_IPP=ON \
    -DWITH_TBB=ON \
    -DWITH_EIGEN=ON \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DBUILD_PYTHON_SUPPORT=ON \
    -DBUILD_opencv_python2=OFF \
    -DBUILD_opencv_python3=ON \
    -DCMAKE_INSTALL_PREFIX=$(python3 -c "import sys; print(sys.prefix)") \
    -DPYTHON_EXECUTABLE=$(which python3) \
    -DPYTHON_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
    -DPYTHON_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
    ..

RUN cd /opencv/build && \
    make -j$(nproc) && \
    make install && \
    ldconfig 


# Additional Python 
RUN conda install -c conda-forge \
    matplotlib \
    scikit-learn \
    pandas \
    theano \
    tensorflow \
    tensorflow-gpu \
    keras \
    scikit-image \
    jupyter \
    jupyterlab

# Segmentation Git
WORKDIR /home/jovyan/work
RUN git clone https://github.com/matterport/Mask_RCNN 
RUN cd Mask_RCNN && \
    pip3 install -r requirements.txt && \
    python3 setup.py install

ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]


