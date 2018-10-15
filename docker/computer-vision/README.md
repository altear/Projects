# Docker - OpenCV Jupyter
Docker file for segmentation of video. Contains Jupyter, OpenCV (3.4), and FFmpeg. 
- Now added Theano, TensorFlow, and Keras

## Requirements 
- Docker ([download](https://www.docker.com/products/docker-desktop))

## Usage
### Build
To download the dockerfile ([install Git](https://git-scm.com/downloads), or download manually):
```
git clone blah
cd jupyter-cv
```

To build the docker image run the following code. Note that it will likely take a few hours since it's downloading and compiling some very large files. The good news is that it only needs to be run once.
```
docker build -t altear:jupyter-cv .
```

### Run
To create a container from the image, run something **like** this (change the path for the `-v` parameter to reflect the system you want to mount) 
```
docker run -p 8888:8888 -v C:\Users\Owner\Documents\projects:/home/jovyan/work/shared altear:jupyter-cv
```

To access the local Jupyter Notebook, go to
http://localhost:8888/tree

## Sources
Some sources were used as examples for how to build the sources. They may be useful as supplementary documentation 

### FFmpeg 
A wire frame for the FFmpeg installation was found here:
https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu

### OpenCV
A frame for the OpenCV installation was found here:
https://hub.docker.com/r/valian/docker-python-opencv-ffmpeg/~/dockerfile/

## Author

Andre Telfer
telfer006@gmail.com