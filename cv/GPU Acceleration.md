# GPU Acceleration

GPU acceleration is typically done using CUDA (NVidia) or OpenCL interfaces

- TensorFlow currently only supports up to CUDA 9 (current version is CUDA 10)

## Methods

### NVidia Docker
This is the default method, however it only works on a Linux base system. Docker on Windows cannot connect to the CUDA

### Amazon EC2

Amazon EC2 has several images that are prepared for deep learning. They can use elastic GPUs for acceleration

1. Setup EC2 Instance

2. Open ports 8888 and 8889

3. SSH into the cluster ([puTTY guide](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/putty.html))

4. Run Jupyter Lab with the command

   ```
   jupyter lab --ip 0.0.0.0
   ```

   This command will print a URL, copy it into the browser. Modify the ip so that it reflect's the host's actual IP found in the ec2 console

5. Install opencv and tensorflow

   ```
   conda install -c conda-forge opencv tensorflow-gpu
   ```

### Windows (/desktop)

- [Install CUDA 9.0](https://developer.nvidia.com/cuda-90-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal) (NVidia)

- Setup CUDA path variables using [tensorflow's instructions](https://www.tensorflow.org/install/gpu)

  Make sure the environment variable path matches the instructions. Likely this will mean manually adding CUPTI path (`CUDA\v9.0\extras\CUPTI\libx64`), despite other parts of CUDA being added automatically.

- [Install cuDNN 7](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#installwindows) (NVidia). This will require an account, but its free.  

  Manually move the files from the cuDNN zip to the CUDA installation path (bin -> bin, lib-> lib, etc). CUDA installation path can be found in environment variables, but it is probably here:
  `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0`

- Install [TensorFlow from source](https://www.tensorflow.org/install/source_windows)

  - Install Bazel
  - Install MSYS2
  - Install VS2015 C++ (2017 may also work)
  - pip (if using anaconda then: `conda install -y -c anaconda pip`)

  All instructions in the link. This will require a fair bit of configuration of environment variables