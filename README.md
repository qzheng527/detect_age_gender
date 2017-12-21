# detect_age_gender
Age and gender detection using dlib and tensorflow

This is modified based on the https://github.com/dpressel/rude-carnie, adding support for UVC camera and RealSense DS435.
The checkpoinit is downloaded from googdle drive as below.
    Pre-trained Checkpoints

    You can find a pre-trained age checkpoint for inception here:

    https://drive.google.com/drive/folders/0B8N1oYmGLVGWbDZ4Y21GLWxtV1E

    A pre-trained gender checkpoint for inception is available here:

    https://drive.google.com/drive/folders/0B8N1oYmGLVGWemZQd3JMOEZvdGs

To run this program,
1) Get checkpoints from above google drive and put them on 
    checkpoints/age
    checkpoints/gender
2) UVC camera, make sure the camera is inserted. The default setting is cv2.VideoCapture(0) you can do the change based on your system. And if something realsense related error reported, you can delete all the realsense libs/calls.
    python guess.py uvc
3) RealSense DS435, Intel depth camera with RGB camera on the module, which is the one I am using for develop. Insert the camera to USB 3.0 port, and run
    python guess.py rs

Then you can get a window to show the age and gender like below.
![image](https://github.com/qzheng527/detect_age_gender/blob/master/example.png)

Optimization,
My test is on one desktop with Intel(R) Core(TM) i7-5557U CPU @ 3.10GHz.
I did below two to optimize the performance,
1) Build dlib from source, enable AVX
    Clone the code from github:

    git clone https://github.com/davisking/dlib.git
    Build the main dlib library:

    cd dlib
    mkdir build; cd build; cmake .. -DDLIB_USE_CUDA=0 -DUSE_AVX_INSTRUCTIONS=1; cmake --build .
2) Build tensorflow from source to enable MKL, SSE, AVX...

    https://software.intel.com/en-us/articles/build-and-install-tensorflow-on-intel-architecture
    
    bazel build --config=mkl --copt="-DEIGEN_USE_VML" -c opt --copt=-mavx --copt=-msse4.1 --copt=-msse4.2 ...


