# 在Ubuntu上安装CUDA并行计算开发环境

* ************************************************* *

NVIDIA GPU硬件配置 [hardware]: GK110GL [Tesla K20c]，共2颗。

操作系统版本 [OS]: Ubuntu 16.04.2 LTS [Xenial Xerus]。

NVIDA驱动版本 [driver]: 375.26。

CUDA开发工具包版本 [toolkit]: cuda_8.0.61_375.26_linux。

* ************************************************* *

## 阅读CUDA官方安装指南

阅读[CUDA官方安装指南](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#axzz4eDfKJkHm)，获取基本的安装信息（**强烈推荐阅读**）。

```
查看NVIDIA GPU型号与颗数
$ lspci | grep -i nvidia
82:00.0 3D controller: NVIDIA Corporation GK110GL [Tesla K20c] (rev a1)
83:00.0 3D controller: NVIDIA Corporation GK110GL [Tesla K20c] (rev a1)

查看操作系统版本
$ cat /etc/os-release
x86_64
NAME="Ubuntu"
VERSION="16.04.2 LTS (Xenial Xerus)"
ID=ubuntu
ID_LIKE=debian
PRETTY_NAME="Ubuntu 16.04.2 LTS"
VERSION_ID="16.04"
HOME_URL="http://www.ubuntu.com/"
SUPPORT_URL="http://help.ubuntu.com/"
BUG_REPORT_URL="http://bugs.launchpad.net/ubuntu/"
VERSION_CODENAME=xenial
UBUNTU_CODENAME=xenial

查看GCC编译器版本
$ gcc --version
gcc (Ubuntu 5.4.0-6ubuntu1~16.04.4) 5.4.0 20160609
Copyright (C) 2015 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

查看操作系统内核信息
$ uname -r
4.8.0-46-generic

安装Linux内核头文件
$sudo apt-get install linux-headers-$(uname -r)
```

## 使用命令行模式安装

采用命令行模式进行安装，切勿使用界面模式。既可使用本地命令行模式，也可使用远程命令行模式（不推荐）。

```
切换到本地命令行模式快捷键：Ctrl + Alt + F1~F6
切换到本地界面模式的快捷键：Ctrl + Alt + F7

禁用本地界面模式
$ sudo service lightdm stop

删除文件/etc/X11/xorg.conf（如果存在的话）
安装之前与安装成功之后，都没有xorg.conf文件
$ ls /etc/X11/ | grep -i xorg

禁用nouveau
创建名为blacklist-nouveau.conf的文件
添加一行即可，不要随意换行；虽然官方指南有换行
$ vi /etc/modprobe.d/blacklist-nouveau.conf
blacklist nouveau option nouveau modeset=0

$ sudo update-initramfs -u
```

下载[CUDA](https://developer.nvidia.com/cuda-downloads)，下载的版本为：cuda_8.0.61_375.26_linux.run（不推荐使用.deb安装包）。

```
赋予执行权限
$ chmod a+x cuda_8.0.61_375.26_linux.run
$ ls -al
-rwxrwxr-x

执行安装程序，特别注意相关参数的设置
不能安装opengl-libs，否则会出现循环登录问题
$ sudo sh cuda_8.0.61_375.26_linux.run --no-opengl-libs

安装过程中，选择：接收安装协议，安装NVIDIA驱动，安装CUDA工具包，安装CUDA示例；但不接收Xserver配置服务

查看NVIDIA设备是否被加载
$ cd /dev/ | grep -i nvidia

加载NVIDIA设备
$ sudo modprobe nvidia

配置环境变量，在.bashrc文件尾部添加以下内容
$ vi .bashrc

export CUDA_HOME="/usr/local/cuda-8.0/bin"
export PATH=${CUDA_HOME}:${PATH}

export CUDA_LD_LIBRARY_PATH="/usr/local/cuda-8.0/lib64"
export LD_LIBRARY_PATH=${CUDA_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}

$ . .bashrc
```

查看安装是否成功。

```
查看NVIDIA版本输出
$ cat /proc/driver/nvidia/version
NVRM version: NVIDIA UNIX x86_64 Kernel Module  375.26  Thu Dec  8 18:36:43 PST 2016
GCC version:  gcc version 5.4.0 20160609 (Ubuntu 5.4.0-6ubuntu1~16.04.4)

查看NVIDIA CUDA编译器版本
$ nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2016 NVIDIA Corporation
Built on Tue_Jan_10_13:22:03_CST_2017
Cuda compilation tools, release 8.0, V8.0.61

查看GPU驱动信息
$ nvidia-smi
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 375.26                 Driver Version: 375.26                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla K20c          Off  | 0000:82:00.0     Off |                    0 |
| 30%   40C    P0    45W / 225W |      0MiB /  4742MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   1  Tesla K20c          Off  | 0000:83:00.0     Off |                    0 |
| 30%   37C    P0    48W / 225W |      0MiB /  4742MiB |     96%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID  Type  Process name                               Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+

查看CUDA常用命令
$ ls /usr/local/cuda/bin/
bin2c        cuda-gdbserver               nsight        nvprune
computeprof  cuda-install-samples-8.0.sh  nvcc          nvvp
crt          cuda-memcheck                nvcc.profile  ptxas
cudafe       cuobjdump                    nvdisasm      uninstall_cuda_8.0.pl
cudafe++     fatbinary                    nvlink
cuda-gdb     gpu-library-advisor          nvprof

恢复界面模式
$ sudo service lightdm start
```

运行CUDA示例，检查CUDA开发环境是否初步搭建成功。注意CUDA示例的文件位置（默认在安装目录下，名为“NVIDIA_CUDA-8.0_Samples”）。

```
$ cd NVIDIA_CUDA-8.0_Samples

耗时较长，需耐心等待
$ make

进入执行目录
$ cd bin/x86_64/linux/release

运行CUDA示例deviceQuery
$ ./deviceQuery

./deviceQuery Starting...

 CUDA Device Query (Runtime API) version (CUDART static linking)

Detected 2 CUDA Capable device(s)

Device 0: "Tesla K20c"
  CUDA Driver Version / Runtime Version          8.0 / 8.0
  CUDA Capability Major/Minor version number:    3.5
  Total amount of global memory:                 4742 MBytes (4972412928 bytes)
  (13) Multiprocessors, (192) CUDA Cores/MP:     2496 CUDA Cores
  GPU Max Clock rate:                            706 MHz (0.71 GHz)
  Memory Clock rate:                             2600 Mhz
  Memory Bus Width:                              320-bit
  L2 Cache Size:                                 1310720 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(65536), 2D=(65536, 65536), 3D=(4096, 4096, 4096)
  Maximum Layered 1D Texture Size, (num) layers  1D=(16384), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(16384, 16384), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  2048
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 2 copy engine(s)
  Run time limit on kernels:                     No
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Enabled
  Device supports Unified Addressing (UVA):      Yes
  Device PCI Domain ID / Bus ID / location ID:   0 / 130 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

Device 1: "Tesla K20c"
  CUDA Driver Version / Runtime Version          8.0 / 8.0
  CUDA Capability Major/Minor version number:    3.5
  Total amount of global memory:                 4742 MBytes (4972412928 bytes)
  (13) Multiprocessors, (192) CUDA Cores/MP:     2496 CUDA Cores
  GPU Max Clock rate:                            706 MHz (0.71 GHz)
  Memory Clock rate:                             2600 Mhz
  Memory Bus Width:                              320-bit
  L2 Cache Size:                                 1310720 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(65536), 2D=(65536, 65536), 3D=(4096, 4096, 4096)
  Maximum Layered 1D Texture Size, (num) layers  1D=(16384), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(16384, 16384), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  2048
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 2 copy engine(s)
  Run time limit on kernels:                     No
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Enabled
  Device supports Unified Addressing (UVA):      Yes
  Device PCI Domain ID / Bus ID / location ID:   0 / 131 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >
> Peer access from Tesla K20c (GPU0) -> Tesla K20c (GPU1) : Yes
> Peer access from Tesla K20c (GPU1) -> Tesla K20c (GPU0) : Yes

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 8.0, CUDA Runtime Version = 8.0, NumDevs = 2, Device0 = Tesla K20c, Device1 = Tesla K20c
Result = PASS （运行成功标识）

运行CUDA示例bandwidthTest
$ ./bandwidthTest

[CUDA Bandwidth Test] - Starting...
Running on...

 Device 0: Tesla K20c
 Quick Mode

 Host to Device Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)        Bandwidth(MB/s)
   33554432                     6038.7

 Device to Host Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)        Bandwidth(MB/s)
   33554432                     6553.5

 Device to Device Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)        Bandwidth(MB/s)
   33554432                     146996.4

Result = PASS （运行成功标识）

NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.
```

## 参考资料

[NeuroSurfer提供的CUDA安装方案](https://devtalk.nvidia.com/default/topic/878117/cuda-setup-and-installation/-solved-titan-x-for-cuda-7-5-login-loop-error-ubuntu-14-04-/1)

* ************************************************* *
此文档不再更新、维护，只做存档目的。如有任何问题，欢迎邮件沟通。

DQQ077 [duanqq077@qq.com]，SUSTC-CS-OPAL

2017-04-26 21:00:00
* ************************************************* *
