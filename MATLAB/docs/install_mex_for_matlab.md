# 为MATLAB软件安装MEX命令

MATLAB版本 : MATLAB R2016b [学术版]。

操作系统版本: Windows 10 [64位]、CentOS 7 [64位]、Ubuntu 16.04 [64位]。

## 在Windows 10操作系统下安装MEX命令

### 原因剖析

无法成功安装MEX命令的主要原因可能在于：没有细读MATLAB提供的官方文档，具体如下所示：

[Troubleshooting and Limitations Compiling C/C++ MEX Files with MinGW-w64](https://www.mathworks.com/help/matlab/matlab_external/compiling-c-mex-files-with-mingw.html)

[Manually Configure MinGW for MATLAB](https://www.mathworks.com/help/matlab/matlab_external/compiling-c-mex-files-with-mingw.html#bu0q4cc-1)

根据MATLAB提供的官方指南进行安装，特别注意：系统环境变量“MW_MINGW64_LOC”的设置。

本人就出现过系统环境变量“MW_MINGW64_LOC”错误设置的情况；需要将系统环境变量“MW_MINGW64_LOC”设置到“mingw64”子文件夹层级，这样MATLAB才能找到“bin”子目录（位于“MW_MINGW64_LOC”中）。

### 成功安装

出现以下安装提示非常重要：

```
... 正在查找编译器 'MinGW64 Compiler (C)'...
... 正在查找环境变量 'MW_MINGW64_LOC'...是('C:\TDM-GCC-64\mingw64')。
... 正在查找文件 'C:\TDM-GCC-64\mingw64\bin\gcc.exe'...是。
... 正在查找文件夹 'C:\TDM-GCC-64\mingw64'...是。
找到已安装的编译器 'MinGW64 Compiler (C)'。
```

完整的成功安装输出如下所示：

```
>> mex -setup -v
详细模式已开。
... 正在查找编译器 'Intel C++ Composer XE 2013 with Microsoft SDK 7.1 (C)'...
... 正在查找环境变量 'ICPP_COMPILER14'...否。
... 正在查找环境变量 'ICPP_COMPILER13'...否。
找不到已安装的编译器 'Intel C++ Composer XE 2013 with Microsoft SDK 7.1 (C)'。
... 正在查找编译器 'Intel C++ Composer XE 2013 with Microsoft Visual Studio 2012 (C)'...
... 正在查找环境变量 'ICPP_COMPILER14'...否。
... 正在查找环境变量 'ICPP_COMPILER13'...否。
找不到已安装的编译器 'Intel C++ Composer XE 2013 with Microsoft Visual Studio 2012 (C)'。
... 正在查找编译器 'Intel C++ Composer XE 2013 with Microsoft Visual Studio 2013 (C)'...
... 正在查找环境变量 'ICPP_COMPILER14'...否。
... 正在查找环境变量 'ICPP_COMPILER13'...否。
找不到已安装的编译器 'Intel C++ Composer XE 2013 with Microsoft Visual Studio 2013 (C)'。
... 正在查找编译器 'Intel Parallel Studio XE 2015 with Microsoft SDK 7.1 (C)'...
... 正在查找环境变量 'ICPP_COMPILER15'...否。
找不到已安装的编译器 'Intel Parallel Studio XE 2015 with Microsoft SDK 7.1 (C)'。
... 正在查找编译器 'Intel Parallel Studio XE 2015 with Microsoft Visual Studio 2012 (C)'...
... 正在查找环境变量 'ICPP_COMPILER15'...否。
找不到已安装的编译器 'Intel Parallel Studio XE 2015 with Microsoft Visual Studio 2012 (C)'。
... 正在查找编译器 'Intel Parallel Studio XE 2015 with Microsoft Visual Studio 2013 (C)'...
... 正在查找环境变量 'ICPP_COMPILER15'...否。
找不到已安装的编译器 'Intel Parallel Studio XE 2015 with Microsoft Visual Studio 2013 (C)'。
... 正在查找编译器 'Intel Parallel Studio XE 2015 with Microsoft Visual Studio 2015 (C)'...
... 正在查找环境变量 'ICPP_COMPILER15'...否。
找不到已安装的编译器 'Intel Parallel Studio XE 2015 with Microsoft Visual Studio 2015 (C)'。
... 正在查找编译器 'Intel Parallel Studio XE 2016 with Microsoft SDK 7.1 (C)'...
... 正在查找环境变量 'ICPP_COMPILER16'...否。
找不到已安装的编译器 'Intel Parallel Studio XE 2016 with Microsoft SDK 7.1 (C)'。
... 正在查找编译器 'Intel Parallel Studio XE 2016 with Microsoft Visual Studio 2012 (C)'...
... 正在查找环境变量 'ICPP_COMPILER16'...否。
找不到已安装的编译器 'Intel Parallel Studio XE 2016 with Microsoft Visual Studio 2012 (C)'。
... 正在查找编译器 'Intel Parallel Studio XE 2016 with Microsoft Visual Studio 2013 (C)'...
... 正在查找环境变量 'ICPP_COMPILER16'...否。
找不到已安装的编译器 'Intel Parallel Studio XE 2016 with Microsoft Visual Studio 2013 (C)'。
... 正在查找编译器 'Intel Parallel Studio XE 2016 with Microsoft Visual Studio 2015 (C)'...
... 正在查找环境变量 'ICPP_COMPILER16'...否。
找不到已安装的编译器 'Intel Parallel Studio XE 2016 with Microsoft Visual Studio 2015 (C)'。
... 正在查找编译器 'MinGW64 Compiler (C)'...
... 正在查找环境变量 'MW_MINGW64_LOC'...是('C:\TDM-GCC-64\mingw64')。
... 正在查找文件 'C:\TDM-GCC-64\mingw64\bin\gcc.exe'...是。
... 正在查找文件夹 'C:\TDM-GCC-64\mingw64'...是。
找到已安装的编译器 'MinGW64 Compiler (C)'。
... 正在查找编译器 'Microsoft Visual C++ 2012 (C)'...
... 正在查找注册表设置 'HKLM\SOFTWARE\Microsoft\VisualStudio\SxS\VS7' 11.0...否。
... 正在查找注册表设置 'HKCU\SOFTWARE\Microsoft\VisualStudio\SxS\VS7' 11.0...否。
... 正在查找注册表设置 'HKLM\SOFTWARE\Wow6432Node\Microsoft\VisualStudio\SxS\VS7' 11.0...否。
... 正在查找注册表设置 'HKCU\SOFTWARE\Wow6432Node\Microsoft\VisualStudio\SxS\VS7' 11.0...否。
找不到已安装的编译器 'Microsoft Visual C++ 2012 (C)'。
... 正在查找编译器 'Microsoft Visual C++ 2013 Professional (C)'...
... 正在查找注册表设置 'HKLM\SOFTWARE\Microsoft\VisualStudio\SxS\VS7' 12.0...否。
... 正在查找注册表设置 'HKCU\SOFTWARE\Microsoft\VisualStudio\SxS\VS7' 12.0...否。
... 正在查找注册表设置 'HKLM\SOFTWARE\Wow6432Node\Microsoft\VisualStudio\SxS\VS7' 12.0...否。
... 正在查找注册表设置 'HKCU\SOFTWARE\Wow6432Node\Microsoft\VisualStudio\SxS\VS7' 12.0...否。
找不到已安装的编译器 'Microsoft Visual C++ 2013 Professional (C)'。
... 正在查找编译器 'Microsoft Visual C++ 2015 Professional (C)'...
... 正在查找注册表设置 'HKLM\SOFTWARE\Microsoft\VisualStudio\SxS\VS7' 14.0...否。
... 正在查找注册表设置 'HKCU\SOFTWARE\Microsoft\VisualStudio\SxS\VS7' 14.0...否。
... 正在查找注册表设置 'HKLM\SOFTWARE\Wow6432Node\Microsoft\VisualStudio\SxS\VS7' 14.0...否。
... 正在查找注册表设置 'HKCU\SOFTWARE\Wow6432Node\Microsoft\VisualStudio\SxS\VS7' 14.0...否。
找不到已安装的编译器 'Microsoft Visual C++ 2015 Professional (C)'。
... 正在查找编译器 'Microsoft Windows SDK 7.1 (C)'...
... 正在查找注册表设置 'HKLM\SOFTWARE\Microsoft\Microsoft SDKs\Windows\v7.1' InstallationFolder...否。
... 正在查找注册表设置 'HKLM\SOFTWARE\Wow6432Node\Microsoft\Microsoft SDKs\Windows\v7.1' InstallationFolder...否。
找不到已安装的编译器 'Microsoft Windows SDK 7.1 (C)'。
已将选项文件从 'C:\Program Files\MATLAB\R2016b\bin\win64\mexopts\mingw64.xml' 复制到 'C:\Users\syhdqq\AppData\Roaming\MathWorks\MATLAB\R2016b\mex_C_win64.xml'。
MEX 配置为使用 'MinGW64 Compiler (C)' 以进行 C 语言编译。
警告: MATLAB C 和 Fortran API 已更改，现可支持
	包含 2^32-1 个以上元素的 MATLAB 变量。不久以后，
	 您需要更新代码以利用
	 新的 API。您可以在以下网址找到相关详细信息:
	 http://www.mathworks.com/help/matlab/matlab_external/upgrading-mex-files-to-use-64-bit-api.html。

要选择不同的语言，请从以下选项中选择一种命令:
 mex -setup C++ 
 mex -setup FORTRAN
```

## 在CentOS 7操作系统下安装MEX命令

### 升级gcc编译器

使用SHELL命令```gcc -v```，查看gcc的版本号；默认为“gcc 版本 4.8.5 20150623 (Red Hat 4.8.5-11) (GCC)”。

查看MATLAB要求的[gcc最低版本号](https://www.mathworks.com/support/compilers.html)，默认为“GCC C/C++ 4.9 ”（注：“GCC C/C++ 4.9 will be replaced by a newer version in a future release”）。

显然，CentOS 7操作系统自带的gcc版本号较低，不能被MATLAB有效地识别；因此需要进一步升级gcc编译器（至少为4.9版本）。

使用**devtoolset-4**工具，即可便捷地升级gcc编译器。具体的SHELL操作步骤，参阅[官方安装指南](https://www.softwarecollections.org/en/scls/rhscl/devtoolset-4/)。

升级后的gcc版本为：“gcc version 5.3.1 20160406 (Red Hat 5.3.1-6) (GCC)”。

### 安装MEX命令

在SHELL或MATLAB中输入以下命令：

```
mex -setup -v
```

## 在Ubuntu 16.04操作系统下安装MEX命令

Ubuntu 16.04操作系统中，gcc编译器默认为：“gcc version 5.4.0 20160609 (Ubuntu 5.4.0-6ubuntu1~16.04.4)”。因此，无需对gcc编译器进行升级。

### 安装MEX命令

在SHELL或MATLAB中输入以下命令：

```
mex -setup -v
```
