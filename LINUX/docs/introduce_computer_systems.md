# 深入理解底层计算机系统及其对应用程序的影响

https://www.amazon.com/Computer-Systems-Programmers-Perspective-3rd/dp/013409266X

## 了解编译系统

```
源程序 ---> .c ---> 预处理器 ---> .i ---> 编译器 ---> .s ---> 汇编器 ---> .o ---> 链接器 ---> 可执行目标程序（二进制）
```

## 数字的机器表示方式

### 对比单精度与双精度浮点类型数值

分析程序：

```C
#include <stdio.h>
#include <time.h>

int main( void ) {
    time_t run_time_start, run_time_end;

    /*
     *  compare computational efficiency and accuracy for float- vs. double-precision floating point data type.
     */
    double ft = 0.0, dt = 0.0;
    unsigned long num_iter = 100000000;
    float fv = 2.57, fsum = 0.0;
    double dv = 2.57, dsum = 0.0;
    time( &run_time_start );
    for ( unsigned long ind_iter = 0; ind_iter < num_iter; ind_iter++ ) {
       fsum += fv * fv;
    }
    time( &run_time_end );
    ft = difftime( run_time_end, run_time_start );
    time( &run_time_start );
    for( unsigned long ind_iter = 0; ind_iter < num_iter; ind_iter++ ) {
        dsum += dv * dv;
    }
    time( &run_time_end );
    dt = difftime( run_time_end, run_time_start );
    printf( "* float : %lf,  %f.\n", ft, fsum );
    printf( "* double: %lf,  %lf.\n", dt, dsum );
}

```

运行结果：

```
* float : 1.000000,  134217728.000000.
* double: 0.000000,  660490000.232815.
```
