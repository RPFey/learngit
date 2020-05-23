# CMAKE

[official tutorials](https://cmake.org/cmake/help/latest/guide/tutorial/index.html)

对于源码编译的库 

cmake -D [parameters] ..

make -j 

make install 是把.h 文件写入/usr/lib/local 下

MESSAGE(STATUS " path " ${...} ) 可以在编译时检查路径是否出错

```
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11") 
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()
```

判断编译器支持





