# CMAKE

[official tutorials](https://cmake.org/cmake/help/latest/guide/tutorial/index.html)

对于源码编译的库 

cmake -D [parameters] ..

make -j 

make install 是把.h 文件写入/usr/lib/local 下

MESSAGE(STATUS " path " ${...} ) 可以在编译时检查路径是否出错

```cmake
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

## Compiler and Linker

cmake provides a set of variables that specify compiler and linker flags. The variable invovled has the following form:

* CMAKE_\<LANG>_FLAGS

* CMAKE_\<LANG>\_FLAGS_\<CONFIG>

\<LANG> specifies the languange, like C, CXX, Fortran, ... 

\<CONFIG> is the uppercase of the build types.

The first variable will be applied to all build types. The second will to applied to the build type specified by \<CONFIG>.

The same goes for linker FLAGS

* CMAKE_\<TARGETTYPE>_LINKER_FLAGS

* CMAKE_\<TARGETTYPE>\_LINKER_FLAGS_\<CONFIG>

\<TARGETTYPE> :

1. EXE created by the add_executable(...)
2. SHARED created by add_library(name SHARED ...)
3. STATIC created by add_library(name STATIC ...)
4. MODULE created by add_library(name MODULE ...)

**Append flags!**

```cmake
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror")
# or use
string(APPEND CMAKE_CXX_FLAGS " -Wall -Werror")
```
