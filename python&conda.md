# python

python可以直接加版本号在终端中运行相应的python 版本 ： eg. python3.7

python -m pip 使用pip 工具

当在根目录下时 ， 可以用 pyhton -m pip install --usr 来安装在当前用户下

# Multiprocessing 

采用 Pool/Queue 管理

在 Ubuntu 环境下

```python
import multiprocessing
multiprocessing.set_start_method('spawn', True)
q = multiprocessing.Queue()
jobs = []
for i in range(N):
      p = multiprocessing.Process(target=self.SingleFrameLoss, args = (i, cur_xy, tar_xy, desc)) 
      jobs.append(p)
      p.start()

for p in jobs:
      p.join()

results = [p.get() for p in jobs]
# 得到所有的返回值

# 采用 Pool
pool = multiprocessing.Pool()
```
## 安装包

python setup.py build  <https://docs.python.org/2/install/>

put the file to install into a build directory 

params : 

--build-base=/path/to/build-directory (redirect the build path)

在 setup.py 中 (先以 flann 为例)

```python
setup(name='flann',
      version='1.9.1',
      description='Fast Library for Approximate Nearest Neighbors',
      author='Marius Muja',
      author_email='mariusm@cs.ubc.ca',
      license='BSD',
      url='http://www.cs.ubc.ca/~mariusm/flann/',
      packages=['pyflann', 'pyflann.lib'],
      package_dir={'pyflann.lib': find_path() },  # 在 find_path() 返回目录中寻找 package_data 中的文件，并将找到的文件放在 ./pyflann/lib 下
      package_data={'pyflann.lib': ['libflann.so', 'flann.dll', 'libflann.dll', 'libflann.dylib']}, 
)
```

这里创建了 ./build/lib/pyflann 与 ./build/lib/pyflann/lib 两个目录( 但是只有一个 package,  pyflann ) (python 中 dir.dir 代表 dir/dir) 并且在./pyflann 下要有一个 __init__.py

package_dir 与 package_data ; 前者是规定各个目录下应该放置的数据所在的目录， 后者是数据的名字

python setup.py install (在虚拟环境下直接装入虚拟环境中的 site-packages 下)

copy everything to the installation directory 

params :

--prefix=/prefix/path  (the prefix of the installation path , eg. install to /prefix/lib/site-packages/..)

--user (when don't have root permission)



conda 作为环境管理工具 

如果出现安装包之间不兼容（incompatible with each other）:

1. 可能是安装包与python 版本不匹配，不支持最新的版本。

2. 安装的python 不对（64bit 装了 32bit 的）

   conda 可以构建不同的python 环境（对应不同的版本）

   