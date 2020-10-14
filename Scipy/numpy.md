# Numpy

Don't Use np.linalg.inv !!! Use a.I to get the inverse matrix

## operation

np.stack : axis = -1 是在最后一个维度堆叠（合并）

### 随机数

np.random.choice(a, size=None) ; a 1-D array-like or int, size int or tuple. 从 a 中随机选取 size 个元素组成 size 的矩阵。

## Data types

* [built-in data type](https://numpy.org/doc/stable/reference/arrays.scalars.html#arrays-scalars-built-in)

这个 numpy 自己定义的数据类型，比如常见的 `np.int64`, `np.float32` 之类的

* 自定义的数据类型

1. 大小端存储模式
2. 数据类型（整数？浮点数？）
3. 存储位数 （字节数）

eg. `>i4`

大端四位整数

| 符号 | 类型 |
| :-: | :-: |
| `?` | boolean |
| `b` | (signed) byte |
| `B` | unsigned byte |
| `i` | (signed) int |
| `u` | (unsigned) integer |
| `f` | floating point |
| `c` | complex floating point |
| `m` | timedelta |
| `M` | datetime |
| `O` | (Python) Objects |
| `S`, `a` | zero-terminated bytes |
| `U` | Unicode srting |
| `V` | raw data |

## 序列化

这就涉及到字节转换与传输问题了。使用函数 `np.tobytes()` 转换为字节。
