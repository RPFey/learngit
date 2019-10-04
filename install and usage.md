# Install

pyrealsense 安装需要一些依赖项 , 否则cmake 时会报错

<https://blog.csdn.net/u012180635/article/details/82143340>

# 报错及解决

## usb 端口问题

```
Traceback (most recent call last):
  File "align-depth2color.py", line 26, in <module>
    profile = pipeline.start(config)
RuntimeError: Couldn't resolve requests
```

摄像头必须连接在usb3.0 的端口上，2.0会报错

# 相机使用

与 c++ 中直接设置枚举不同，python 中将需要的信息放在 camero_info 这个类中。

直接选取相应的属性即可获得。

```python
sensor.get_info(rs.camera_info.name)
```

便可以返回相应的信息字符串。



rs.options 类中有有对各种option 操作的方法，但是貌似是一个抽象类，其中的方法必须在sensor 对象上是使用。所以 docs 中 

```python
def f(self: pyrealsense2.options, option:pyrealsense2.option) -> str
```

前者的self 其实必须是 sensor 类。