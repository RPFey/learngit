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

