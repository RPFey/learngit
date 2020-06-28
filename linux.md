# 网络问题

```
sudo ifconfig docker0 down
/etc/init.d/networking restart
```

安装docker 0 导致

```
sudo apt-get remove bcmwl-kernel-source
```

安装其他驱动覆盖导致

软件商店有时候会报错：

has install snap change in progress 

是由于正在进行安装导致

```
snap changes
```

显示所有的进程

```
abort <the progress number>
```

# 后台管理 

采用 nohup command &

重定向输出文件 ：

nohup python -u gcn_v2_lei/train.py >train.log 2>&1 &  (针对 python 而言， 重定向到 train.log )

# 内核问题

内核更换会导致系统监测不到硬件，比如无线网卡....

查看内核 uname -r

# Linux on windows

file location :

C:\Users\dell\AppData\Local\Packages\CanonicalGroupLimited.Ubuntu18.04onWindows_79rhkp1fndgsc\LocalState\rootfs

# wine

如果遇到依赖问题

```plain
sudo apt install --install-recommends winehq-stable
正在读取软件包列表... 完成
正在分析软件包的依赖关系树
正在读取状态信息... 完成
有一些软件包无法被安装。如果您用的是 unstable 发行版，这也许是
因为系统无法达到您要求的状态造成的。该版本中可能会有一些您需要的软件
包尚未被创建或是它们已被从新到(Incoming)目录移出。
下列信息可能会对解决问题有所帮助：

下列软件包有未满足的依赖关系：
 winehq-stable : 依赖: wine-stable (= 5.0.0~bionic)
E: 无法修正错误，因为您要求某些软件包保持现状，就是它们破坏了软件包间的依赖关系。
```

这是因为一些 i386 的包被卸载了。

```bash
#  Add PPA for the required libfaudio0 library
sudo add-apt-repository ppa:cybermax-dexter/sdl2-backport
sudo apt install wine-stable-amd64
sudo apt install --install-recommends winehq-stable
```

winetrick 安装 xmlx6 失败时，可以直接在命令行末尾安装

```bash
WINEPREFIX=~/.wine/office2013 WINEARCH=win32 winetricks xmlx6
```
