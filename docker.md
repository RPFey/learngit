# DOCKER

## ssh connection

创建容器时，需要`-p [host_port]:[container_port]` 建立端口映射。

运行容器后，需要

```bash
apt-get install openssh-server openssh-client
service ssh start
```

在 `/etc/ssh/sshd_config` 修改配置，允许 `root` 用户登陆。修改

```plain
PermitRootLogin yes
```

并在 bash 下用 `passwd` 修改密码。

## NVIDIA

需要安装 `nvidia-docker2` ，[教程](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)

主机上需要安装 nvidia 驱动 `nvidia-driver-*` (版本号 440, 450, ...)

之后在 nvidia docker hub 上拉取相应版本镜像，直接在本地部署即可。