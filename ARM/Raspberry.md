# Raspberry Pi

## Ethernet 连接

用网线连接树梅派和 PC 。在 Ubuntu 上修改连接设置`edit connections` 或者命令行 `nm-connection-editor`。

`sudo arp-scan -l` 查看树梅派 ip 地址。（可能需要用 apt 下载相应软件。）

## Configuration

### UART

`UART0` / `UART1` 作为系统调试串口和蓝牙串口。两者可在两种模式中切换 --------- `primary` 和 `secondary` 。`primary` 与 `GPIO Pin14` 和 `GPIO Pin15` 相连，并且与 `console` 相连，作为调试串口，输出命令行信息，作为一种交互调试工具。`secondary` 与蓝牙相连。在硬件方面，`UART0` 是 PCL001，性能更好，故在默认情况下，作为 `secondary` 与蓝牙连接，作为通讯口。`UART1` 是 `mini UART`，性能更差，默认失能。两者方式转换见 [here](https://www.raspberrypi.org/documentation/configuration/uart.md)

| Mini UART set to | core clock | Result  |
| ------------ | ------------ | ------------ |
| primary | variable | mini UART disabled |
| primary | set `enable_uart=1` | enabled, clock fixed to 250MHz |
| secondary  | variable  | disabled |
| secondary | set `core_freq=25` | enabled |

> 使用 `sudo raspi-config` 后在 `config.txt` 中发现多一行 `enable_uart=1` 说明将 `mini_uart` 作为  `primary` ，作为控制台输出串口。 

`UARTx` （2～5） 含有 `CTS` 与 `RTS` 作为控制信号，可以作为外部通信串口。

> `UART5` 数据口与 `GPIO Pin14` 和 `GPIO Pin15` 相连，但是由于有控制信号，故两者不会冲突。

> `CTS` 与 `RTS` 作为控制信号，表示双方是否准备好接受数据。与数据口 `TX` 与 `RX` 配合使用。

### I2C

可以直接在 `bash` 中配置。输入 `sudo raspi-config` 在 `Interface` --> `I2C` 选项中使能。然后在 `/dev` 下查找相应设备。eg, `i2c-1` 。

可以使用工具包 `i2ctool` 查看设备状态并且读写数据到设备相应的寄存器。需要提供 `I2C` 总线上从设备的地址和设备中寄存器的地址。`I2C` 总线上设备地址可以通过硬件上的引脚拉高拉低配置。

