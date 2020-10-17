[TOC]

# STM32

## hardware setup

VDD digital power supply (positive digital input)

VSS GROUND (pin ground) 两者之间加入电容 一个是滤波(电源不稳定)

ST-LINK v2

SWCLK (clock) ; SWDIO (data input & output) ; power supply (for the micro chip)  you can find the corresponding pins on stm32 chip (SWCLK SWDIO)

SWCLK pull low externally (link to GND); SWDIO pull high

for JTAG , pins are same (正点原子 STM32F1 的板子)

![avatar](./img/JTAG.png)

you can also find SWDIO is pulled up while SWCLK is pulled down.

### Degub protoptype (JTAG / SWD)

<https://blog.csdn.net/LEON1741/article/details/72846434>

JTAG 是芯片内部的测试协议。标准的JTAG接口是4线：TMS、 TCK、TDI、TDO，分别为模式选择、时钟、数据输入和数据输出线。 相关JTAG引脚的定义为：

TMS：测试模式选择，TMS用来设置JTAG接口处于某种特定的测试模式；

TCK：测试时钟输入；

TDI：测试数据输入，数据通过TDI引脚输入JTAG接口；

TDO：测试数据输出，数据通过TDO引 脚从JTAG接口输出；

SWD，串行调试(Serial Wire Debug), 更少的引脚(4 个), 大数据下更稳定。

![avatar](img/JTAGandSWD.jpg)

> 可以看到对上面正点原子的板子， SWD 与 JTAG 共用了一个口

SWD 相比于 JTAG 使用接口更少，只需要 SWDIO, SWCLK, VCC, GND 四根线(stm32f103-minimum　上的四个借口)

### 仿真器

jlink

基于 JTAG 仿真，面对计算机采用 USB 口， 对板仍然采用 JTAG 口

ST - link

专门针对 ST 公司的芯片， 可以烧写，仿真，下载。 st-link v2 对于 JTAG 和 SWD 是通用的，在 Keil 可以随意选择。

### 烧写

* openocd

参见 nuttx 中 openocd 安装与烧写方法

```bash
openocd -f interface/stlink-v2.cfg -f target/stm32f1x.cfg -c init -c "reset halt" -c "flash write_image erase nuttx.bin 0x08000000"
```

openocd 也可以用来和芯片相连接，提供调试的本地端口。

```bash
openocd -f interface/stlink-v2.cfg -f target/stm32f1x.cfg
```

openocd 配置文件在 `/usr/local/share/openocd/scripts` 下。`interface` 对应的是调试器硬件 (stlink-v2, ...) `target` 对应是板子

* stlink

使用 stlink 驱动烧写

注意烧写时的 BOOT 引脚设置。

| BOOT0 | BOOT1 | 启动方式 |
| :-: | :-: | :-: |
| 0 | 无关 | 用户闪存存储器启动 (User Flash memory) |
| 1 | 0 | 从系统存储器启动 (System memory) |
| 1 | 1 | 从内嵌SRAM 启动 (Embedded SRAM) |

* User Flag memory

正常工作模式

* System Memory

系统存储器是芯片内部一块特定的区域， ST 在出厂时，在内部预置了一段 bootloader, (ROM)。在 bootloader 帮助下，通过串口下载程序到 flash 中。

* Embedded SRAM

该模式用于调试代码（不需要全部擦除从来）

> 2020.5.30 晚上一直不清楚 stlink-v2 通过 openocd 无法连接芯片的原因。后来发现是 Boot 接线不对，导致驱动检测不到芯片。

若从主闪存启动，从地址 0x0800 0000 访问，这也就是通过 openocd 烧写指定的地址。

内置 SRAM 是在 0x2000 0000

### Debug

目前调试可以用 openocd 完成。注意在启动文件中修改烧写文件区域的大小。

```bash
arm-none-eabi-gdb nuttx # program name

# gdb interface
target remote localhost:3333 # connect to the localhost port provided by openocd
monitor reset # reset chip
monitor halt  # halt the program

load nuttx    # load binary file
```

`monitor` 是本地远程发送到芯片的命令。

## communication

### UART

异步收发串行通信  <--> 同步方式 (USART)

规范标准 ： RS232. RS449

COM 是 PC 上异步串行通信口 RS232

![UART](./img/UART.png)

开始时保持低电平是为了保证数据线正常。以低电平表示开始传输数据。UART 在接受完成后，将设置标记表示数据可用( 中断处理函数中常作检验 )，并产生一个处理器中断。

### CAN

CAN 协议通过以下五种数据帧通讯。

## chip Reg

### CLOCK

RCC (Reset Clock Control)

AHB (Advanced High performance Bus)

### GPIO

MODER (mode register)

OTYPER (output type register)

## IDE

## Program

stmf4xx.h has defines about registers

```c++
RCC -> AHBENR |= RCC_AHBENR_GPIOCEN ; // or (1 << 19) specify the bits

// set zero

GPIOC -> OTYPER &= ~(...)
```

### ASSEMBLY CODE

eg. inline assembly code

```c
int x = 1, y = 2 ;
int res = 0;
__asm ("ADD %[result], %[input_x], %[input_y]"
    : [result] "=r" (res)
    : [input_x] "r" (x), [input_j] "r" (y)
)
```

格式为

```c
stem Type > STM32 Peripheral Support 中确定使用的外设

\> System Type > Alternate Pin Mapping 中确定重映射关系

### PWM

在 boards 下 stm32f103-minimum config 中配置 pwm Timer 和 output channel 。在 stm32f103-minimum.h 中确认定义的通道和定时器与 config 中一致

\> System Type > Timer Configuration 配置定时器的使用模式，输出通道 

> help 键可以查询配置文件来源。关于模式等详细说明在 help 中查找。
