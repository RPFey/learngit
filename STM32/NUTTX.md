# NUTTX

## 配置

1. 在目录中使用 `make menuconfig`
2. 在 `boards/` 文件夹下使用 `configs/` 下的 `Kconfig` 文件

生成的配置选项在 `include/nuttx/config.h` 中以宏定义形式存在，用于设置芯片功能。

## 启动流程

上点电后芯片复位，复位流程包括

1. 设置处理器
2. 设置时钟供外设使用
3. 设置堆栈指针
4. 启动 NUTTX

* 内存初始化

分为两类。定义时已经初始化的变量，在内存中用`.data` 标记。未显式初始化的变量，（赋默认值0）用`.bss` 标记。

* Example STM32 F4 复位

在 `ld.script` 中，所有的中断向量 （`.vectors`）放在了 `FLASH` 最初，（复位后 `PC` ）指向位置。

```plain
SECTIONS
{
    .text : {
        _stext = ABSOLUTE(.);
        *(.vectors)
        *(.text .text.*)
        *(.fixup)
        *(.gnu.warning)
        *(.rodata .rodata.*)
        *(.gnu.linkonce.t.*)
        *(.glue_7)
        *(.glue_7t)
        *(.got)
        *(.gcc_except_table)
        *(.gnu.linkonce.r.*)
        _etext = ABSOLUTE(.);
    } > flash
```

从 `/arch/arm/src/arm/arm_vectortab.S` 中可以看出，复位后 `PC` 指向 `__start`。即复位的入口函数在 `__start` （`/arch`）。在编译时 `ld` 明确命令 `--entry=__start` 。

`stm32_bringup` （位于 `board/src` ） ，用于配置底层的硬件设备接口。

## 硬件初始化

### GPIO

底层的 GPIO 配置采用的是
