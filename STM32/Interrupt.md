# ADVANCE

## 中断

stm32 的每个IO口都可以作为外部中断的中断输入口。GPIOx.0 ~ GOIOx.15 分别对应 16 个中断线。需要配置中断线和 GPIO 口（组）的对应关系。

编写一般步骤：

1. 初始化 IO 口作为输入
2. 开启 AFIO 时钟
3. 设置 IO 口与中断线的映射关系
4. 初始化线上中断，设置触发条件 (EXTI_InitStructure)
5. 配置中断服务函数(NVIC), 使能中断
6. 编写中断服务函数

NVIC (中断矢量控制器)

```c
// 将 GPIOE.2 与中断线2 相连
GPIO_EXTILineConfig(GPIO_PortSourceGPIOE, GPIO_PinSource2);

// 配置中断线 2 参数
EXTI_InitTypeDef EXTI_InitStructure;
EXIT_InitStructure.EXTI_Line = EXTI_Line2;
EXIT_InitStructure.EXTI_Mode = EXTI_Mode_Interrupt;
EXTI_InitStructure.EXTI_Trigger = EXTI_Trigger_Falling;
EXTI_InitStructure.EXTI_LineCmd = ENABLE;
EXTI_init(&EXTI_InitStructure);

// 配置 NVIC
NVIC_InitStructure.NVIC_IRQChannel = EXTI2_IRQn; // 相应的中断线
NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 0x02;
NVIC_InitStructure.NVIC_IRQChannelSubPriority = 0x02; 
NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;

NVIC_Init(&NVIC_InitStructure);

// 中断服务程序，与中断线对应

```

> `EXTI_Mode_Interrupt` 和 `EXTI_Mode_Event`, 前者可以引发中断服务程序，后者不行。

## ADC

STM32 将ADC 的转换分为 2 个通道组：规则通道组和注入通道组。规则通道相当于你正
常运行的程序，而注入通道呢，就相当于中断。在你程序正常执行的时候，中断是可以打断你
的执行的。同这个类似，注入通道的转换可以打断规则通道的转换，在注入通道被转换完成之
后，规则通道才得以继续转换。
