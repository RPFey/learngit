# RT-Thread

## [线程管理](https://www.rt-thread.org/document/site/programming-manual/thread/thread/)

线程是 RT-Thread 中基本的调度单位。

![线程](https://www.rt-thread.org/document/site/programming-manual/thread/figures/04Object_container.png)

* 线程创建与删除

线程创建时会从内核容器中分配线程对象，线程删除时会从对象容器中删除。每个线程都有自己的属性（控制块，线程栈，入口函数）。此属性由`struct rt_thread` 描述。

* 线程调度

线程间切换时，上一个线程的上下文（局部变量，寄存器值）放入线程堆栈中。等优先级高线程运行结束后，再从线程堆栈中恢复执行。

* 线程状态

`初始状态`、`就绪状态`、`运行状态`、`挂起状态`、`关闭状态`

* 时间片

对于相同优先级的线程，采用时间片轮转的方式执行。`OS Tick`定义为一个单位时间片段。相同优先级的线程在轮流执行自己时长后挂起。

* 线程入口函数

1. 无限循环模式：此类模式注意编写挂起的操作。避免由于死循环导致低优先级的线程无法执行。一般使用延迟函数或者主动挂起。
2. 顺序或者有限执行


