# ARM 汇编

```bash
as -o ${file_name}.o ${file_name}.s
ld -o ${file_name} ${file_name}.o
# 查看编译的机器码
objdump -s -d ${file_name}.o 
```

## 处理器原理

### 存储器组织

* 标准 ARM 字为32位长
* 每个字划分为4个8位字节

### LOAD & STORE 结构

数据操作必须先被装入到 CPU 中，运算后再存回存储器保存结果。

### CPSR

四个状态位（NZCV），用于存储带符号数的补码运算结果状态

Z：如果结果为全0，则为1。

N：如果最高位（符号位）为1（结果为负数）则为1。

C：最高位（符号位）产生进位，为1。

V：溢出为1。（双高位判别法）

## 指令

### LDR&STR&ADR

寻址方式：

* 直接寻址

```assembly
    LDR r0, #0DB3
```

* 寄存器间接寻址

```assembly
    ; 使用寄存器
    LDR r0, [r1]
    ; 寄存器与偏移
    LDR r0, [r1, -r2];  装入 r1-r2 地址处的值
    ; 从变量处获取（类似于取地址 &）
    ADR r1, FOO
    LDR r0, [r1]; 得到 FOO 的值
```

* 自动变址与后变址

```assembly
    LDR r0, [r1, #16]!; 先偏移再取值
    LDR r0, [r1], #16; 先取值再偏移
```

### 分支转移

`BL` 用作跳转指令，跳转到标记符处。同时会把跳转时`PC` 的值复制给`lr` 寄存器（`r14`），以供函数结束能程序能返回原来的位置。如果函数调用嵌套的话，需要将`r14` 值放入堆栈中保存。

`B{XX}` 用作判断转移指令。从 `CPSR` 获取运算结果的状态，完成条件转移。 IF 条件：

```assembly
    CMP r0, r1
    BGE fblock
    ; the r0 < r1 block here


    ; the r0 >= r1 here
    fblock
```

while 语句

```assembly
	MOV r0, #9; 设定循环次数
loop
	; main loop for while block
	CMP r0
	BL loop
; 循环结束
```

### MOV & MOVT

前者操作低16位，后者操作高16位。可以使用：

1. `#`赋值立即数。
2. `Operand2` 例如：`MOV R1, R2, LSL #1` 

> 这是由于在进入 ALU 之前， Operand2 先经过移位器计算，再送入 ALU 单元。

> 立即数只能低于 16 位，高于 16 位采用 `MOV` 和 `MOVT` 同时对高低位赋值。

`MOVN` 可以直接作为取反操作。取反，加一相当于乘 -1。

### ADD / ADC

三操作数，源寄存器，加数1，加数2。加数二可以是操作数，立即数和寄存器。`ADC` 是带进位加，用在数值大于 32 位时。指令尾部的 `S` 代表是否修改 `CPSR`。

## 分支转移