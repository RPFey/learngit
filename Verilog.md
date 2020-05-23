# 数字电路

## 布尔代数

异或门与同或门：

同或：

x | y | z 
:-:|:-:|:-:
0 | 0 | 1 
1 | 0 | 0 
0 | 1 | 0 
1 | 1 | 1 

布尔逻辑式化简(最小项与最大项)。与项(乘积)是越来越小，称其为最小项；反之或项(加和)为最大项。最小项之积，最大项之和。

# 模块化设计与模板

模块间的调用

```Verilog
// 顶层模块，调用　T 触发器
module counter(q, clk, reset);

output[3:0] q;
input clk,reset;

TFF tff1(q[0], clk, reset);

endmodule;

module TFF(q, clk, reset);

output q;
input clk, reset;

D_FF (q, clk, reset);  // 调用　D 触发器

endmodule;
```

激励块

```Verilog
module stimulus;

reg clk;
reg reset;
wire[3:0] q;

// 引用设计模块
counter r1(q,clk,reset);

// 驱动时钟信号
initial
	clk = 1'b0;
always
	#5 clk = ~clk;

initial
begin
	reset = 1'b1;
	#15 reset = 1'b0;
	#180 reset = 1'b1;
	#10 reset = 1'b0;
	#20 $finish // 停止仿真
end

// 监视输出
initial
	$monitor($time, " Output q = %d", q);

endmodule
```

# 组件

## Module & component

```Verilog
module FullAdder(a, b, cin, sum, cout);
	input a, b, cin;
	output sum, count;

	wire a, b, cin;
	wire sum, cout;

	assign sum = (a^b)^c;
	assign cout = cin&(a^b)|(a^b);

endmodule

module TOP;
	reg A, B, CIN;
	wire SUM, COUT;
	
	FullAdder(.sum(SUM), .cout(COUT), .a(A), .b(B), .cin(CIN));
// 指定输入参数间的对应关系, 模块内输入不能为 reg, 但是模块外给模块输入的器件可以是　reg
endmodule
```
有 input \ output \ inout , 端口默认数据类型为 wire, 只有输出能改成 reg 类型。

### always module

**always** 模块: 当敏感信号中信号发生变化时，执行 always 到 end 为止。
>敏感信号分　电平敏感；边缘敏感：posedge \ negedge 描述上升沿与下降沿。采用　always@ * 表示所有信号。
>过程赋值只能用在 always \ initial 模块中。阻塞赋值与Ｃ相同，非阻塞赋值则是在always 结束后同时赋值

```Verilog
module eq1(
	input i0, i1,
	output reg eq);
	reg p0, p1;

	always@{i0, i1} // 敏感信号列表，当i0, i1 变化时此模块激活
	begin
		p0 = ~i0&~i1;
		p1 = i0&i1;
		eq = p0 | p1; // 此语句执行前 p0, p1 必须赋值
	end
endmodule
```
常见的错误:

1.多个 always 块中赋值
```Verilog
always@*
	if(clear) y =1'b0;
always@*
	y=a&b;
```
<font color='#ff0000'>不要在多个always块中给同变量赋值</font>, 物理上每个 always 块为电路的一部分，这里表示y是每个电路的输出，并且可以同时更新。

2.不完整分支和不完整输出赋值。 *所以最好是预先赋初始值*


## 逻辑控制

casex 或 casez, 认为 ? 为无关值
```Verilog
// HDL 编码器
module encoder_casez
( input [3:0]r,
  output reg [2:0] y );
	always@ *
		casez(r)
		4'b1???:y = 3'b100  // 首位为１
		... //
		endcasez
endmodule
```

## 数字 与 数据类型

wire : 用 assign 指定信号，wire[n-1:0] a,  定义位宽为n.
```Verilog
// FPGA 中设计可以指定位宽
wire [7:0] bus; // 8 位总线

reg [0:40] addr; // 地址
//　其抽象为向量
bus[2:0] // slice

addr[0:1] // 注意两者高地位的区别
```

### wire vs. reg

**wire代表直通,只要输入有变化，输出立刻变化**,wire 相当于一条直线，用来连接电路,**不能驱动**,**不能存储数据** 。wire 只能在 assign 的左侧赋值，且是立即赋值。

reg 表示只有触发才会反映输入。 **只能在always或initial中赋值** ,上升沿或者下降沿(锁存器)，高电平或者低电平(锁存器)

输入端口可以是由reg/wire`驱动`，输入端口只能是wire.	输出端口可以是wire/reg, 只能`驱动`wire。 

1.\<size>'\<format>\<number> size 指明了数字的位宽度
2.常数量(localparam)： 尽量使用常数变量代替字面值, **localparam不会综合出物理电路**
3.参数： 类似函数的传入参数

```Verilog
module adder
	#(parameter N = 4) \\ 默认值是4
	...
endmodule

module adder_insta (...)
	adder #(.N(8)) unit1 (...);
endmodule
```

## Template

1.常数加法器

```Verilog
module adder
(
	input wire[3:0] a,b;
	output wire [3:0] sum,
	output wire cout
);
	localparam N = 4,
			   N1 = N-1; \\ 使用 localparam 方便修改参数
	wire[N:0] sum_ext; \\ use assign for wire type 
	assign sum_ext = {1'b0, a} + {1'b0, b}; \\ {} concatenates the value by bit.
	assign sum = sum_ext[N1:0];
	assign sount = sum_ext[N:0]
endmodule 
```

2.多路选择器

<img src=./img/Verilog/MUX.png width="300" height="300" align=center />

s1 | s0 | out 
:-:| :-: | :-: 
0 | 0 | in1
0 | 1 | in2 
1 | 0 | in3 
1 | 1 | in4 

```Verilog
module mux41_if
(
	input in0,in1,in2,in3,
	input s0,s1,
	output reg out
);

always@*
	begin 
		if({s1,s0} == 2'b00)			out = in0;
		else if({s1,s0} == 2'b01) 		out = in1;
		else if({s1,s0} == 2'b10)		out = in2;
		else 							out = in3;
	end
endmodule
```
多路选择器是多输入，单输出。从n个输入通道中选择一个输出到公共输出端

3.比较器

```Verilog
module comp_N
	#(paraneter N = 8)
	(
		input[N-1:0] in0, in1,
		output reg gt,eq,lt
	)
	always@*
		begin
			gt = 0;
			eq = 0;
			lt = 0;
			if(in0>in1)
				gt = 1;
				if(in0 == in1)
					eq = 1;
				if(in0<in1)
					lt=1;
			end
endmodule		
```

## 时序电路设计

### 复位D触发器

对于基本的D触发器，对上升沿信号的分析。en代表使能，reset复位，clk时钟信号。只有当时间信号在上升沿且使能时，q获得d的值。本例作为对上升沿捕获信号的例子。

| reset | clk  | en   | q    |
| ----- | ---- | ---- | ---- |
| 1     | -    | -    | 0    |
| 0     | 0    | -    | q    |
| 0     | 1    | -    | q    |
| 0     | up   | 0    | q    |
| 0     | up   | 1    | d    |



```Verilog
module dff_reset_en
    (
        input clk, reset,
        input en,
        input d,
        output reg q
    );
    
    always@(posedge clk, posedge reset)
        begin
            if(reset)
                q <= 1'b0;
            else if (en)
                q <= d;
        end
endmodule
```

### 清零锁存器

当clk为高电平时，q得到d的输入，否则保持自己的值。

```verilog
module latch_reset
    (
        input clk,reset,
        input d,
        output reg q
    )
    always@(clk, reset, d)
        if(reset)
            q <= 1'b0;
    	else if(clk)
        	q <= d;
    end
endmodule
```

