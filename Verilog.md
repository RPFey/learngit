# 数字电路

## 布尔代数

异或门与同或门：

同或：

x | y | z |
:-:|:-:|:-:|
0 | 0 | 1 |
1 | 0 | 0 |
0 | 1 | 0 |
1 | 1 | 1 |

布尔逻辑式化简(最小项与最大项)

与项(乘积)是越来越小，称其为最小项；反之或项(加和)为最大项。最小项之积，最大项之和。

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

always 模块: 当敏感信号中信号发生变化时，执行 always 到 end 为止。

敏感信号分　电平敏感；边缘敏感：posedge \ negedge 描述上升沿与下降沿。采用　always@ * 表示所有信号。

过程赋值只能用在　always \ initial 模块中。阻塞赋值与Ｃ相同，非阻塞赋值则是在always 结束后同时赋值

```Verilog
module eq1(
	input i0, i1,
	output reg eq);
	reg p0, p1;

	always@{i0, i1} // 敏感信号列表，当i0, i1 变化是时此模块激活
	begin
		p0 = ~i0&~i1;
		p1 = i0&i1;
		eq = p0 | p1; // 此语句执行前 p0, p1 必须赋值
	end
endmodule
```

## 逻辑控制

casex或casez, 认为? 为无关值

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

\<size>'\<format>\<number> size 指明了数字的位宽度

```Verilog
// FPGA 中设计可以指定位宽
wire [7:0] bus; // 8 位总线

reg [0:40] addr; // 地址
//　其抽象为向量
bus[2:0] // slice

addr[0:1] // 注意两者高地位的区别
```
