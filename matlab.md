# Matlab in Robotics

variable, Vector

```matlab
x = -2:0.1:2 ;
% startValue:spacing:endValue (up to the endValue, but may not include)
```

function

```matlab
y = fun(x)
% multiple inputs and outputs
```

Use bool value to index

```matlab
l = a < 0.005
% element-wise logical operation

s = a(l)
% use logical array to index
```

for loop
```matlab
y(1) = 1
for n = 1:6
	y(n+1) = y(n) - 0.5 * y(n)
end
% automatically append the value
```
