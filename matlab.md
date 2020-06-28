<font face="Apple Symbol" size="3.5">

# Matlab

## basics

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

## Analog circuit

## Signal Analysis

construct the system function :

$$
H(z) = \frac{1}{1 + \frac{5}{6} z^{-1} + \frac{1}{6} z^{-2}}
$$

```matlab
b = [1 0];
a = [1 5/6 1/6]; % the coefficients of the polynomial
zplane(b,a); % pole-zero-plot
figure
freqz(b,a); % freq response
figure
implz(b,a);
```

Given the system function

$$
H(z) = 1 - z^{-N}
$$

其零点沿着单位圆 N 等分

```matlab
N = 6;

w = [0:1:500]*2*pi*
b = [1 0 0 0 0 -1];
a = [1];
zplane(b, a);
```

</font>
