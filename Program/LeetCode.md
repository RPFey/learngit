# Algorithm

## leetcode

prob 77 (感觉像是回溯算法)。从后往前递归，每当k位前移一格时，k+1 位要循环一次。(枚举题)

## linked list

prob 2 

感觉还是重新建一个链表更简洁(虽然耗费空间)

prob 23

可以考虑尝试用优先队列，应该比把所有元素都放在一起排序快一些。

prob 25 

链表反序。要求常数空间。最好的方法是直接用三个指针，然后顺着列表往下走。每次操作都是断开，反接。这样只需要 O(n) 时间。有的答案用队列也行，(不是常数空间吗。。。。)

```plain
   <- a[n-1]  ->  a[n]  ->  a[n+1]
        ^          ^          ^
        |          |          |
    previous    current      next

a[n].next = a[n-1]
previous = current
current = next
next = next -> next

   <- a[n-1]  <-  a[n]  ->  a[n+1]  ->  a[n+2]
                   ^          ^           ^
                   |          |           |
                previous   current       next
```

## Hash Table

prob 3 就是构建 ascii 码的字母表，对表查找即可。前后两个指针。如果碰到相同，则移动后者。

prob12 罗马字符转换，建表代码更简洁（与单纯的while, if 相比）

```c++
string intToRoman(int num) {
        string sym[] = {"M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"};
        int val[] = {1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1};
        string ret = "";
        for (int i = 0; num ; i ++){
            while (num >= val[i]){ // while 主要处理 1000~3000， 100~300， 10~30， 1~3
                num -= val[i];
                ret += sym[i];
            }
        }
        return ret;
    }
```

## Array

* Prob4

在两个排序好的数组中找到中位数。中位数满足两个条件:

1. 分割后左右元素数量相同
2. 左侧元素最大值小于右侧元素最小值

在理解这两个条件之后，找中位数就是在 A, B 两个数组中找到这样一种分割

| left_A | right_A |
| :-: | :-: |
| A[0], A[1], ..., A[i-1] | A[i], A[i+1], ..., A[n] |

由于 **A** 有 **n** 个元素，所以有 **n+1** 种分割方法。

对于 A, B 数组而言

| left | right |
| :-: | :-: |
| A[0], ..., A[i-1] | A[i], ..., A[n] |
| B[0], ..., B[j-1] | B[j], ..., B[m] |

此时只需要

$$
\begin{aligned}
  A[k-1] &\leq B[j] \\
  B[j-1] &\leq A[k] 
\end{aligned}
$$

在查找时，可以在 A 中用二分查找法（ B 中的分割为总元素数量的一半减去 A 中的分割），确定这样一种分割。

* Prob15 

基本思想是一个指针固定，另两个指针从头尾开始查找，然后第一个指针后移。这里关键是要去除重复的选项。

```c++
\\ 由于本身排好序了，避免重复，(left,right) 为左，右指针
while(left < right and nums[left] == nums[left-1]) left++;
while(left < right and nums[right] == nums[right+1]) right--;
```

* Prob16 

同样的道理，一个指针固定，另两个指针分别从前后开始查找。如果小了则右移前指针，大了则左移后指针。

其实这两个应该算一类题，列表里面查找三元素的。基本思想都是固定一个指针，另外两个指针从两头开始查找。这样可以达到 O(n^2)

* Prob 113

Problem Definition:

```plain
Given an array of integers, every element appears k (k > 1) times except for one, which appears p times (p >= 1, p % k != 0). Find that single one.
```

1-bit number:

Suppose we have a array of **1-bit** numbers, we'd like to count the number of 1's in the array. We also have a **counter** (binary) so that whenever the counted number of 1 reaches `k`, the counter returns to 0 and starts over. Suppose the counter has `m` bits, `xm`, ..., `x1`.

1. the initial state of the counter is zero
2. if we hit a 0, the counter remain unchanged
3. if we hit a 1, the counter increases by 1
4. the bit number `m`, `2^m>k` --> `m > lg(k)`

## Binary Tree

Prob : 给定 n 个元素，问有多少种不同的二叉树

这个用递归。

$$
    numTree(n) = \sum_{i=0}^{i=n} numTree(i) * numTree(n-i-1)
$$

但是直接递归爆栈了，所以联想到斐波那契数列的计算，采用一个数组改成迭代格式。

$$
    numTree[n] = \sum_{i=0}^{i=n} numTree[i] * numTree[n-i-1]
$$

以此不断延伸数组，直到 n 个。
