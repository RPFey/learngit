# Algorithm

## leetcode

prob 77 (感觉像是回溯算法)。从后往前递归，每当k位前移一格时，k+1 位要循环一次。(枚举题)

## linked list 

prob 2 感觉还是重新建一个链表更简洁(虽然耗费空间)

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

Prob15 基本思想是一个指针固定，另两个指针从头尾开始查找，然后第一个指针后移。这里关键是要去除重复的选项。

```c++
\\ 由于本身排好序了，避免重复，(left,right) 为左，右指针
while(left < right and nums[left] == nums[left-1]) left++;
while(left < right and nums[right] == nums[right+1]) right--;
```

