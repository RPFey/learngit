# 廖雪峰教程

<https://www.liaoxuefeng.com/wiki/896043488029600/896827951938304>

```
git add <file name> 提交文件
git commit -m "...<说明>" 上传到仓库
git log 查看版本历史
会显示：

commit 0b2ebef8cedd4151bb2e09eb6781545289a3c531
Author: peter <1023842624@qq.com>
Date:   Sun Jul 14 11:34:51 2019 +0800

    update git.md

commit 3af8a6f60c214b07a92b58d2b952a1d5ec50691c
Author: peter <1023842624@qq.com>
Date:   Sun Jul 14 11:27:46 2019 +0800

    update linux problem .md

```

comiit 之后的是版本库的id 

```ascii
┌────┐
│HEAD│
└────┘
   │
   └──> ○ append GPL
        │
        ○ add distributed
        │
        ○ wrote a readme file
```

一个HEAD 的指针指向不同的版本库



git 可以在不同版本之间切换 

仓库里面的文件会因此改变（电脑里面的也会。。。）

```
git reset --hard HEAD^  返回上一次
git reset --hard <地址>  // 比如 3af8a6
```

当然，也可以使用

```
git reflog // 查看之前的命令 甚至在电脑重启后
```

