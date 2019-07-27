# 廖雪峰教程

<https://www.liaoxuefeng.com/wiki/896043488029600/896827951938304>

## 注释

```
<...>
```

表示需要自己添加的参数

## 上传文件

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

## 版本切换

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

在 .git/logs/refs/heads/master 中也可以查看命令

## 分区

git add 是将文件添加到暂存区

git commit 是将文件提交到当前分支

当前目录是工作区

使用命令：

```
git status 
```

有 modified （修改） 与 untracked （新文件尚未提交）

前者是需要提交到分支 ， 后者需要先提交到暂存区（git add）

注意： git commit 只会将暂存区的文件提交到分支，而不是工作区的，所以每次修改后都必须

git add ..

git commit -m ...

## 修改

```
git checkout -- <file> // -- 很重要
```

将工作区的文件恢复到最近一次 git add 或者 git commit 时的状态

删除（也是一种修改）

如果错误删除了

可以采用 git checkout 从版本库中恢复

而如果真的要从版本库中删除，可以

```
git add rm <file>
git commit -m "..."
```

从版本库中删除

## 远程仓库 （github）

```
ssh-keygen -t rsa -C "youremail@example.com"
```

在用户主目录里找到`.ssh`目录，里面有`id_rsa`和`id_rsa.pub`两个文件，这两个就是SSH Key的秘钥对，`id_rsa`是私钥，不能泄露出去，`id_rsa.pub`是公钥，可以放心地告诉任何人。

github 通过公钥（SSH 协议）判断是否推送

添加远程库：

```
git remote add origin git@github.com:RPFey/<repository name>.git
```

这里 origin 是远程库的名字，可以自己修改



也可以将origin 删除（解除关联）

```
git remote rm origin
```



```
git push -u origin master
```

将分支master 推送到 远程库 origin

-u 是个参数

## 分支

![git-br-dev-fd](/home/peter/图片/0)

在同一条时间线上的不同指针

查看分支：`git branch`

创建分支：`git branch <name>`

切换分支：`git checkout <name>`

切换分支时要求当前工作区内的修改必须提交，也可以用

```
git stash  保存当前工作区后可以切换分支
处理完切换回来后，采用：
git stash pop 
或者, 例如
git stash apply stash@{0} /* 需要 git stash list 查看*/ 
git stash drop 
```

创建+切换分支：`git checkout -b <name>`

合并某分支到当前分支：`git merge <name>`

删除分支：`git branch -d <name>`

位于当前分支下只能看到当前分支下的内容（相当于一个电脑中的不同用户？）

但是！

当分支变成如下图时，就会有冲突：

![git-br-feature1](/home/peter/图片/1)

此时就需要修改一只分支使其能重合:

![git-br-conflict-merged](/home/peter/图片/3)

无非就是向前走了一步

```
git log --graph --pretty=oneline --abbrev-commit
```

可以查看分支图

当然，上述问题是由于 fast forward 设置存在

可以在合并时关闭

```
git merge --no-ff -m "..." <branch name>
```

