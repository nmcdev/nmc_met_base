# GitHub多人协作指南
 
## 前期准备：
创建SSH Key连接GitHub
这里就直接看廖老师的[教程](https://www.liaoxuefeng.com/wiki/0013739516305929606dd18361248578c67b8067c8c017b000/001374385852170d9c7adf13c30429b9660d0eb689dd43a000)
[也是教程](https://www.liaoxuefeng.com/wiki/0013739516305929606dd18361248578c67b8067c8c017b000)

## [小组协作](https://www.liaoxuefeng.com/wiki/0013739516305929606dd18361248578c67b8067c8c017b000/0013760174128707b935b0be6fc4fc6ace66c4f15618f8d000)

### 第一步： 
首先每个小组成员，在自己本地建立一个目录，作为工作空间，再去git clone 这个远程仓库：
```sh
git clone https://github.com/nmcdev/demo0.git
```
### 第二步： 
一般的，小组成员需要建立属于自己的分支，每个分支代表着开发不同的功能
```sh
git branch jiantest//创立一个名字叫jiantest的分支
git branch //查看分支  你会看到：
         *master
            Jiantest
```
这表示，你有两个分支，一个master(正在使用)，还有一个新建的jiantest分支
### 第三步： 
一般都是，小组成员切换到自己分支里进行开发，而不要用master进行开发
```sh
git checkout jiantest //切换到jiantest分支
//然后进行一顿开发操作，开发工作结束之后
git add . //保存所有操作
git commit -m "what did you do" //提交所有操作
```
### 第四步： 
master是主分支，要与远程保持同步，所以我们的开发分布不要直接推送到远程， 
应该先在本地和master合并，再推送到远程
```sh
git checkout master //切换到主分支
git merge jiantest //合并分支里的操作
git push
```
## 补充:
一般的团队协作模式是这样的：
* 通过上面的步骤推送的自己的修改
* 如果推送失败，说明远程master 已经更新过了，我们需要git pull 尝试合并
* 如果合并有冲突，我们需要再本地解决冲突，提交。然后再去推送

## 参考：
https://zhuanlan.zhihu.com/p/23478654
http://xiaocong.github.io/blog/2013/03/20/team-collaboration-with-github/
https://segmentfault.com/a/1190000015798490
https://www.worldhello.net/gotgithub/04-work-with-others/010-fork-and-pull.html
