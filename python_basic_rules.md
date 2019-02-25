# Python编码规范

FROM https://blog.csdn.net/sinat_38682860/article/details/80368241

## 本文目的
  * 动态语言在初期开发比较爽，但是到后期维护起来比较困难。
  * Python 作为动态语言之一，自然也会有这样的缺点。所以需要严格的遵守一组规范。
  * 不以规矩不成方圆，规范自然是十分重要的，而在动态语言中，尤其重要。

## 本文参考
 * PEP-8（请重点阅读）
  * https://www.python.org/dev/peps/pep-0008/
  * https://zhuanlan.zhihu.com/p/31212390
  * https://www.jianshu.com/p/52f4416c267d
  * https://www.jianshu.com/p/e132bea1d2c9
 * 浮生若梦的编程
  * https://juejin.im/post/5afe94845188254267264da1
 * Google Python Guide 
  * https://zh-google-styleguide.readthedocs.io/en/latest/google-python-styleguide/
 * 让 Python 代码更易维护的七种武器
  * https://zhuanlan.zhihu.com/p/56585808

## 适用范围 & 原则
  * Python 2.7 - Python 3.x 。
  * 以 PEP 8 为蓝本，紧紧团结在 PEP 8 周围。任何非官方的文档，只是参考之。
  * PEP 8 已经有了的，就不要重复了。

## 规范
### 【强制 + 强制】【挑选并使用静态检查工具】
一开始就要使用，并且从严使用：
  * Pylint
  * Flake8
  * pytest

### 【强制】【文件编码 & Unicode & License】
PS：下面这几条，能帮你避免很多无聊的编码解码问题，所以我觉得很重要
  * 使用 4 空格缩进，禁用任何 TAB 符号
  * 源码文件使用 UTF-8 无 BOM 编码格式
  * 总是使用 Unix \n 风格换行符
  * 在每一个 py 文件头，都添加如下内容：

```sh
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2019 NMC Developers.
# Distributed under the terms of the GPL V3 License.
```

### 【强制】【命名】
* class，function等命名，严格照着PEP8：
* 模块
模块尽量使用小写命名，首字母保持小写，尽量不要用下划线(除非多个单词，且数量不多的情况)
```sh
# 正确的模块名
import decoder
import html_parser

# 不推荐的模块名
import Decoder
```

* 类名
类名使用驼峰(CamelCase)命名风格，首字母大写，私有类可用一个下划线开头
```sh
class Farm():
    pass

class AnimalFarm(Farm):
    pass

class _PrivateFarm(Farm):
    pass
```
将相关的类和顶级函数放在同一个模块里. 不像Java, 没必要限制一个类一个模块.

* 函数
函数名一律小写，如有多个单词，用下划线隔开
```sh
def run():
    pass

def run_with_env():
    pass
```
私有函数在函数前加一个下划线_
```sh
class Person():

    def _private_func():
        pass
```
* 变量名
变量名尽量小写, 如有多个单词，用下划线隔开
```sh
if __name__ == '__main__':
    count = 0
    school_name = ''
```    
* 常量采用全大写，如有多个单词，使用下划线隔开
```sh
MAX_CLIENT = 100
MAX_CONNECTION = 1000
CONNECTION_TIMEOUT = 600
```
* 常量
常量使用以下划线分隔的大写命名
```sh
MAX_OVERFLOW = 100

Class FooBar:

    def foo_bar(self, print_):
        print(print_)
```

* 全局变量
全局变量，一般是常量，我们认为：凡是全局的，都是常量，应该始终使用全大写，如：
```sh
GLOBAL_PUBLIC = "G1"
_GLOBAL_PRIVATE = "G2"

class Person:
    _GLOBAL_IN_CLASS = 'G3'
```

### 【强制】【注释】
1、注释
1.1、块注释
“#”号后空一格，段落件用空行分开（同样需要“#”号）
```sh
# 块注释
# 块注释
#
# 块注释
# 块注释
```
1.2、行注释
至少使用两个空格和语句分开，注意不要使用无意义的注释
```sh
# 正确的写法
x = x + 1  # 边框加粗一个像素

# 不推荐的写法(无意义的注释)
x = x + 1 # x加1
```
1.3、建议
•	在代码的关键部分(或比较复杂的地方), 能写注释的要尽量写注释
•	比较重要的注释段, 使用多个等号隔开, 可以更加醒目, 突出重要性
```sh
app = create_app(name, options)

# =====================================
# 请勿在此处添加 get post等app路由行为 !!!
# =====================================

if __name__ == '__main__':
    app.run()
```    

2、文档注释（Docstring）
作为文档的Docstring一般出现在模块头部、函数和类的头部，这样在python中可以通过对象的__doc__对象获取文档.
编辑器和IDE也可以根据Docstring给出自动提示.
文档注释以 """ 开头和结尾, 首行不换行, 如有多行, 末行必需换行, 以下是Google的docstring风格示例
```sh
# -*- coding: utf-8 -*-
"""Example docstrings.

This module demonstrates documentation as specified by the `Google Python
Style Guide`_. Docstrings may extend over multiple lines. Sections are created
with a section header and a colon followed by a block of indented text.

Example:
    Examples can be given using either the ``Example`` or ``Examples``
    sections. Sections support any reStructuredText formatting, including
    literal blocks::

        $ python example_google.py

Section breaks are created by resuming unindented text. Section breaks
are also implicitly created anytime a new section starts.
"""
```
•	不要在文档注释复制函数定义原型, 而是具体描述其具体内容, 解释具体参数和返回值等
```sh
#  不推荐的写法(不要写函数原型等废话)
def function(a, b):
    """function(a, b) -> list"""
    ... ...

#  正确的写法
def function(a, b):
    """计算并返回a到b范围内数据的平均值"""
    ... ...
```
•	对函数参数、返回值等的说明采用numpy标准, 如下所示
```sh
def func(arg1, arg2):
    """在这里写函数的一句话总结(如: 计算平均值).

    这里是具体描述.

    参数
    ----------
    arg1 : int
        arg1的具体描述
    arg2 : int
        arg2的具体描述

    返回值
    -------
    int
        返回值的具体描述

    参看
    --------
    otherfunc : 其它关联函数等...

    示例
    --------
    示例使用doctest格式, 在`>>>`后的代码可以被文档测试工具作为测试用例自动运行

    >>> a=[1,2,3]
    >>> print [x + 3 for x in a]
    [4, 5, 6]
    """
```    
* 文档注释不限于中英文, 但不要中英文混用
* 文档注释不是越长越好, 通常一两句话能把情况说清楚即可
* 模块、公有类、公有方法, 能写文档注释的, 应该尽量写文档注释


### 【强制】【强化private的概念】
即：最小知识原则，对外暴露的东西越少越好

翻译成大白话就是：
* 实例属性，一般定义成private的
* class，对外提供的方法越少越好
* module，对外提供的接口越少越好
* package，对外提供的 module 越少越好

翻译成代码就是：
1. 项目布局
```sh
package/
    __init__.py
    _private_mod.py
    public_mod.py    
```

2. 某模块内容
```sh
public_mod.py
PUBLIC_GLOBAL = 'G1'
_PRIVATE_GLOBAL = 'G2'
class _Class:
    pass
class PublicClass:
    _PRIVATE_GLOBAL = 'G3'
    
    def __init__(self, name,age):
        self._name = name
        self._age = age
    def public_method(self):
        pass
    def _private(self):
        pass
```

所有东西，一开始就要定义成私有的，等到确实需要开放访问了，才开放出去。

### 【强制&重要】【关注公开接口的复杂性】
最好的接口是这样的，调用者无脑使用
```sh
def interface():
    pass
```    
次等接口是这样的
```sh
def interface(param1):
    pass
```    
    
次次等接口是这样的
```sh
def interface(p1, p2):
    pass
```

最大忍受限度的接口是这样的
```sh
def interface(p1, p2, p3='SOME DEFAULT'):
    pass
def interface(p1, *args):
    pass
```

不可接受的接口是这样的
```sh
def interface(p1, p2, **kwargs):
    pass
```

令人无语的接口是这样的
```sh
def interface(*args, **kwargs):  
```

尽量不要使用
```sh
**kwargs
```
某些流行库有这样的毛病，极大地增加了调用者的心理负担，反映了接口设计者的懒惰

### 【推荐】【以package去设计命名空间，而不是基于module】
### 【推荐】【了解如下内容】
```sh
__init__.py 的作用

__main__.py 的作用

if __name__ == '__main__': 的作用
```
Python的命名空间加载机制，即：sys.path sys.modules 的内容

### 【推荐】【合理设计项目目录结构】
```sh
project
    project/
        __init__.py
        core/
        utils/
        constants/
        
        
        __main__.py
        
    tests/
    docs/
    examples/
    README.md
    .pylintrc
    .flake8
```    

### 【其他注意问题 】
* 1.【必须】去除代码中的 print，否则导致正式和测试环境输出大量信息 
* 2.逻辑块空行分隔 
* 3.变量和其使用尽量放到一起
日志输出级别 
* Error：系统异常，影响用户使用，必须通知到开发者及时修复。 
* Warning：系统异常，不影响用户使用，但有异常需要跟进修复。
* Info：系统流水日志或日常记录。不做通知配置日志输出格式
* 日志需要包含当前函数名，方便查找和定位


每次代码提交必须有备注说明，注明本次提交做了哪些修改 
commit 分类： 
* 1.bugfix: ——— 线上功能 bug 
* 2.sprintfix: —— 未上线代码修改 （功能模块未上线部分bug） 
* 3.minor: ——— 不重要的修改（换行，拼写错误等） 
* 4.feature: —–—新功能说明
