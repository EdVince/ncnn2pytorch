## (丐版)ncnn2pytorch
顾名思义，该项目就是将ncnn模型转换为pytorch模型。

详情可看知乎文章：https://zhuanlan.zhihu.com/p/458468021

### 设计思路
ncnn模型转pytorch模型有两步：
1. 通过param文件信息构建pytorch网络
2. 通过bin文件信息为pytorch的layer塞权重数据

不考虑libtorch的话，有两种做法：
1. 魔改现有ncnn的python接口，把各种layer也暴露出来，然后直接抽权重
优点：全程在python上操作，优雅
缺点：pybind不太会，搞不来
2. 基于c++工程，从c++版本抽权重保存为文件，然后再用python解析生成模型
优点：直观，简单
缺点：不够优雅

！！！考虑到我是练手c++用的，因此这里我使用的是第二种方法！！！

所以该项目有两个子项目：
1. 基于vs2019&c++的ncnn网络提取
2. 基于python的pytorch网络生成

### Details
##### 网络提取
1. 先用ncnn的load_param和load_model先把网络加载好
2. 把加载好的网络的layers和blob抽出来，把基本信息和layer参数保存到txt文件，权重信息保存到bin文件
3. extract跑一遍推理，把layer之间运行的先后顺序抓出来
##### 网络生成
1. 解析txt和bin文件，把网络的py文件生成出来
2. 解析bin文件，把对应的权重塞回去

### PS
1. 目前只做了ncnn自带的demo模型squeezenet_v1.1的转换，而且softmax层有点问题，推理出来的分类index是对的，但是值有一点小偏差
2. ncnn支持的框架太多了，pytorch转ncnn好弄，ncnn转回去pytorch的话，有一大堆pytorch不支持的东西

### 参考
1. [ncnn](https://github.com/Tencent/ncnn)