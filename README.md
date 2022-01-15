### ncnn2pytorch
顾名思义，该项目就是将ncnn模型转换为pytorch模型。

#### 设计思路
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
