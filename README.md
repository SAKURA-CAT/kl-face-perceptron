<div align="center">
<h1>kl-face-perceptron</h1>
    
[![SwanHub Demo](https://img.shields.io/static/v1?label=在线体验&message=SwanHub%20Demo&color=blue)](https://swanhub.co/cunyue/kl-face-perceptron/demo)


</div>

基于K-L变换的人脸识别器，本项目将基于K-L变换完成人脸特征提取，实现人脸识别，具体功能是：

1. 给予人像集合A，对A分别进行特征提取
2. 给予任意人像B，对B进行特征提取
3. 分别将人像B与A中人像进行比对，判断B是否为A中某一人脸。

> 测试数据来源：[PubFig: Public Figures Face Database](https://www.cs.columbia.edu/CAVE/databases/pubfig/)
> 

但事实上，完成人脸识别任务的先决条件是完成人脸裁剪，换句话说就是判断图像中是否存在人脸并将此人脸裁剪出来。这并不在本项目的研究范围之内，所以使用开源轻量的人脸检测库[mtcnn](https://pypi.org/project/mtcnn/)。

# 功能介绍

本项目将对输入的人脸进行人脸检测，在已知人像数据集中进行人像匹配，以分辨“输入人像”与数据集中的“谁”更像。

人脸识别系统是指以人脸识别技术为核心，是一项新兴的生物识别技术，是当今困际科技领域攻关的高精尖技术。它广泛采用区域特征分析算法，融合了计算机图像处理技术与生物统计学原理于一体，利用计算机图像处理技术从视频中提取人像特征点，利用生物统计学的原理进行分析建立数学模型，具有广阔的发展前景。人脸识别技术包含人脸检测、人脸跟踪、人脸比对三个部分。在K-L变换中，主要进行人脸检测。在人脸识别中，可以用 K-L 变换对人脸图像的原始空间进行转换，即构造人脸图像数据集的协方差矩阵，求出协方差矩阵的特征向量，再依据特征值的大小对这些特征向量进行排序， 这些特征向量表示特征的一个集合，它们共同表示一个人脸图像。在人脸识别领域，人们常称这些特征向量为特征脸。每一个体人脸图像都可以确切地表示为一组特征脸的线性组合。

## 基本处理思想

本项目首先需要对已知人像数据集进行处理，提取其特征和转换矩阵进行本地持久化保存以加速处理速度；其次对输入的带检测人像基于数据集的转换举证进行特征提取，得到特征矩阵，最后依次计算数据集特征与处理的输入人像特征的欧式距离，欧式距离最小的特征对应的人像就是输入人像对应的人。

接下来基于上述处理步骤，详细阐述整个系统的运行原理。

# 详细步骤

本部分将详细阐述系统运行原理，并附上重要代码片段。

## 第一步：定义人脸检测接口

完成人脸识别的第一步是对`输入图像/数据集`进行人像矫正，主要功能如下：

1. 拿到人脸的定位信息，裁剪出人头图像
2. 基于监测到的关键点数据，对人头图像进行人像矫正

人脸关键点检测将基于一个轻量级人脸检测模型mtcnn实现，由于人脸关键点检测并非本项目的重点，在这不再赘述mtcnn网络的具体结构。总之，通过人脸检测接口，我们校准了人像，为后面提取特征点做准备。

## 第二步：K-L变换与特征提取

K-L变换 （Karhunen-Loeve Transform）是建立在统计特性基础上的一种变换。K-L变换的突出优点是去相关性好，是均方误差（MSE，Mean Square Error）意义下的最佳变换，它在数据压缩技术中占有重要地位。

K-L变换是从K-L展开引出的，它讲的是对于一个任意维度向量，都可以用一个完备的正交归一向量系来展开。展开之后是一个有无穷多项的无穷级数，如果只选取有限项来表示这一向量，就达到了降维的目的。而选取有限项势必会存在误差，选取的项数越多误差会越小，但是项数增加就会影响降维的效果。如何在这二者之间找到一个平衡， K-L 变换给出了答案，K-L 变换将任意维度向量分解各个正交向量之和，使每个向量所携带的信息都尽可能多，达到了以最少项数，包含最多信息的目的。在不丢失主要信息的条件下，起到了最好的降维效果。

### 图像标准化

我们对图像进行标准化处理，主要包括大小、纬度的标准化，具体而言就是缩放裁切后的人头图像为64x64像素，并对图像进行灰度化处理：

```python
def standardize(img: np.ndarray, size: int):
    """标准化一个图像,裁切、缩放、灰度化

    Parameters
    ----------
    img : np.ndarray
        图像
    size : int
        图像标准化尺寸
    """
    # target人像矫正, 裁切出人脸
    img = align_face(img)
    # 图像灰度化
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 图像标准化，将图像转换为size*size的大小
    img = cv2.resize(img, (size, size))
    # 图像向量化，将图像转换为一维列向量
    img = img.flatten()
    return img
```

### k-l变换

<aside>
💡 64x64=4096；我们假定输入的数据集图像为5

</aside>

将sources中每个列向量合并得到一个矩阵，`sizexsize`的图像转换为`(size*size)*len(sources)`的矩阵，也就是`4096x5`的矩阵，依次求协方差和特征值，特征向量：

```python
# 将sources中每个列向量合并得到一个矩阵，size*size的图像转换为(size*size)*len(sources)的矩阵
ss = np.array(ss).T
# 计算sources的协方差矩阵
cov = np.cov(ss)
# 计算协方差矩阵的特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(cov)
```

然后对特征值从大到小排序，计算前k项的和占比所有项的和>99%，得到对应特征向量，可求得系数矩阵c：

```python
# 计算特征值的和，取前k个特征值的和大于总和的阈值
total = sum(eigenvalues)
k = 0
for i, _ in enumerate(eigenvalues):
    if sum(eigenvalues[:i]) / total >= threshold:
        k = i
        break
# 取前k个特征向量，得到vc
vc = eigenvectors[:, idx[:k]]
# 最终的系数矩阵为vc.T * (ss - mean)
c = vc.T.dot(ss - np.mean(ss, axis=1).reshape(-1, 1))
```

最后保存数据集，供人脸检测使用：

```python
# 保存数据集，顺便保存数据源
np.savez(DATASET, c=c, vc=vc, mean=np.mean(ss, axis=1), sources=sources)
return c, vc, np.mean(ss, axis=1), sources
```

至此完成了第二步，对提取了人脸的特征向量和变换矩阵

## 第三步：输入图像特征转换

同样对输入图像进行图像标准化，经历裁切、旋转、灰度化等步骤后，载入事先保存好的数据集信息，计算输入图像的系数矩阵：

```python
# target标准化
target = standardize(target, size)
c, vc, mean, sources = create_dataset(sources, size)
# 计算target的系数
t = vc.T.dot(target - mean)
```

## 第四步：计算欧式距离，人脸匹配

依次计算`sources`中每个人脸与`target`的距离，取距离最小的作为检测结果，代码如下：

```python
scores = []
for i in range(c.shape[1]):
    scores.append(np.linalg.norm(t - c[:, i]))
idx = np.argmin(scores)
print("this is", sources[idx], "with score", scores[idx])
```

最后`idx`保存的就是输入图像与数据集中最像的人脸。

# 结果分析

数据集为：

![1](kl-face-perceptron%20adf42954a8b64a28b9d6110df5a73341/Untitled.jpeg)

1

![2](kl-face-perceptron%20adf42954a8b64a28b9d6110df5a73341/Untitled%201.jpeg)

2

![3](kl-face-perceptron%20adf42954a8b64a28b9d6110df5a73341/Untitled%202.jpeg)

3

![4](kl-face-perceptron%20adf42954a8b64a28b9d6110df5a73341/Untitled%203.jpeg)

4

![5](kl-face-perceptron%20adf42954a8b64a28b9d6110df5a73341/Untitled%204.jpeg)

5

输入图像为：

![6](kl-face-perceptron%20adf42954a8b64a28b9d6110df5a73341/Untitled%205.jpeg)

6

![7](kl-face-perceptron%20adf42954a8b64a28b9d6110df5a73341/Untitled%206.jpeg)

7

其中，6号图像与4号图像在[PubFig: Public Figures Face Database](https://www.cs.columbia.edu/CAVE/databases/pubfig/)中实际上是一个人，都为*Jim OBrien*；5号图像与7号图像也为一个人，名为*Martha Bowen*。预期情况是，当输入6号图像时，程序能判断出6号与4号图像最为相似，当输入7号图像时，程序能判断出7号图像与5号图像最为相似。

我们对最终的分析结果进行可视化展示，界面如下：

![Untitled](kl-face-perceptron%20adf42954a8b64a28b9d6110df5a73341/Untitled.png)

![Untitled](kl-face-perceptron%20adf42954a8b64a28b9d6110df5a73341/Untitled%201.png)

![Untitled](kl-face-perceptron%20adf42954a8b64a28b9d6110df5a73341/Untitled%202.png)

事实与预期相同，实验成功。
