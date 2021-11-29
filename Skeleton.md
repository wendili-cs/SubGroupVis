# CS765 DC-2 / Project
## 框架 - 文字说明

### 1 - 用户界面 & 输入接口

**(需要解决的问题)**

**1 - 1. 我们给用户多少元件，以用于确定模型？**

比如选择数据集，聚类的办法，聚类的超参数，subgroup的定义。


### 2 - 聚类 & 分析结果
K-means 聚类， 使用sklearn。 对于一个数据集，模型是确定的：即利用数值变量（numerical variable)来建模。
变动的是：subgroups。

Assume that we have numerical variables $X_1, X_2, ..., X_n$, we fit a k means model on the $n$ numerical variables and there are $K$ clusters.

**Subgroup的严密定义**：

Suppose now we have categorical variables as $F_1, F_2, ..., F_m$. For each variable $F_i$, it takes $k_i$ different values: {$l_{i, 1}, l_{i, 2}, ..., l_{i, k_i}$}

A subgroup $G$ is defined as below

$G := \wedge_j \{F_j = v_j\}$, where $v_j$ takes value from {$l_{j, 1}, l_{j, 2}, ..., l_{j, k_j}$}

Notice that, for a subgroup, not all the categorical variables would be used.

Say, in the car dataset, we have categorical variables *engineType, doorNum, powerType, fuelType, etc*. When defining a subgroup, we may only consider *engineType* and *fuelType* (Just an example, we do not have any assumption). This would be important for the following process.

**Subgroup Similarity / 组间相似性**:

For a subgroup $G$, suppose it involves the following categorical variables $F_{g1}, F_{g2}, ... F_{gk}$. Then, to find its most similar and distinct subgroups, we consider all other subgroups defined on these categorical variables $G_1, G_2, ..., G_{n_G}$. In previous part, we have $K$ clusters obtained from the k-means model. 

Within the given subgroup $G$, there are samples with different numerical variable values. We compute the proportions of samples that are classified into each cluster. For instance, 25% of the samples in $G$ would be in the first cluster, 15% of the samples in $G$ would be in the second cluster, etc. We define a corresponding feature vector $h_G$. 

$$h_G = (p_1^{(G)}, p_2^{(G)}, ..., p_K^{(G)})$$

Where $p_i^{(G)}$ stands for the proportion of samples in $G$ that are classified into the $i^{th}$ cluster.

Then, we compute the feature vector for all other subgroups and obtain $H_G = \{ h_{G_1}, h_{G_2}..., h_{G_{n_G}}\}$

At last, we compute the inner products between $h_G$ and all the elements in $H_G$. We choose the ones that have the smallest and largerst inner product values as the most similar and different subgroups.

**需要解决的问题:**

**2 - 1. 如何从数据集建立subgroups？**





### 3 - 可视化实现和展示

**(需要解决的问题)**




```python

```
