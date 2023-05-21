# 评价指标

## Precision@N


<img src="https://www.zhihu.com/equation?tex=precision@N=\frac{|R\cap \hat{R}_{1:N}|}{N}
" alt="precision@N=\frac{|R\cap \hat{R}_{1:N}|}{N}
" class="ee_img tr_noresize" eeimg="1">

 <img src="https://www.zhihu.com/equation?tex=R" alt="R" class="ee_img tr_noresize" eeimg="1"> 用户实际交互的列表， <img src="https://www.zhihu.com/equation?tex=\hat{R}_{1:N}" alt="\hat{R}_{1:N}" class="ee_img tr_noresize" eeimg="1"> 预测的Top-N列表。预测的Top-N的命中率。

缺点：通常在所有指标中是最不稳定的，相关结果的总数会对Precision@N有非常强的影响。

## Recall@N


<img src="https://www.zhihu.com/equation?tex=recall@N=\frac{|R\cap \hat{R}_{1:N}|}{|R|}
" alt="recall@N=\frac{|R\cap \hat{R}_{1:N}|}{|R|}
" class="ee_img tr_noresize" eeimg="1">

 <img src="https://www.zhihu.com/equation?tex=R" alt="R" class="ee_img tr_noresize" eeimg="1"> 用户实际交互的列表， <img src="https://www.zhihu.com/equation?tex=\hat{R}_{1:N}" alt="\hat{R}_{1:N}" class="ee_img tr_noresize" eeimg="1"> 预测的Top-N列表。用户实际交互的item，有多少被召回了。

## F-Score


<img src="https://www.zhihu.com/equation?tex=FScore=(1+\beta^2)\frac{P\times R}{(\beta^2 P+R)}
" alt="FScore=(1+\beta^2)\frac{P\times R}{(\beta^2 P+R)}
" class="ee_img tr_noresize" eeimg="1">

当 <img src="https://www.zhihu.com/equation?tex=\beta=1" alt="\beta=1" class="ee_img tr_noresize" eeimg="1"> 时，即为 <img src="https://www.zhihu.com/equation?tex=F_1 Score" alt="F_1 Score" class="ee_img tr_noresize" eeimg="1"> 。

## HR@N

在top-K推荐中，HR是一种常用的衡量召回率的指标，其计算公式如下：

<img src="https://www.zhihu.com/equation?tex=HR@N=\frac{NumberofHitsUser@K}{NumberofTotalUser}
" alt="HR@N=\frac{NumberofHitsUser@K}{NumberofTotalUser}
" class="ee_img tr_noresize" eeimg="1">
分子是推荐列表命中用户ground_truth的用户总数。
分母是所有用户数。

**当用户的ground_truth只有1个时， <img src="https://www.zhihu.com/equation?tex=Hit Rate@N" alt="Hit Rate@N" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=Recall@N" alt="Recall@N" class="ee_img tr_noresize" eeimg="1"> 等价。**

## MAP(Mean Average Precision)


<img src="https://www.zhihu.com/equation?tex=AP=\frac{\sum_{N=1}^{|\hat{R}|}precision@N\times rel(N)}{|R|}\\
rel(N)=1,\hat{R}中的第N个item在R中\\
MAP=\frac{1}{Q}\sum_{i=1}^{|Q|}AP_i
" alt="AP=\frac{\sum_{N=1}^{|\hat{R}|}precision@N\times rel(N)}{|R|}\\
rel(N)=1,\hat{R}中的第N个item在R中\\
MAP=\frac{1}{Q}\sum_{i=1}^{|Q|}AP_i
" class="ee_img tr_noresize" eeimg="1">

MAP是所有用户AP的平均值。
对于单个信息需求来说，平均正确率是未插值的正确率-召回率曲线下的面积的近似值，因此，MAP可以粗略地认为是某个查询集合对应的多条正确率-召回率曲线下面积的平均值。

**当用户的ground_truth只有一个时，MRR和MAP等价。**

```python
def AP(ranked_list, ground_truth):
    hits = 0
    sum_precs = 0
    for n in range(len(ranked_list)):
        if ranked_list[n] in ground_truth:
            hits += 1
            sum_precs += hits / (n + 1.0)
    if hits > 0:
        return sum_precs / len(ground_truth)
    else:
        return 0
```

## MRR(Mean Reciprocal Rank)



<img src="https://www.zhihu.com/equation?tex=MRR=\frac{1}{Q}\sum_{i=1}^{|Q|}\frac{1}{rank_i}
" alt="MRR=\frac{1}{Q}\sum_{i=1}^{|Q|}\frac{1}{rank_i}
" class="ee_img tr_noresize" eeimg="1">
 <img src="https://www.zhihu.com/equation?tex=|Q|" alt="|Q|" class="ee_img tr_noresize" eeimg="1"> 是用户数， <img src="https://www.zhihu.com/equation?tex=rank_i" alt="rank_i" class="ee_img tr_noresize" eeimg="1"> 是对于第 <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1"> 个用户，推荐列表中第一个在ground-truth中出现的item所在的位置。

**当用户的ground_truth只有一个时，MRR和MAP等价。**

## NDCG(Normalized Discounted Cummulative Gain)

DCG的计算：

 <img src="https://www.zhihu.com/equation?tex=rel_i" alt="rel_i" class="ee_img tr_noresize" eeimg="1"> 表示在位置 <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1"> 上的效益， <img src="https://www.zhihu.com/equation?tex=\frac{1}{log_2(i+1)}" alt="\frac{1}{log_2(i+1)}" class="ee_img tr_noresize" eeimg="1"> 表示在位置 <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1"> 上的折损(显然有效结果排名越靠前越好)，则在位置 <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1"> 上的增益为 <img src="https://www.zhihu.com/equation?tex=rel_i*\frac{1}{log_2(i+1)}" alt="rel_i*\frac{1}{log_2(i+1)}" class="ee_img tr_noresize" eeimg="1"> 。
当 <img src="https://www.zhihu.com/equation?tex=rel\in \{0,1\}" alt="rel\in \{0,1\}" class="ee_img tr_noresize" eeimg="1"> 时，

<img src="https://www.zhihu.com/equation?tex=DCG_P=\sum_{i=1}^{p}\frac{rel_i}{log_2(i+1)}
" alt="DCG_P=\sum_{i=1}^{p}\frac{rel_i}{log_2(i+1)}
" class="ee_img tr_noresize" eeimg="1">
CG相关性不止是两个，可以是实数的形式，所以有以下形式， <img src="https://www.zhihu.com/equation?tex=rel_i\in [0,1]" alt="rel_i\in [0,1]" class="ee_img tr_noresize" eeimg="1"> 。

<img src="https://www.zhihu.com/equation?tex=DCG_p=\sum_{i=1}^{p}\frac{2^{rel_i}-1}{log_2(i+1)}
" alt="DCG_p=\sum_{i=1}^{p}\frac{2^{rel_i}-1}{log_2(i+1)}
" class="ee_img tr_noresize" eeimg="1">
NDCG的计算：

由于DCG是一个累计值，当不同的query，返回结果数量不同时，它们之间将无法比较，所以对其进行归一化得到NDCG。

<img src="https://www.zhihu.com/equation?tex=NDCG_p=\frac{DCG_p}{IDCG_p}
" alt="NDCG_p=\frac{DCG_p}{IDCG_p}
" class="ee_img tr_noresize" eeimg="1">
IDCG为理想情况下最大的DCG值。

<img src="https://www.zhihu.com/equation?tex=IDCG_P=\sum_{i=1}^{|REL|}\frac{2^{rel_i}-1}{log_2(i+1)}
" alt="IDCG_P=\sum_{i=1}^{|REL|}\frac{2^{rel_i}-1}{log_2(i+1)}
" class="ee_img tr_noresize" eeimg="1">
其中， <img src="https://www.zhihu.com/equation?tex=|REL|" alt="|REL|" class="ee_img tr_noresize" eeimg="1"> 表示结果按照相关性从大到小顺序排序，取前p个结果组成的集合。

## AUC



## ROC


