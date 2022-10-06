---
title: "不一样的论文解读：2018 Real Time Personalization Using Embeddings for Search Ranking at Airbnb"
date: 2022-10-06T15:33:57+08:00
draft: false
---

Airbnb这篇论文拿了今年KDD best paper，和16年google的W&D类似，并不fancy，但非常practicable，值得一读。可喜的是，据我所知，国内一线团队的实践水平并不比论文中描述的差，而且就是W&D，国内也有团队在论文没有出来之前就做出了类似的结果，可见在推荐这样的场景，大家在一个水平线上。希望未来国内的公司，也发一些真正实用的paper，不一定非要去发听起来fancy的。

自从Word2vec出来后，迅速应用到各个领域中，夸张一点描述，万物皆可embedding。在NLP中，一个困难是如何描述词，传统有onehot、ngram等各种方式，但它们很难表达词与词之间的语义关系，简单来讲，即词之间的距离远近关系。我们把每个词的Embedding向量理解成它在这个词表空间的位置，即位置远近能描述哪些词相关，那些词不相关。

对于互联网场景，比如电商、新闻，同样的，我们很难找到一个合适表达让计算机理解这些实体的含义。传统的方式一般是给实体打标签，比如新闻中的娱乐、体育、八卦等等。且不说构建一个高质量标签体系的成本，就其实际效果来讲，只能算是乏善可陈。类似NLP，完全可以将商品本身或新闻本身当做一个需要embedding的实体。当我们应用embedding方案时，一般要面对下面几个问题：

1. 希望Embedding表达什么，即选择哪一种方式构建语料
2. 如何让Embedding向量学到东西
3. 如何评估向量的效果
4. 线上如何使用

下面我们结合论文的观点来回答上面问题，水平有限，如有错误，欢迎指出。

## 希望Embedding表达什么
前面我们提了Embedding向量最终能表达实体在某个空间里面的距离关系，但并没有讲这个空间是什么。在NLP领域，这个问题不需要回答，就是语义空间，由我们文明中的各式各样的文本语料组成。在其他场景中，以电商举例，我们会直接对商品ID做Embedding，其训练的语料来至于用户的行为日志，故这个空间是用户的兴趣点组成。行为日志的类型不同，表达的兴趣也不同，比如点击行为、购买行为，表达的用户兴趣不同。故商品Embedding向量最终的作用，是不同商品在用户兴趣空间中的位置表达。

很多同学花很多时间在尝试各种word2vec的变种上，其实不如花时间在语料构建的细节上。首先，语料要多，论文中提到他们用了800 million search clicks sessions，在我们尝试Embedding的实践中，语料至少要过了亿级别才会发挥作用。其次，session的定义很重要。word2vec在计算词向量时和它context关系非常大，用户行为日志不像文本语料，存在标点符合、段落等标识去区分词的上下文。

举个例子，假设我们用用户的点击行为当做语料，当我们拿到一个用户的历史点击行为时，比如是list(商品A, 商品B，商品C，商品D)，很有可能商品B是用户搜索了连衣裙后点的最后一个商品，而商品C是用户搜索了手机后点击的商品，如果我们不做区分，模型会认为B和C处以一个上下文。

具体的session定义要根据自身的业务诉求来，不存在标准答案，比如上面的例子，如果你要做用户跨兴趣点的变换表达，也是可以的，论文中给出了airbnb的规则：
>A new session is started whenever there is a time gap of more than 30 minutes between two consecutive user clicks.

值得一提的是，论文中用点击行为代表短期兴趣和booking行为代表长期兴趣，分别构建Embedding向量。关于长短期兴趣，业界讨论很多，我的理解是长期兴趣更稳定，但直接用单个用户行为太稀疏了，无法直接训练，一般会先对用户做聚类再训练。

## 如何让Embedding向量学到东西
### 模型细节
一般情况下，我们直接用Word2vec，效果就挺好。论文作者根据Airbnb的业务特点，做了点改造，主要集中在目标函数的细节上，比较出彩。先来看一张图：
![55ed976756eca15151fa18cad78dc66a.png](evernotecid://9F673C11-F1B8-439E-A5F2-1F887AF2931E/appyinxiangcom/5069487/ENResource/p9854)
主要idea是增加一个global context，普通的word2vec在训练过程中，词的context是随着窗口滑动而变化，这个global context是不变的，原文描述如下：
> Both are useful from the standpoint of capturing contextual similarity, however booked sessions can be used to adapt the optimization such that at each step we predict not only the neighboring clicked listings but the eventually booked listing as well. This adaptation can be achieved by adding booked listing as global context, such that it will always be predicted no matter if it is within the context window or not

再看下它的公式，更容易理解：
![3deb61be9c288024fc6f1fbe958a249b.png](evernotecid://9F673C11-F1B8-439E-A5F2-1F887AF2931E/appyinxiangcom/5069487/ENResource/p9855)
注意到公式的最后一项和前面两项的区别，在累加符号的下面，没有变D限制。我的理解是，word2vec的算法毕竟是非监督的，而Airbnb的业务最终是希望用户Booking，加入一个约束，能够将学到的Embedding向量更好的和业务目标靠近。后面还有一个公式，思路是类似的，不再赘述。

这个思路也可以理解成另一种简单的多目标融合策略，另一篇阿里的论文也值得一读，提出了完整空间多任务模型（Entire Space Multi-Task Model，ESMM）来解决。

### 数据稀疏是核心困难
word2vec的算法并不神奇，还是依赖实体出现的频次，巧妇难为无米之炊，如果实体本身在语料中出现很少，也很好学到好的表达。曾经和阿里的同学聊过一次Embedding上线效果分析，认为其效果来源于中部商品的表达，并不是大家理解的长尾商品。头部商品由于数据量丰富，类似i2i的算法也能学的不错，而尾部由于数据太稀疏，一般也学不好，所以embedding技术要想拿到不错的收益，必须存在一批中部的商品。

论文中也提到，他们会对entity做个频次过滤，过滤条件在5-10 occurrences。有意思的是，以前和头条的同学聊过这个事情，他们那边也是类似这样的频次，我们这边会大一点。目前没有做的很细致，还未深究这个值的变化对效果的影响，如果有这方面经验的同学，欢迎指出。

另一个方法，也是非常常见，即对稀疏的id做个聚类处理，论文提了一个规则，但和Airbnb的业务耦合太深了，其他业务很难直接应用，但可以借鉴思想。阿里以前提过一种sixhot编码，来缓解这个问题，不知道效果如何。也可以直接hash，个人觉得这个会有损，但tensorflow的官网教程上，feature columns部分关于Hashed Column有一段话说是无损的：
>At this point, you might rightfully think: "This is crazy!" After all, we are forcing the different input values to a smaller set of categories. This means that two probably unrelated inputs will be mapped to the same category, and consequently mean the same thing to the neural network. The following figure illustrates this dilemma, showing that kitchenware and sports both get assigned to category (hash bucket) 12:


![1167d6491b834139ca69c6057d9c0dfd.png](evernotecid://9F673C11-F1B8-439E-A5F2-1F887AF2931E/appyinxiangcom/5069487/ENResource/p9860)


> As with many counterintuitive phenomena in machine learning, it turns out that hashing often works well in practice. That's because hash categories provide the model with some separation. The model can use additional features to further separate kitchenware from sports.

## 如何评估效果
向量评估的方式，主要用一些聚类、高维可视化tnse之类的方法，论文中描述的思路和我的另一篇文章https://zhuanlan.zhihu.com/p/35491904比较像。当Airbnb的工具做的比较好，直接实现了个系统来帮助评估。

值得一提的是，论文还提出一种评估方法，用embedding向量做排序，去和真实的用户反馈数据比较，直接引用airbnb知乎官方账号描述：
>更具体地说，假设我们获得了最近点击的房源和需要排序的房源候选列表，其中包括用户最终预订的房源；通过计算点击房源和候选房源在嵌入空间的余弦相似度，我们可以对候选房源进行排序，并观察最终被预订的房源在排序中的位置。

![a7b456c6adc4c7a25056b8e677993f5c.png](evernotecid://9F673C11-F1B8-439E-A5F2-1F887AF2931E/appyinxiangcom/5069487/ENResource/p9858)
上图可以看出，d32 book+neg的效果最好。

## 线上如何用
论文中反复提到的实时个性化并不难，只要支持一个用户实时行为采集的系统，就有很多种方案去实现实时个性化，最简单就是将用户最近的点击序列中的实体Embedding向量做加权平均，再和候选集中的实体做cosine距离计算，用于排序。线上使用的细节比较多，论文中比较出彩的点有两个：
### 多实体embedding向量空间一致性问题
这是一个很容易被忽视的问题，当需要多个实体embedding时，要在意是否在一个空间，否则计算距离会变得很奇怪。airbnb在构建long-term兴趣是，对用户和list做了聚类，原文如此描述：
> To learn user_type and listinд_type embeddings in the same vector space we incorporate the user_type into the booking sessions.

下图更容易理解：![dfa8cf16d769844da112fdcb73f1f617.png](evernotecid://9F673C11-F1B8-439E-A5F2-1F887AF2931E/appyinxiangcom/5069487/ENResource/p9862)


即直接将二者放在一个语料里面训练，保证在一个空间。如此，计算的cosine距离具有实际的意义。

## Negative反馈
无论是点击行为还是成交行为，都是用户的positive反馈，需要用户付出较大的成本，而另一种隐式的负反馈，我们很少用到（主要是噪音太强）。当前主流的个性化被人诟病最多的就是相似内容扎堆。给用户推相似内容，已经是被广泛验证有效的策略，但我们无法及时有效的感知到用户的兴趣是否已经发生变化，导致损坏了用户体验。因此，负反馈是一个很好的思路，airbnb给出了skipped listing_ids的策略。

比较可惜的是，我们目前没有在蘑菇街的场景拿到这个负反馈的收益，如果有同学在相关方面有经验，欢迎来指导。

## 最后
论文地址：https://www.kdd.org/kdd2018/accepted-papers/view/real-time-personalization-using-embeddings-for-search-ranking-at-airbnb

怎么学：
    多目标怎么融合。
    稀疏问题怎么处理
怎样用
    实时还是离线用
    怎么做实时：预测，求平均

如何评估向量的结果

我们验证的结果

## 总结：
1. 非监督训练embedding向量的思路可以借鉴，构建负反馈对，构建book的长期context对（多目标的一种思路）。
2. 训练的规模可以学习
3. 处理id过于稀疏，长期兴趣难学的问题，基于规则的聚类，但是电商比较难找规则
4. 我们的wide pair天然有负反馈，没有用。
5. user和list怎么放在一个空间内训练，直接构建成[<u,l>]对来学，是一种方式。
6. 排序的评估，figure 6
7. id类特征的频次5-10.
## 实时特征
## id的稀疏处理
## 负反馈
## global的全局，对比多目标
## 结合实践

## 问题

## 读厚
继Wide&deep之后，有一篇非常接地气的论文。通篇读下来，国内的大厂也能发，工作上差不多。推荐这篇文章的点在于有非常多可行的detail。

short-term：realtime
long-term
we can use the negative signal, e.g. skips of high ranked listings
listing embeddings 是什么

认为click是short-term，booking（购买）是long-term

Leveraging Conversions as Global Context 对比多目标的方法

• User Type Embeddings一种hash的方式

800 million search clicks sessions的数据量级，参考下

wordvec怎么到list vec？最后还是到单个listing id：

今日头条的哥们也提过类似的频次
Third, to learn a meaningful embedding for any entity from
contextual information at least 5 − 10 occurrences of that
entity are needed in the data, and there are many listinд_ids
on the platform that were booked less than 5 − 10 times.

u和list的共同空间：
![4fdfb4f409c9d7d79b3f571188bc5ad0.png](evernotecid://9F673C11-F1B8-439E-A5F2-1F887AF2931E/appyinxiangcom/5069487/ENResource/p9834)

他这个负反馈，由于是做无监督，和i2i类似的问题，只有点击序或者成交序，没有负反馈的存在，而我们的pair对天然就包含负反馈



