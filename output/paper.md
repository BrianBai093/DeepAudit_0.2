# BlockBoost: Scalable and Efficient Blocking through Boosting

Thiago R. Ramos USP

Rodrigo Schuller IMPA

Alex A. Okuno NYU

Lucas Nissenbaum IMPA

Roberto I. Oliveira IMPA

Paulo Orenstein IMPA

# Abstract

As datasets grow larger, matching and merging entries from different databases has become a costly task in modern data pipelines. To avoid expensive comparisons between entries, blocking similar items is a popular preprocessing step. In this paper, we introduce BlockBoost, a novel boosting-based method that generates compact binary hash codes for database entries, through which blocking can be performed efficiently. The algorithm is fast and scalable, resulting in computational costs that are orders of magnitude lower than current benchmarks. Unlike existing alternatives, BlockBoost comes with associated feature importance measures for interpretability, and possesses strong theoretical guarantees, including lower bounds on critical performance metrics like recall and reduction ratio. Finally, we show that BlockBoost delivers great empirical results, outperforming state-of-the-art blocking benchmarks in terms of both performance metrics and computational cost.

# 1 INTRODUCTION

With larger datasets and disparate data sources becoming more prevalent, properly identifying, integrating and linking entries across multiple datasets is now a crucial step in many data pipelines. This process of identifying and disambiguating entities within one or more datasets is known as entity matching, entity resolution or record link-

age [Christen, 2012b]. Applications include merging health records [Clark, 2004, Kelman et al., 2002], aggregating census data [Winkler, 2006], identifying war casualties [Steorts and Shrivastava, 2018], detecting crimes [Jonas and Harper, 2006], cataloging bibliographic citations or business products, and matching genome sequences [Christen, 2012a]. By Rademacher Inequality [2], we have that with probability at least A fundamental issue in entity matching is the quadratic number of comparisons necessary between database items. Blocking [Steorts et al., 2014] is a popular technique to reduce the number of comparisons. It consists of blocking together items considered similar (in some metric) and only comparing entries within the same block. For example, if the goal is to match a list of customer purchase records to a list of customer accounts, blocks may be based on the customer’s last name, or the ZIP code of their billing address, or all of these combined. If the blocks are well-crafted, this can significantly improve the speed and efficiency of the matching process.

A common way to create blocks is via hashing. A hash function yields a low-dimension binary representation of each entry, called the hash code. Unlike standard dimensionality-reduction methods, the fact that this representation is binary is important to ensure fast retrieval time, which is of the utmost importance for large databases [Andoni and Indyk, 2006, Charikar, 2002, Kulis and Darrell, 2009]. An efficient hash code maps similar candidates to the same hash code and dissimilar items to different hash codes, significantly reducing the associated number of comparisons and ensuring that similar items are indeed between these candidates.

However, devising good hash functions can be very hard. An important technique to create hash functions is locality-sensitive hashing (LSH). In locality-sensitive hashing [Andoni and Indyk, 2006, Har-Peled et al., 2012], hash codes are built in such a way that points

that are close in some metric typically have similar hash codes. It has many interesting theoretical guarantees that are valid under broad circumstances. However, the fact that LSH is agnostic to the nature of the underlying data often leads to suboptimal performance. To address this issue, there is a class of techniques known as learning to hash [Andoni and Beaglehole, 2021, Weiss et al., 2008a, Kulis and Darrell, 2009, Wang et al., 2018] that aims to improve the hashing efficiency by learning hash functions tailored for specific tasks, such as entity matching [Steorts et al., 2014]. While this approach can improve performance, it also introduces new challenges. For instance, methods like kernel LSH and TLSH maintain LSH’s theoretical guarantees, but sacrifice scalability and efficiency [Kulis and Grauman, 2009, Jiang et al., 2014, Oliver et al., 2013]. A more recent alternative for blocking dispenses hashing altogether and is based on deep learning embeddings applied to entity matching [Thirumuruganathan et al., 2021a, Mudgal et al., 2018]. These state-of-the-art solutions use neural networks to learn feature representations that capture the underlying relationships between records, thus improving the accuracy and effectiveness of blocking.

In this paper, we propose a new blocking method called BlockBoost, that combines a boosting step that learns a pairwise similarity function and a hashing step on top of which blocking can be quickly performed (see Figure 1). Since boosting is a fast machine learning method with great out-of-the-box performance, BlockBoost works well for many different types of unstructured data. Furthemore, unlike many traditional blocking alternatives, it is possible to devise lower bounds on BlockBoost’s performance in terms of relevant metrics such as recall and reduction ratio. This results in a data-driven technique that is efficient and scalable, with provable guarantees. Finally, Block-Boost achieves superior results to state-of-the-art deep learning solutions on multiple datasets.

Main Contributions. We introduce BlockBoost, a novel blocking algorithm that combines hashing and boosting with several features:

• Efficient data compression: by learning hash codes from data, BlockBoost is able to extract and the combine the most distinguishing features and automatically pick the right hash size; e.g., in one of the empirical examples considered, it obtains state-of-the-art results by compressing 9600 bits in the original features into a 150-bit hash. Also, hashing dimensions are ordered by importance, and can be trimmed for further compression;

• Speed: the training is quasi-linear in the entries, with linear prediction time; it is an order of magnitude faster than alternatives, it scales to millions of entries and runs well on CPUs;   
• Simple tuning: BlockBoost has a single, easy-tointerpret hyperparameter;   
• Theoretical results: unlike most blocking algorithms, BlockBoost has theoretical guarantees on its performance and lower bounds on popular metrics such as recall and reduction ratio;   
• Interpretability: it is possible to interpret the contribution of each data feature to the final hashes by looking at importance measures derived from boosting; this is useful to identify the most distinguishing features available in the data;   
• Empirical performance: BlockBoost outperforms state-of-the-art solutions on many canonical blocking datasets in terms of recall, reduction ratio and their harmonic average.

# 2 RELATED WORK

Due to its importance in data pipelines across many applications, entity matching [Winkler, 2004, Christophides et al., 2020, Elmagarmid et al., 2007] is a widely studied field. While blocking is an old idea [Fellegi and Sunter, 1969], it remains an active area of research [Papadakis et al., 2020].

Several blocking techniques are based on exact matches or certain blocking keys, such as attribute matching [Azzalini et al., 2020] and token blocking [O’Hare et al., 2019]. However, many real-world data possesses unnormalized or corrupted data, posing a serious challenge to such methods [Zhang et al., 2020]. Because BlockBoost learns attributes through boosting, it does not suffer from this problem.

There are also methods that do not require exact attribute matching for block creation. For example, the canopy clustering algorithm [McCallum et al., 2000] groups together items based on the similarity of certain fields using a clustering algorithm. However, it is slow and requires the tuning of several hyperparameters to obtain a competitive performance. BlockBoost, on the other hand, has a single hyperparameter.

Hashing-based blocking methods, such as LSH, are closer in spirit to our approach. They partition records to the same blocks if they share the same hash value, using a prespecified random hash function. For example, in [Steorts et al., 2014, Steorts and Shrivastava, 2018], community detection techniques [Oliver et al., 2013]


[ImageDescription]
- source: images/a040d411ef1be791d2e514a9d82a3c9416ee38367e62e8bf7d2fa8b42daccdba.jpg
- alt: (no-alt)
- description: Image found in markdown. Detailed vision caption is unavailable in this runtime.
  
Figure 1: Overview of how BlockBoost performs blocking through (i) boosting and (ii) hashing.

and clustering algorithm [Paulevé et al., 2010] were used as a post-processing steps to LSH and lead to good results in entity matching. There are also learning to hash algorithms [Andoni and Beaglehole, 2021, Weiss et al., 2008a, Kulis and Darrell, 2009, Wang et al., 2018] that try to learn the hash functions via a training stage or refine already existing hash functions to lessen the correlations and redundancies between bits [Liu et al., 2024]. However, they are typically not employed for blocking due to scalability issues with learning overly complex algorithms [Kulis and Grauman, 2009, Jiang et al., 2014, Oliver et al., 2013]. BlockBoost, in contrast, can be orders of magnitude faster.

In recent years, there has been a growing interest in using deep learning embeddings to enhance the performance of blocking in entity matching [Thirumuruganathan et al., 2021a, Mudgal et al., 2018]. These approaches are oftentimes considered to be state-of-the-art, as they leverage neural networks to learn intricate feature representations that capture the inherent relationships between records.

Finally, other works, such as [Kim et al., 2020] and [Shakhnarovich, 2005], also employ boosting techniques for hashing. Still their setting significantly differs from ours as they do not consider blocking for entity matching. Indeed, [Kim et al., 2020] is concerned with distance functions in metric spaces and [Shakhnarovich, 2005] employ a gradient ascent algorithm with no generalization results.

# 3 BLOCKBOOST

Given datasets $\mathcal { A } : = \{ A _ { \ell } \} _ { \ell = 1 } ^ { N _ { \mathcal { A } } }$ and $B : = \{ B _ { r } \} _ { r = 1 } ^ { N _ { B } }$ both contained in a set $\mathcal { X }$ , and a relationship between items $\sim _ { R }$ , we want to find pairs $\left( A _ { \ell } , B _ { r } \right)$ such that $A _ { \ell } \sim _ { R } B _ { r }$ , where we assume $\sim _ { R }$ is unknown and must be learned. In our entity matching application, $A _ { \ell }$ and $B _ { r }$ will correspond to entities in different datasets and we will say that $A _ { \ell } \sim _ { R } B _ { r }$ if and only if $A _ { \ell }$ and $B _ { r }$ are

the same entity. For ease of presentation, our setup assumes two databases; however, the results hold for an arbitrary number of data collections (or one collection with many representations of the same item).

Our goal is to build a hash table for these items so that, given an item $A _ { \ell }$ , one can find $B _ { r } \sim _ { R } A _ { \ell }$ with as few table lookups as possible. To this end, we suppose we have access to a training sample,

$$
\mathcal {S} _ {\text {t r a i n}, n} := \left\{\left(\left(A _ {i}, B _ {i}\right), y _ {i}\right) \in (\mathcal {A} \times \mathcal {B}) \times \{- 1, 1 \}, i \in [ n ] \right\},
$$

such that, $y _ { i } = 1$ if $A _ { i } \sim _ { R } B _ { i }$ and $^ { - 1 }$ otherwise. This sample will be used in a training stage so our algorithm can learn a similarity classifier via a sample of similar/dissimilar items using boosting.

Our method consists of two steps:

Boosting step. Using boosting, we learn binary classifiers $\{ k _ { t } ^ { * } \} _ { t = 1 } ^ { T }$ over $\mathcal { X }$ , as well as convex weights $\{ \alpha _ { t } ^ { * } \} _ { t = 1 } ^ { T }$ for these classifiers. Then, given items $A \in { \mathcal { A } }$ and $B \in B$ , we construct a similarity function $\begin{array} { r } { f ^ { * } ( A , B ) \ = \ \sum _ { t = 1 } ^ { T } \alpha _ { t } ^ { * } k _ { t } ^ { * } ( A ) k _ { t } ^ { * } ( B ) } \end{array}$ . In this step, weights $\alpha _ { t } ^ { * }$ are expected to be large when the product $k _ { t } ^ { * } ( A ) k _ { t } ^ { * } ( B )$ correlates strongly with the similarity relation $A \sim _ { R } B$ , and consequently $f ^ { * } ( A , B )$ is close to $+ 1$ when $A \sim _ { R } B$ and $f ^ { * } ( A , B )$ is close to $^ { - 1 }$ when $A \not \sim _ { R } B$ .

Hashing step. From the $\{ \alpha _ { t } ^ { * } , k _ { t } ^ { * } \} _ { t = 1 } ^ { T }$ learned in the previous step, various techniques can be employed to create hash codes for blocking. In this paper, we focus on using the similarity function learned in the boosting to create blocks via a weighted hamming distance between pairs. The Supplementary Material discusses another plausible option based on LSH [Andoni and Indyk, 2006, Har-Peled et al., 2012].

We now consider each of the above steps in further detail.

# 3.1 Boosting Step

Fix a family $\kappa$ of binary classifiers $k : \mathcal { X }  \{ - 1 , + 1 \}$ and a max number of iterations $T _ { \operatorname* { m a x } } \in \{ 1 , 2 , \dots \}$ . To

Algorithm 1 Boosting step   
Input: $S_{\mathrm{train},n} = ((A_i,B_i),y_i)_i = 1$ max number of iterations $T_{\mathrm{max}}\in \mathbb{N}$ , binary family $\kappa$ 1: for $i\gets 1$ to $n$ do   
2: $Q_{1}(i)\leftarrow \frac{1}{n}$ 3: end for   
4: $t\gets 1$ 5: $T\gets T_{\mathrm{max}}$ 6: while $t\leq T_{\mathrm{max}}$ do   
7: $k_{t}^{*}\gets$ classifier in $\kappa$ with smallest error $\varepsilon_t = \sum_{i = 1}^n Q_t(i)\mathbf{1}_{[y_i k_t^* (A_i)k_t^* (B_i) <   0]}$ 8: if $\varepsilon_t\geq 1 / 2$ then   
9: $T\gets t - 1$ 10: break   
11: else   
12: $\alpha_{t}^{\prime}\gets \frac{1}{2}\log \left(\frac{1 - \varepsilon_{t}}{\varepsilon_{t}}\right)$ 13: $Z_{t}\gets 2[\varepsilon_{t}(1 - \varepsilon_{t})]^{1 / 2}$ 14: for $i\gets 1$ to $n$ do   
15: $Q_{t + 1}(i)\gets \frac{Q_t(i)\exp(-\alpha'_t y_i k_t^*(A_i)k_t^*(B_i))}{Z_t}$ 16: end for   
17: end if   
18: $t\gets t + 1$ 19: end while   
20: for $t\gets 1$ to $T$ do   
21: $\alpha_{t}^{*}\gets \frac{\alpha_{t}^{\prime}}{\sum_{s = 1}^{T}\alpha_{s}^{\prime}}$ 22: end for   
Output: $(\alpha_{t}^{*})_{t = 1}^{T},(k_{t}^{*})_{t = 1}^{T}$

find the functions $\{ k _ { t } ^ { * } \} _ { t = 1 } ^ { T } \in \mathcal { K }$ and the convex weights $\{ \alpha _ { t } ^ { * } \} _ { t = 1 } ^ { T }$ , with $T \leq T _ { \mathrm { m a x } }$ , we use a boosting algorithm over our training sample $S _ { \mathrm { t r a i n } , n } = ( ( A _ { i } , B _ { i } ) , y _ { i } ) _ { i = 1 } ^ { n }$ ; see Algorithm 1. Note that $T$ , rather than $T _ { \mathrm { m a x } }$ , is the key determinant of the total number of binary classifiers produced by our method. As we will demonstrate in forthcoming discussions, it also plays a pivotal role in determining the ultimate level of compression achieved by our method, as it directly influences the number of bits required for the hash.

While Algorithm 1 is reminiscent of AdaBoost [Freund and Schapire, 1997], note we are optimizing a function that is quadratic over the chosen classifier. This is crucial for the hashing step, described in Section 3.2. Also, there are classifier families for which this optimization problem is feasible. An example is the set of decision stumps ${ \mathcal { H } } _ { \mathrm { s t u m p s } }$ : for $x \in \mathbb { R } ^ { d }$ , if $x _ { ( j ) }$ indicates the $j$ -th coordinate of $\boldsymbol { x } \in \mathbb { R } ^ { d }$ , then

$$
\mathcal {H} _ {\text {s t u m p s}} = \left\{\mathbf {1} _ {[ x _ {(j)} <   \xi ]} \cup \mathbf {1} _ {[ x _ {(j)} \geq \xi ]}: \xi \in \mathbb {R}, j \in [ p ] \right\}. \tag {1}
$$

Intuitively, since the model is trying to predict matches and non-matches, we expect that, given items $A \in { \mathcal { A } }$ and $B \in B$ , the learned function

$$
f ^ {*} (A, B) = \sum_ {t = 1} ^ {T} \alpha_ {t} ^ {*} k _ {t} ^ {*} (A) k _ {t} ^ {*} (B), \tag {2}
$$

will be a good similarity measure between $A$ and $B$ . That is, $f ^ { * } ( A , B )$ should be close to $+ 1$ when $A \sim _ { R } B$ , and close to $^ { - 1 }$ otherwise. Notice that, as usual in boosting, larger weights $\alpha _ { t } ^ { * }$ are given to the classifiers $k _ { t } ^ { * }$ that achieve the smallest values of boosted errors $\varepsilon _ { t }$ . That is, our algorithm naturally gives more weight to functions $k _ { t } ^ { * } ( A ) k _ { t } ^ { * } ( B )$ that correlate more strongly to the similarity relation.

# 3.2 Hashing Step

We now use the convex weights $\left( \alpha _ { t } ^ { * } \right) _ { t = 1 } ^ { T }$ and the functions $( k _ { t } ^ { * } ) _ { t = 1 } ^ { T }$ to construct hash functions that will be used for blocking. Our solution involves a straightforward calculation of a weighted Hamming distance over hash codes to create the blocks.

For each element $A \in { \mathcal { X } }$ , we create a $T$ -bit hash function $g$ that is given by

$$
g (A) = \left(k _ {1} ^ {*} (A), \dots , k _ {T} ^ {*} (A)\right),
$$

where $T$ is the number of iterations used in the boosting step. Items $( A , B )$ will be part of the same block if for a given small $\delta \in [ 0 , 1 ]$ ,

$$
f ^ {*} (A, B) = \sum_ {i = 1} ^ {T} \alpha_ {i} ^ {*} k _ {t} ^ {*} (A) k _ {t} ^ {*} (B) \geq 1 - \delta \tag {3}
$$

After creating the blocks, we can reduce direct comparisons only to pairs $( A , B )$ in each block, sparing a number of unnecessary comparisons when declaring a match.

An additional benefit of the proposed hashing approach is that it allows for the expression in Equation 3 to be written as a weighted Hamming distance between the hashes $g ^ { * } ( A )$ and $g ^ { * } ( B )$ ,

$$
\begin{array}{l} \sum_ {i = 1} ^ {T} \alpha_ {i} ^ {*} k _ {t} ^ {*} (A) k _ {t} ^ {*} (B) = \sum_ {i = 1} ^ {T} \alpha_ {i} ^ {*} (1 - | k _ {t} ^ {*} (A) - k _ {t} ^ {*} (B) |) \\ = 1 - \sum_ {i = 1} ^ {T} \alpha_ {i} ^ {*} | k _ {t} ^ {*} (A) - k _ {t} ^ {*} (B) |, \\ \end{array}
$$

since $\textstyle \sum _ { t = 1 } ^ { T ^ { \prime } } \alpha _ { t } ^ { * } \ = \ 1$ . Therefore, items $( A , B )$ will be

$$
\sum_ {i = 1} ^ {T} \alpha_ {i} ^ {*} \left| k _ {t} ^ {*} (A) - k _ {t} ^ {*} (B) \right| \leq \delta , \tag {4}
$$

and this can be computed efficiently, as we discuss in Section 5.6.

# 4 THEORETICAL GUARANTEES

For our theoretical analysis, we assume we have two datasets $\mathcal { A } : = \{ A _ { \ell } \} _ { \ell = 1 } ^ { N _ { \mathcal { A } } }$ and $\boldsymbol { B } : = \{ B _ { r } \} _ { r = 1 } ^ { N _ { B } }$ , both contained in a set $\mathcal { X }$ and a notion of similarity $\sim _ { R }$ between

items $\left( A _ { \ell } , B _ { r } \right)$ . For entity matching problems, we assume that $A _ { \ell } \sim _ { R } B _ { r }$ if and only if $A _ { \ell }$ and $B _ { r }$ are the same entity. All proofs for the results in this section can be found in the Supplementary Material.

# 4.1 Performance Metrics

We first define traditional performance metrics for entity matching. The goal of our method is to ensure that, for each $A \in { \mathcal { A } }$ , one can find all similar $B \in B$ while doing as few pairwise comparisons as possible. This is made precise by the Recall and the Reduction Ratio (RR) metrics [Christen, 2012b]:

$$
\operatorname {R e c a l l} := \frac {1}{| \mathcal {M} |} \sum_ {(\ell , r) \in \mathcal {M}} \mathbf {1} _ {[ A _ {\ell} \text {a n d} B _ {r} \text {s h a r e a b l o c k} ]}; \tag {5}
$$

$$
\operatorname {R R} := 1 - \frac {1}{| \mathcal {N} |} \sum_ {(\ell , r) \in \mathcal {N}} \mathbf {1} _ {[ A _ {\ell} \text {a n d} B _ {r} \text {s h a r e a b l o c k} ]}, \tag {6}
$$

where $\mathcal { N } : = [ N _ { A } ] \times [ N _ { B } ]$ denotes all possible pairs and $\mathcal { M }$ denotes the set of matching pairs:

$$
\mathcal {M} := \left\{\left(\ell , r\right) \in \mathcal {N}, A _ {\ell} \sim_ {R} B _ {r}, \left(A _ {\ell}, B _ {r}\right) \in \mathcal {A} \times \mathcal {B} \right\}. \tag {7}
$$

Recall, also known as pair completeness, measures the proportion of similar pairs that end up in the same block, whereas RR, also known as efficiency, measures the proportion of the $N _ { A } \cdot N _ { B }$ potential pairwise comparisons that are avoided. Ideally, we would like to find as many as possible matching pairs (Recall $\approx 1$ ), while avoiding as many comparisons as possible (RR $\approx 1$ ).

To be able to compare performance through a single real value, a commonly used metric in the blocking literature [Azzalini et al., 2020] is the the harmonic mean between Recall and RR, $\mathrm { H } ( \mathrm { R e c a l l } , \mathrm { R R } )$ , given by:

$$
\mathrm {H} (\text {R e c a l l}, \mathrm {R R}) := 2 \cdot \frac {\text {R e c a l l} \cdot \mathrm {R R}}{\text {R e c a l l} + \mathrm {R R}}. \tag {8}
$$

# 4.2 Performance Guarantees

The notion of margin plays a central role in our analysis. It corresponds to our intuition that $f ^ { * }$ is a good estimator of the similarity relation $\sim _ { R }$ if it confidently identifies matches and non-matches.

Definition 4.1 ( $\theta$ -margin condition). For a fixed $\theta >$ 0, given classifiers $\{ k _ { t } ^ { * } \} _ { t = 1 } ^ { T }$ and convex weights $\{ \alpha _ { t } ^ { * } \} _ { t = 1 } ^ { T }$ we say that the similarity function $f ^ { * }$ defined in (2) has $\theta$ -margin if, with probability at least $1 - \eta$ over the choice of $( A , B , y )$ , it holds that

$$
\begin{array}{l} f ^ {*} (A, B) > + \theta , \text {i f} y = + 1, \quad \text {a n d} \\ f ^ {*} (A, B) <   - \theta , \text {i f} y = - 1. \\ \end{array}
$$

Our first theorem shows that the function $f ^ { * }$ obtained in the boosting step via Algorithm 1, satisfies the $\theta$ -margin condition with high probability. The value of $\eta$ depends on the Rademacher complexity [Bartlett and Mendelson, 2002] of the base classifiers $\kappa$ and the samples $\textstyle S _ { A , n } ~ : = ~ \{ A _ { i } \} _ { i = 1 } ^ { n }$ and $\begin{array} { r l } { S _ { B , n } } & { { } : = } \end{array}$ $\{ B _ { i } \} _ { i = 1 } ^ { n }$ given by

$$
\mathfrak {R} _ {\mathcal {S} _ {A, n}} (\mathcal {K}) = \frac {1}{n} \mathbb {E} _ {\sigma} \left[ \sup  _ {k \in \mathcal {K}} \sum_ {i = 1} ^ {n} \sigma_ {i} y _ {i} k (A _ {i}) \right],
$$

$$
\mathfrak {R} _ {\mathcal {S} _ {\mathcal {B}, n}} (\mathcal {K}) = \frac {1}{n} \mathbb {E} _ {\sigma} \left[ \sup  _ {k \in \mathcal {K}} \sum_ {i = 1} ^ {n} \sigma_ {i} y _ {i} k (B _ {i}) \right],
$$

where $\sigma _ { 1 } , \ldots , \sigma _ { n }$ are independent Rademacher random variables, i.e., uniformly chosen in $\{ - 1 , 1 \}$ .

Theorem 4.2 (Performance of the boosting step). With probability at least $1 ~ - ~ \delta$ , the function $f ^ { * }$ corresponding to the output of Algorithm 1 satisfies the $\theta$ -margin condition with the value of $\eta : =$ $\eta _ { t r a i n } ( f ^ { * } , S _ { t r a i n , n } , \theta , \delta )$ given by:

$$
\begin{array}{l} \eta := 2 ^ {T} \prod_ {t = 1} ^ {T} \varepsilon_ {t} ^ {1 / 2 - \theta} (1 - \varepsilon_ {t}) ^ {\theta - 1 / 2} \\ + \frac {8}{\theta} \left(\Re_ {\mathcal {S} _ {\mathcal {A}, n}} (\mathcal {K}) + \Re_ {\mathcal {S} _ {\mathcal {B}, n}} (\mathcal {K})\right) + \sqrt {\frac {\log (1 / \delta)}{2 n}}, \\ \end{array}
$$

where $\varepsilon _ { t } > 0$ are the errors defined in Algorithm 1.

Furthermore, if there exists $\gamma > 0$ such that for all $t \in [ T ]$ , $\gamma \le ( 1 / 2 - \varepsilon _ { t } )$ and $\theta \leq 2 \gamma$ , then the first term in the right-hand side above decreases exponentially with $T$ .

Intuitively, the product term in the definition of $\eta _ { \mathrm { t r a i n } } ( f , S _ { \mathrm { t r a i n } , n } , \theta , \delta )$ is a margin bound over the training data. When $\begin{array} { r c l } { \varepsilon _ { t } } & { \leq } & { 1 / 2 \ - \ \gamma } \end{array}$ for all $t$ , the term will decay exponentially fast in $T$ for suitable choices of margin parameters $\theta$ . The other terms in $\eta _ { \mathrm { t r a i n } } ( f , S _ { \mathrm { t r a i n } , n } , \theta , \delta )$ correspond to a generalization bound used for the test error.

Our next result shows that when Definition 4.1 holds, then the proposed hashing steps of BlockBoost produce high values of the Recall and RR metrics in expectation, with a suitable choice of hyperparameters.

Theorem 4.3 (Performance Weighted Hamming distance hashing). Consider databases $\boldsymbol { A }$ and $\boldsymbol { \beta }$ such that $| \mathcal { A } | = N _ { \mathcal { A } }$ , $| B | = N _ { B }$ and let $\mathcal { M }$ be the set of matching pairs as in 7. For given $\theta > 0$ if the output $f ^ { * }$ of $A l$ - gorithm 1 satisfies the $\theta$ -margin condition and we set $1 - \theta > \delta$ in 3.2, then BlockBoost achieves

$$
\begin{array}{l} \mathbb {E} \left[ \mathrm {R R} \right] \geq (1 - \eta) \left(1 - \frac {| \mathcal {M} |}{N _ {\mathcal {A}} \cdot N _ {\mathcal {B}}}\right), \\ \mathbb {E} \left[ \text {R e c a l l} \right] \geq 1 - \eta \\ \end{array}
$$

where expectations are with respect to the training step, and Recall and RR are defined in (5) and (6).

Note that, in entity matching problems, it is usually the case that $| { \mathcal { M } } | \ll N _ { \mathcal { A } } \cdot N _ { \mathcal { B } }$ . Thus, the expected RR is close to 1, meaning that only a few comparisons have to be made.

# 4.3 Algorithmic Complexity and Speed

We now analyze the algoritmic complexity and speed of BlockBoost in each step.

Boosting. We use stump functions (1) as the family of base classifiers. As described in [Mohri et al., 2012], to determine the stump with the minimal weighted error at each round of boosting we can presort each component in ${ \mathcal { O } } ( n \log n )$ time with a total computational cost of $\mathcal { O } ( n d \log n )$ . For a given component, there are only $n + 1$ possible distinct thresholds, since two thresholds between the same consecutive component values are equivalent. To find the best threshold at each round of boosting, all of these possible $n { \mathrel { + { 1 } } }$ values can be compared, which can be done in ${ \mathcal { O } } ( n )$ time. Thus, the total computational complexity of the algorithm for $T$ rounds of boosting is $\mathcal { O } ( n d \log n + n d T )$ .

Hashing and Blocking. Our binary hashing, with their low number of bits, allow for faster construction of candidate pairs compared to floating-point vectorizations. As a comparison, DeepBlocker’s algorithm [Thirumuruganathan et al., 2021a], one of the fastest blocking alternatives, uses a NVIDIA V100 GPU and the well-optimized FAISS library, and still took 34 minutes for a dataset of 1 million entries [Thirumuruganathan et al., 2021b]. In contrast, BlockBoost achieved the same empirical performance in 2 minutes using a consumer-grade i7 CPU.

# 5 EXPERIMENTS

Our code is available at https://github.com/ thiagorr162/blockboost. See the Supplementary Material for further details regarding the experiments.

# 5.1 Datasets

We make use of canonical blocking datasets that are sourced from a broad spectrum of domains and span a wide range of sizes as shown in Table 1. They are all publicly available and have been used in previous work on entity matching [Steorts et al., 2014, Steorts and Shrivastava, 2018, Thirumuruganathan et al., 2021a, Christen, 2012b, Christen, 2012a, Köpcke et al., 2010]. Further datasets are considered in the Supplementary Material.

The datasets are divided into train, validation, and test sets, with $1 5 \%$ , $1 5 \%$ , and 70% of the total entries, respectively. Additionally, we ensure that the distribution of matches in each fold aligns with these proportions. The train set is used for models that require a training phase and the validation set is employed for adjusting hyperparameters, as needed. Note Block-Boost only has the single parameter $\delta$ in (4); in practice, $T$ is chosen large enough that Algorithm 1 hits the stopping condition in line 8. Lastly, the test set is utilized to evaluate the performance metrics defined in Section 4.1.

We opted to use small training and validation proportions to more accurately represent a real-world scenario where a blocking model must be applied, but there are limited labeled examples available.

# 5.2 Benchmark Models

BlockBoost is compared against the following wellknown methods for blocking in entity matching.

• Canopy [McCallum et al., 2000]: groups records into blocks based on the similarity of certain fields, using a clustering algorithm. For our experiments, we utilized the implementation in [CanopyByPython, 2018].   
• K-Means locality-sensitive hashing (KLSH) [Paulevé et al., 2010]: uses $k$ -means algorithm to construct a low-dimensional projection of the data. We use the code in [klsh, 2020] for the algorithm.   
• Transitive locality-sensitive hashing (TLSH) [Oliver et al., 2013] uses a community detection technique to find similar entities. Our code follows [tlsh, 2020], which implements the work [Steorts et al., 2014].   
• Spectral hashing (Spect) [Weiss et al., 2008b]: this learn to hash method uses a graph partitioning relaxation that is closely related to semantic hashing. We use the implementation in [LearnHash, 2018].   
• AGHasher (AG) [Liu et al., 2011]: a learn to hash method automatically finds compact hash codes using graph-based neighborhood structure in the data. Our experiments follow the code in [AGHasher, 2022].   
• DeepBlocker [Thirumuruganathan et al., 2021a]: a state-of-the-art blocking algorithm for entity matching using deep learning. We included the three distinct neural network architectures available as individual models: CTT, Autoencoder


[ImageDescription]
- source: images/8977fe774c19895280d91ff34ecb5bb2d77e1c2a0b1db32f8e486a27004fb263.jpg
- alt: (no-alt)
- description: Image found in markdown. Detailed vision caption is unavailable in this runtime.
  
(a) Training cost in second.


[ImageDescription]
- source: images/ce562286c4a19962034a039d4f914e0d83c2793ee339ed381af79d2228b911ff.jpg
- alt: (no-alt)
- description: Image found in markdown. Detailed vision caption is unavailable in this runtime.
  
(b) Embedding cost in seconds.


[ImageDescription]
- source: images/fc7fb421b6ee255884ac0c7255b6daf140715e040d0a7c9a2eb61e6b07a46374.jpg
- alt: (no-alt)
- description: Image found in markdown. Detailed vision caption is unavailable in this runtime.
  
(c) Blocking cost in seconds.   
Figure 2: BlockBooster’s cost in seconds on an i7 processor, as the size of the artificially created dataset based on restaurant gets larger. For $n = 1 0 ^ { 6 }$ , BlockBoost’s blocking step takes at most 116 seconds, while DeepBlocker’s CTT takes over 34 minutes on an NVIDIA V100 GPU.

Table 1: Baseline datasets for blocking, and compression (in bits) achieved by BlockBoost’s hashes over the original set of features   

<table><tr><td>DATASET</td><td>ENTITIES</td><td>MATCHES</td><td>TABLES</td><td>FEATURES</td><td>BLOCKBOOST BIT COMPRESSION</td></tr><tr><td>ABTBuy</td><td>2,173</td><td>1,097</td><td>2</td><td>4</td><td>9×</td></tr><tr><td>AMZ_GG</td><td>4,589</td><td>1,300</td><td>2</td><td>5</td><td>23×</td></tr><tr><td>DBLP_ACM</td><td>4,908</td><td>2,224</td><td>2</td><td>4</td><td>8×</td></tr><tr><td>DBLP_SCH</td><td>66,879</td><td>5,347</td><td>2</td><td>4</td><td>5×</td></tr><tr><td>RESTAURANT</td><td>865</td><td>752</td><td>1</td><td>4</td><td>2×</td></tr><tr><td>RLDATA500</td><td>500</td><td>50</td><td>1</td><td>8</td><td>7×</td></tr><tr><td>RLDATA10K</td><td>10,000</td><td>1,000</td><td>1</td><td>8</td><td>6×</td></tr><tr><td>MUSICBRAINZ</td><td>19,375</td><td>10,000</td><td>5</td><td>7</td><td>3×</td></tr><tr><td>WM_AMZ</td><td>24,583</td><td>1,145</td><td>2</td><td>29</td><td>101×</td></tr></table>

(AE), and Hybrid, which integrates both the CTT and Autoencoder models. We use the official code repository provided by the authors [Thirumuruganathan et al., 2021b].

# 5.3 Vectorization

To ensure consistency with the original implementations of the baseline blocking models, we employed two distinct types of vectorization.

Shingling and MinHash. This vectorization is used for Canopy, TLSH and KSLH baseline models, as described in [Steorts et al., 2014, Steorts and Shrivastava, 2018] and implemented in [klsh, 2020, tlsh, 2020]. We first apply the shingling technique to construct a sparse numerical representation of our textual data. Then, we apply the MinHash algorithm [Broder et al., 2000, Steorts et al., 2014] to transform our sparse numerical information into a dense one.

SIF Embedding. We utilized this vectorization technique for both DeepBlocker and our own model, Block-Boost. SIF (Smooth Inverse Frequency) embedding [Arora et al., 2017] is a vector space model commonly used in NLP tasks, which assigns weights to each word vector based on their frequency in a reference corpus.

This weighting scheme allows the resulting document vectors to capture the semantic meaning of the text while reducing the impact of frequent but less informative words.

# 5.4 Training Setup

For most of our benchmark models, training is performed on the training data fold. The exception is DeepBlocker, which does not require training data to work and instead relies on generating artificial training data. In particular, entities in the test set are compared against other, distinct entities in the test set (these will be assumed to be non-matches) or against a slightly perturbed version of the original entity (these will be declared matches). This allows the method to train on a significantly larger number of examples. Further details regarding this methodology are explained in detail in [Thirumuruganathan et al., 2021a]. BlockBoost works with either of these training setups. Below, we mimic DeepBlocker’s approach so our method also does not require training data to work. We employ a ratio of 16 negative examples to each positive example in all our experiments.

# 5.5 Blocking Performance

Table 2 displays the performance of BlockBoost against all the benchmarks on the datasets from Table 1. Note each of DeepBlocker’s possible configurations (CTT, AE, Hybrid) are considered separately. Overall, BlockBoost (BB) displays the best average harmonic mean of Recall and RR, and even when it is not the top-performing algorithm, it is generally close to it. The Supplementary Material includes the individual values of Recall and RR out of which the harmonic mean was derived. Note that only DeepBlocker, in its best possible network configuration, was able to rival BlockBoost’s performance.

# 5.6 Computational Performance

Even when BlockBoost is trained on millions of pairs, it only takes a couple of minutes on a modern CPU, and, as Figure 2a shows, it scales well as $n$ gets larger. This is crucial for a blocking algorithm, since the number of pairs to be matched in real-world examples can often become unwieldy (and are, after all, the main motivation for blocking algorithms). After that, the process of mapping entries from the initial vectorization to the binary embedding is so fast that it’s dominated by the low level C function atof, which converts text to floats from the csv file (see Figure 2b).

The binary nature of our embedding, coupled with the low number of bits that the boosting produces, means that the set of candidate pairs can be constructed much faster than it would be possible with floating-point based vectorizations (see Table 1 for the bit compression Blockboost’s hashes achieve versus the original feature set on each dataset). As a consequence, BlockBoost on CPU achieves a 17 $\times$ speedup over DeepBlocker using an NVIDIA V100 GPU. To make sense of this, note that our binary embeddings, evaluated at Table 2, have an average size of 158 bits, and a maximum size of 256 bits. This is particularly useful for any x86 CPU with SSE4, since they can compute the number of 1s in a word in a single cycle. Modern GPUs also have instructions dedicated to this operation.

Finally, since the dimensions of the embedding are ordered by importance, the least significant bits can be trimmed to make the operations even faster.

# 5.7 Scalability

The primary motivation for employing blocking methods is to effectively handle large datasets. Here, we present a comparative analysis of timing between BlockBoost and baseline models using the musicbrainz dataset across varying sizes: 20k, 200k,

and 2m entries, with BlockBoost emerging as the most scalable option by a significant margin.

For the dataset comprising 2 million entries, only BlockBoost managed to generate predictions within the allocated time (11 hours) and memory constraints (32GB). Table 3 displays performance metrics across increasing dataset sizes (20k, 200k, 2m), underscoring a key advantage of BlockBoost: not only does it deliver high accuracy, but it also operates at orders of magnitude faster speeds compared to other benchmarks.

# 5.8 Interpretability

At each iteration $t = 1 , \dots , T$ of the boosting process, BlockBoost produces a stump function $k _ { t } ^ { * }$ and a weight $\alpha _ { t } ^ { * }$ . The weight can be interpreted as an indication of how significant the stump is for matching entities. Additionally, as the stump is defined by a projected feature $j _ { t } ^ { * }$ and a threshold $\xi _ { t } ^ { * }$ , the weight also indicates how strongly such feature correlates with the similarity relation between items. Thus, the $\alpha _ { t } ^ { * }$ can be understood as giving an automatic feature importance.


[ImageDescription]
- source: images/48d6f5afdb942bd8456d44c42baa34bb34ac5333a91626fcba6367da37ff38dd.jpg
- alt: (no-alt)
- description: Image found in markdown. Detailed vision caption is unavailable in this runtime.
  
Figure 3: Feature relevance identified by BlockBoost during the boosting step for the musicbrainz dataset.

This can be very helpful when interpreting which features carry most of the signal for finding matches. This also allows for discarding irrelevant features for added speed, as well as simple attribute matching based on the selected ones. As an example, Figure 3 shows the importance of some features in the musicbrainz dataset when each one is vectorized separately. Here, importance is understood by the sum of weights associated to each feature.

Intuitively, if someone intends to correlate songs from various databases, BlockBoost considers the length and title as the most critical attributes. The track number on a CD emerges as a surprisingly reliable third option. This is because it’s a numeric value and is typically automatically extracted, whereas other entries involve textual data and potential manual input. Notably, by generating hashes exclusively from the concatenation of the length and number (both numeric

Table 2: Comparison table of the evaluation metric H(Recall, RR), defined in 8, with the best model per dataset in bold. BlockBoost (BB) has the best performance overall.   

<table><tr><td>DATASET</td><td>BB</td><td>CANOPY</td><td>KLSH</td><td>TLSH</td><td>SPECT</td><td>AG</td><td>CTT</td><td>AE</td><td>HYBRID</td></tr><tr><td>ABTBuy</td><td>0.911</td><td>0.761</td><td>0.365</td><td>0.625</td><td>0.263</td><td>0.503</td><td>0.907</td><td>0.817</td><td>0.822</td></tr><tr><td>AMZ_GG</td><td>0.877</td><td>0.605</td><td>0.515</td><td>0.281</td><td>0.518</td><td>0.539</td><td>0.810</td><td>0.849</td><td>0.849</td></tr><tr><td>DBLP_ACM</td><td>0.993</td><td>0.850</td><td>0.895</td><td>0.861</td><td>0.662</td><td>0.696</td><td>0.993</td><td>0.996</td><td>0.998</td></tr><tr><td>DBLP_SCH</td><td>0.989</td><td>0.891</td><td>0.691</td><td>0.543</td><td>0.602</td><td>0.670</td><td>0.991</td><td>0.980</td><td>0.983</td></tr><tr><td>RESTAURANT</td><td>0.988</td><td>0.785</td><td>0.937</td><td>0.838</td><td>0.519</td><td>0.728</td><td>0.997</td><td>0.997</td><td>0.997</td></tr><tr><td>RLDATA500</td><td>0.992</td><td>0.829</td><td>0.969</td><td>0.982</td><td>0.691</td><td>0.717</td><td>0.966</td><td>0.966</td><td>0.966</td></tr><tr><td>RLDATA10K</td><td>0.999</td><td>0.929</td><td>0.926</td><td>0.987</td><td>0.755</td><td>0.800</td><td>0.957</td><td>0.928</td><td>0.926</td></tr><tr><td>MUSICBRAINZ</td><td>0.991</td><td>0.101</td><td>0.944</td><td>0.950</td><td>0.662</td><td>0.737</td><td>0.994</td><td>0.992</td><td>0.992</td></tr><tr><td>WM_AMZ</td><td>0.943</td><td>0.017</td><td>0.495</td><td>0.005</td><td>0.577</td><td>0.558</td><td>0.943</td><td>0.917</td><td>0.942</td></tr><tr><td>AVERAGE</td><td>0.965</td><td>0.641</td><td>0.749</td><td>0.675</td><td>0.583</td><td>0.660</td><td>0.951</td><td>0.938</td><td>0.942</td></tr></table>

Table 3: Benchmark against baseline models in the musicbrainz dataset, with 20k, 200k and 2m entries. Instances exceeding 11 hours are categorized as Out of Time (OOT), while those surpassing 32gb are designated as Out of Memory (OOM). BlockBoost-1bi includes computational optimizations, as detailed in Section D of the Supplementary Material.   

<table><tr><td>Model</td><td>Time 20k</td><td>Time 200k</td><td>Time 2m</td><td>H(Recall, RR) 2m</td></tr><tr><td>blockboost-1bi</td><td>0.55 sec</td><td>5.61 sec</td><td>1 min 55 sec</td><td>0.9887</td></tr><tr><td>blockboost</td><td>4.3 sec</td><td>45.4 sec</td><td>14 min 76 sec</td><td>0.9895</td></tr><tr><td>deepblocker</td><td>12 min 35 sec</td><td>2 hrs 17 min 57 sec</td><td>OOM</td><td>OOM</td></tr><tr><td>tlsh</td><td>2 min 18 sec</td><td>52 min 23 sec</td><td>OOT</td><td>OOT</td></tr><tr><td>klsh</td><td>14 min 41 sec</td><td>OOT</td><td>OOT</td><td>OOT</td></tr><tr><td>canopy</td><td>14 min 10 sec</td><td>OOM</td><td>OOM</td><td>OOM</td></tr></table>

fields, although not an immediately obvious choice), one can achieve a high blocking performance with a H(Recall, RR) close to 0.91.

# 6 CONCLUSION

This paper introduces BlockBoost, a novel blocking method for entity resolution that is data-driven, efficient, and scalable. The results show that combining boosting techniques with hashing leads to a powerful blocking method that empirically outperforms stateof-the-art hashing and deep learning models in terms of scalability and performance over several canonical blocking datasets.

Beyond speed and efficiency, the algorithm is also interpretable and theoretically sound. It outputs weights that reflect variable importance measures, providing users with insight into the most relevant data features for the blocking process. BlockBoost also comes with guaranteed lower bounds on each performance through an extension of margin theory results.

Finally, BlockBoost is able to generate hashes whose size are automatically determined in a data-dependent fashion, leading to great compression and fast retrieval time. Since the dimensions of the hash are ordered by

importance, one can trim the hashes for added time and better compression.

As a possible future avenue for this line of research, we believe BlockBoost can be adapted to many other applications beyond entity matching, such as image retrieval and similarity search, while carrying the same advantages as presented above.

# References

[AGHasher, 2022] AGHasher (2022). Repository: Aghasher. https://github.com/dstein64/ aghasher.   
[Andoni and Beaglehole, 2021] Andoni, A. and Beaglehole, D. (2021). Learning to hash robustly, guaranteed.   
[Andoni and Indyk, 2006] Andoni, A. and Indyk, P. (2006). Near-optimal hashing algorithms for approximate nearest neighbor in high dimensions. In 2006 47th Annual IEEE Symposium on Foundations of Computer Science (FOCS’06), pages 459–468.   
[Arora et al., 2017] Arora, S., Liang, Y., and Ma, T. (2017). A simple but tough-to-beat baseline for sentence embeddings. In 5th International Conference

on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings. OpenReview.net.   
[Azzalini et al., 2020] Azzalini, F., Jin, S., Renzi, M., and Tanca, L. (2020). Blocking techniques for entity linkage: A semantics-based approach. Data Sci. Eng., 6:20–38.   
[Bartlett and Mendelson, 2002] Bartlett, P. and Mendelson, S. (2002). Rademacher and Gaussian complexities: Risk bounds and structural results. Journal of Machine Learning Research, 3(Nov):463–482.   
[Broder et al., 2000] Broder, A. Z., Charikar, M., Frieze, A. M., and Mitzenmacher, M. (2000). Minwise independent permutations. Journal of Computer and System Sciences, 60(3):630–659.   
[CanopyByPython, 2018] CanopyByPython (2018). Repository: Canopy. https://github.com/ AlanConstantine/CanopyByPython.   
[Charikar, 2002] Charikar, M. S. (2002). Similarity estimation techniques from rounding algorithms. In Proceedings of the Thiry-Fourth Annual ACM Symposium on Theory of Computing, STOC ’02, page 380–388, New York, NY, USA. Association for Computing Machinery.   
[Christen, 2012a] Christen, P. (2012a). Data Matching: Concepts and Techniques for Record Linkage, Entity Resolution, and Duplicate Detection. Springer Publishing Company, Incorporated.   
[Christen, 2012b] Christen, P. (2012b). A survey of indexing techniques for scalable record linkage and deduplication. IEEE Transactions on Knowledge and Data Engineering, 24(9):1537–1555.   
[Christophides et al., 2020] Christophides, V., Efthymiou, V., Palpanas, T., Papadakis, G., and Stefanidis, K. (2020). An overview of end-toend entity resolution for big data. ACM Comput. Surv., 53(6).   
[Clark, 2004] Clark, D. E. (2004). Practical introduction to record linkage for injury research. Injury Prevention, 10(3):186–191.   
[Elmagarmid et al., 2007] Elmagarmid, A. K., Ipeirotis, P. G., and Verykios, V. S. (2007). Duplicate record detection: A survey. IEEE Transactions on Knowledge and Data Engineering, 19(1):1–16.   
[Fellegi and Sunter, 1969] Fellegi, I. P. and Sunter, A. B. (1969). A theory for record linkage. Journal of the American Statistical Association, 64(328):1183– 1210.

[Freund and Schapire, 1997] Freund, Y. and Schapire, R. E. (1997). A decision-theoretic generalization of on-line learning and an application to boosting. J. Comput. Syst. Sci., 55(1):119–139.   
[Har-Peled et al., 2012] Har-Peled, S., Indyk, P., and Motwani, R. (2012). Approximate nearest neighbor: Towards removing the curse of dimensionality. Theory of Computing, 8(14):321–350.   
[Jiang et al., 2014] Jiang, K., Que, Q., and Kulis, B. (2014). Revisiting kernelized locality-sensitive hashing for improved large-scale image retrieval.   
[Jonas and Harper, 2006] Jonas, J. and Harper, J. C. (2006). Effective counterterrorism and the limited role of predictive data mining.   
[Kelman et al., 2002] Kelman, C., Bass, J., and Holman, C. (2002). Research use of linked health data - a best practice protocol. Australian and New Zealand journal of public health, 26:251–5.   
[Kim et al., 2020] Kim, S., Yang, H., and Kim, M. (2020). Boosted locality sensitive hashing: Discriminative binary codes for source separation. In ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 106–110.   
[klsh, 2020] klsh (2020). Repository: Klsh. https: //github.com/cleanzr/klsh.   
[Köpcke et al., 2010] Köpcke, H., Thor, A., and Rahm, E. (2010). Evaluation of entity resolution approaches on real-world match problems. Proc. VLDB Endow., 3(1–2):484–493.   
[Kulis and Darrell, 2009] Kulis, B. and Darrell, T. (2009). Learning to hash with binary reconstructive embeddings. In Bengio, Y., Schuurmans, D., Lafferty, J., Williams, C., and Culotta, A., editors, Advances in Neural Information Processing Systems, volume 22. Curran Associates, Inc.   
[Kulis and Grauman, 2009] Kulis, B. and Grauman, K. (2009). Kernelized locality-sensitive hashing for scalable image search. In 2009 IEEE 12th International Conference on Computer Vision, pages 2130– 2137.   
[LearnHash, 2018] LearnHash (2018). Repository: Learnhash. https://github.com/galad-loth/ LearnHash.   
[Liu et al., 2024] Liu, H., Zhou, W., Wu, Z., Zhang, S., Li, G., and Li, X. (2024). Refining codes for locality sensitive hashing. IEEE Transactions on Knowledge and Data Engineering, 36(3):1274–1284.

[Liu et al., 2011] Liu, W., Wang, J., Kumar, S., Chang, S.-F., and Scheffer, T. (2011). Hashing with graphs proceedings of the 28th international conference on machine learning. ICML 2011, pages 1–8.   
[McCallum et al., 2000] McCallum, A., Nigam, K., and Ungar, L. H. (2000). Efficient clustering of high-dimensional data sets with application to reference matching. In Proceedings of the Sixth ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD ’00, page 169– 178, New York, NY, USA. Association for Computing Machinery.   
[Mohri et al., 2012] Mohri, M., Rostamizadeh, A., and Talwalkar, A. (2012). Foundations of Machine Learning. The MIT Press.   
[Mudgal et al., 2018] Mudgal, S., Li, H., Rekatsinas, T., Doan, A., Park, Y., Krishnan, G., Deep, R., Arcaute, E., and Raghavendra, V. (2018). Deep learning for entity matching: A design space exploration. In Proceedings of the 2018 International Conference on Management of Data, SIGMOD ’18, page 19–34, New York, NY, USA. Association for Computing Machinery.   
[O’Hare et al., 2019] O’Hare, K., Jurek-Loughrey, A., and Campos, C. d. (2019). A Review of Unsupervised and Semi-supervised Blocking Methods for Record Linkage, pages 79–105. Springer International Publishing, Cham.   
[Oliver et al., 2013] Oliver, J., Cheng, C., and Chen, Y. (2013). Tlsh – a locality sensitive hash. In 2013 Fourth Cybercrime and Trustworthy Computing Workshop, pages 7–13.   
[Papadakis et al., 2020] Papadakis, G., Skoutas, D., Thanos, E., and Palpanas, T. (2020). Blocking and filtering techniques for entity resolution: A survey. ACM Comput. Surv., 53(2).   
[Paulevé et al., 2010] Paulevé, L., Jégou, H., and Amsaleg, L. (2010). Locality sensitive hashing: a comparison of hash function types and querying mechanisms. Pattern Recognition Letters, 31(11):1348– 1358.   
[Shakhnarovich, 2005] Shakhnarovich, G. (2005). Learning task-specific similarity.   
[Steorts and Shrivastava, 2018] Steorts, R. C. and Shrivastava, A. (2018). Probabilistic blocking with an application to the syrian conflict.   
[Steorts et al., 2014] Steorts, R. C., Ventura, S. L., Sadinle, M., and Fienberg, S. E. (2014). A comparison of blocking methods for record linkage.

[Thirumuruganathan et al., 2021a] Thirumuruganathan, S., Li, H., Tang, N., Ouzzani, M., Govind, Y., Paulsen, D., Fung, G., and Doan, A. (2021a). Deep learning for blocking in entity matching: A design space exploration. Proc. VLDB Endow., 14(11):2459–2472.   
[Thirumuruganathan et al., 2021b] Thirumuruganathan, S., Li, H., Tang, N., Ouzzani, M., Govind, Y., Paulsen, D., Fung, G., and Doan, A. (2021b). Repository: Deepblocker. https://github.com/qcri/DeepBlocker.   
[tlsh, 2020] tlsh (2020). Repository: Tlsh. https: //github.com/cleanzr/tlsh.   
[Wang et al., 2018] Wang, J., Zhang, T., song, j., Sebe, N., and Shen, H. T. (2018). A survey on learning to hash. IEEE Transactions on Pattern Analysis and Machine Intelligence, 40(4):769–790.   
[Weiss et al., 2008a] Weiss, Y., Torralba, A., and Fergus, R. (2008a). Spectral hashing. In Koller, D., Schuurmans, D., Bengio, Y., and Bottou, L., editors, Advances in Neural Information Processing Systems, volume 21. Curran Associates, Inc.   
[Weiss et al., 2008b] Weiss, Y., Torralba, A., and Fergus, R. (2008b). Spectral hashing. Advances in neural information processing systems, 21.   
[Winkler, 2004] Winkler, W. E. (2004). Methods for evaluating and creating data quality. Information Systems, 29(7):531–550. Data Quality in Cooperative Information Systems.   
[Winkler, 2006] Winkler, W. E. (2006). Overview of record linkage and current research directions. Technical report, BUREAU OF THE CENSUS.   
[Zhang et al., 2020] Zhang, W., Wei, H., Sisman, B., Dong, X. L., Faloutsos, C., and Page, D. (2020). AutoBlock. In Proceedings of the 13th International Conference on Web Search and Data Mining. ACM.

# Checklist

1. For all models and algorithms presented, check if you include:

(a) A clear description of the mathematical setting, assumptions, algorithm, and/or model. [✓Yes/No/Not Applicable]   
(b) An analysis of the properties and complexity (time, space, sample size) of any algorithm. [✓Yes/No/Not Applicable]   
(c) (Optional) Anonymized source code, with specification of all dependencies, including external libraries. [✓Yes/No/Not Applicable]

2. For any theoretical claim, check if you include:

(a) Statements of the full set of assumptions of all theoretical results. [✓Yes/No/Not Applicable]   
(b) Complete proofs of all theoretical results. [✓Yes/No/Not Applicable]   
(c) Clear explanations of any assumptions. [✓Yes/No/Not Applicable]

3. For all figures and tables that present empirical results, check if you include:

(a) The code, data, and instructions needed to reproduce the main experimental results (either in the supplemental material or as a URL). [✓Yes/No/Not Applicable]   
(b) All the training details (e.g., data splits, hyperparameters, how they were chosen). [✓Yes/No/Not Applicable]   
(c) A clear definition of the specific measure or statistics and error bars (e.g., with respect to the random seed after running experiments multiple times). [✓Yes/No/Not Applicable]   
(d) A description of the computing infrastructure used. (e.g., type of GPUs, internal cluster, or cloud provider). [✓Yes/No/Not Applicable]

4. If you are using existing assets (e.g., code, data, models) or curating/releasing new assets, check if you include:

(a) Citations of the creator If your work uses existing assets. [✓Yes/No/Not Applicable]   
(b) The license information of the assets, if applicable. [✓Yes/No/Not Applicable]   
(c) New assets either in the supplemental material or as a URL, if applicable. [✓Yes/No/Not Applicable]   
(d) Information about consent from data providers/curators. [Yes/No/✓Not Applicable]

(e) Discussion of sensible content if applicable, e.g., personally identifiable information or offensive content. [Yes/No/✓Not Applicable]

5. If you used crowdsourcing or conducted research with human subjects, check if you include:

(a) The full text of instructions given to participants and screenshots. [Yes/No/✓Not Applicable]   
(b) Descriptions of potential participant risks, with links to Institutional Review Board (IRB) approvals if applicable. [Yes/No/✓Not Applicable]   
(c) The estimated hourly wage paid to participants and the total amount spent on participant compensation. [Yes/No/✓Not Applicable]

# A PROOFS FOR SECTION 4

Proof of Theorem 4.2. We need to define the following sets:

$$
\mathcal {K} _ {2} := \left\{f _ {k}: (A, B) \mapsto k (A) k (B), k \in \mathcal {K} \right\},
$$

and the convex hull conv( $\mathrm { { { \chi _ { 2 } } } }$ ) of $\mathrm { { { \ / { C _ { 2 } } } } }$ given by,

$$
\begin{array}{l} \operatorname {c o n v} \left(\mathcal {K} _ {2}\right) := \left\{f: (A, B) \mapsto \sum_ {t = 1} ^ {T} \alpha_ {t} f _ {k _ {t}} (A, B): T \geq 1, \alpha_ {t} \geq 0, f _ {k _ {t}} \in \mathcal {K} _ {2}, \sum_ {t = 1} ^ {T} \alpha_ {t} = 1 \right\} \\ = \left\{f: (A, B) \mapsto \sum_ {t = 1} ^ {T} \alpha_ {t} k _ {t} (A) k _ {t} (B): T \geq 1, \alpha_ {t} \geq 0, k _ {t} \in \mathcal {K}, \sum_ {t = 1} ^ {T} \alpha_ {t} = 1 \right\}. \\ \end{array}
$$

To show that the output of Algorithm 1 indeed satisfies Condition 4.1 we prove the following stronger result.

Theorem A.1. Consider an iid sample $\boldsymbol { S _ { t r a i n , n } } = ( ( a _ { i } , b _ { i } ) , y _ { i } ) _ { i = 1 } ^ { n }$ with $( ( a _ { i } , b _ { i } ) , y _ { i } )$ drawn from $P$ . Then, given $\theta \in ( 0 , 1 )$ and $\delta \in ( 0 , 1 )$ , with probability at least $1 - \delta$ , for any $f \in c o n v ( \mathcal { K } _ { 2 } )$

$$
P [ y f (A, B) \leq \theta ] \leq \eta_ {t r a i n} (f, \mathcal {S} _ {t r a i n, n}, \theta , \delta),
$$

where

$$
\eta_ {t r a i n} (f, \mathcal {S} _ {t r a i n, n}, \theta , \delta) := \frac {1}{n} \sum_ {i = 1} ^ {n} \mathbf {1} _ {[ y _ {i} f (a _ {i}, b _ {i}) \leq 2 \theta ]} + \frac {8}{\theta} \Re_ {\mathcal {S} _ {t r a i n, n}} (\mathcal {K}) + \sqrt {\frac {\log (1 / \delta)}{2 n}}.
$$

Proof. First, consider the surrogate margin loss function given by:

$$
\varphi_ {\theta} (x) = \min  \left(1, \max  \left(1 - \frac {x}{\theta}, 0\right)\right).
$$

and the following set:

$$
\Phi_ {\theta} := \left\{\varphi_ {\theta , f}: ((a, b), y) \mapsto \varphi_ {\theta} (y f (a, b) - \theta): f \in \operatorname {c o n v} (\mathcal {K} _ {2}) \right\}.
$$

By Rademacher Inequality [3], we have that with probability at least $1 - \delta$ , for all $f \in \mathrm { c o n v } ( K _ { 2 } )$ :

$$
\mathbb {E} \left[ \varphi_ {\theta} (y f (A, B) - \theta) \right] \leq \frac {1}{n} \sum_ {i = 1} ^ {n} \varphi_ {\theta} (y _ {i} f (a _ {i}, b _ {i}) - \theta) + 2 \Re_ {\mathcal {S} _ {\mathrm {t r a i n}, n}} (\Phi_ {\theta}) + \sqrt {\frac {\log (1 / \delta)}{2 n}}.
$$

Using the fact that $\mathbf { 1 } _ { [ x \leq \theta ] } \leq \varphi _ { \theta } ( x - \theta )$ , we have that with probability at least $1 - \delta$ , for all $f \in \mathrm { c o n v } ( K _ { 2 } )$

$$
P [ y f (a, b) \leq \theta ] \leq \frac {1}{n} \sum_ {i = 1} ^ {n} \varphi_ {\theta} (y _ {i} f (a _ {i}, b _ {i}) - \theta) + 2 \Re_ {\mathcal {S} _ {\mathrm {t r a i n}, n}} (\Phi_ {\theta}) + \sqrt {\frac {\log (1 / \delta)}{2 n}}.
$$

Since $\varphi \theta$ is $1 / \theta$ -Lipschitz, by Talagrand’s Lemma and the fact that $\Re _ { S _ { \mathrm { t r a i n } , n } } ( \mathrm { c o n v } ( \mathcal { K } _ { 2 } ) ) = \Re _ { S _ { \mathrm { t r a i n } , n } } ( \mathcal { K } _ { 2 } )$ [1], we have with probability at least $1 - \delta$ , for all $f \in \mathrm { c o n v } ( K _ { 2 } )$ :

$$
P [ y f (A, B) \leq \theta ] \leq \frac {1}{n} \sum_ {i = 1} ^ {n} \varphi_ {\theta} (y _ {i} f (a _ {i}, b _ {i}) - \theta) + \frac {2}{\theta} \Re_ {\mathcal {S} _ {\mathrm {t r a i n}, n}} (\mathcal {K} _ {2}) + \sqrt {\frac {\log (1 / \delta)}{2 n}}.
$$

Using the fact that $\varphi _ { \theta } ( x - \theta ) \leq \mathbf { 1 } _ { [ x \leq 2 \theta ] }$ , with probability at least $1 - \delta$ , for all $f \in \mathrm { c o n v } ( K _ { 2 } )$ :

$$
P [ y f (A, B) \leq \theta ] \leq \frac {1}{n} \sum_ {i = 1} ^ {n} {\bf 1} _ {[ y _ {i} f (a _ {i}, b _ {i}) \leq 2 \theta ]} + \frac {2}{\theta} \Re_ {\mathcal {S} _ {\mathrm {t r a i n}, n}} (\mathcal {K} _ {2}) + \sqrt {\frac {\log (1 / \delta)}{2 n}}.
$$

Now we just need to bound $\mathfrak { R } _ { S _ { \mathrm { t r a i n } , n } } ( \mathcal { K } _ { 2 } )$ in terms of $\mathfrak { R } _ { S _ { \mathrm { t r a i n } , n } } ( { \cal K } )$ so our final result depends only on the Rademacher complexity of the family $\kappa$ which is usually known. But note that

$$
\begin{array}{l} \mathfrak {R} _ {\mathcal {S} _ {\mathrm {t r a i n}, n}} (\mathcal {K} _ {2}) = \frac {1}{n} \mathbb {E} _ {\sigma} \left[ \sup _ {k \in \mathcal {K} _ {2}} \sum_ {i = 1} ^ {n} \sigma_ {i} y _ {i} k (a _ {i}) k (b _ {i}) \right] \\ = \frac {1}{n} \mathbb {E} _ {\sigma} \left[ \sup _ {k \in \mathcal {K}} \sum_ {i = 1} ^ {n} \sigma_ {i} y _ {i} k (a _ {i}) k (b _ {i}) \right] \\ \leq \frac {1}{n} \mathbb {E} _ {\sigma} \left[ \sup  _ {k _ {1}, k _ {2} \in \mathcal {K}} \sum_ {i = 1} ^ {n} \sigma_ {i} y _ {i} k _ {1} \left(a _ {i}\right) k _ {2} \left(b _ {i}\right) \right] \\ = \frac {1}{n} \mathbb {E} _ {\sigma} \left[ \sup  _ {k _ {1}, k _ {2} \in \mathcal {K}} \sum_ {i = 1} ^ {n} \sigma_ {i} k _ {1} (a _ {i}) k _ {2} (b _ {i}) \right] \\ = \frac {1}{n} \mathbb {E} _ {\sigma} \left[ \sup  _ {k _ {1}, k _ {2} \in \mathcal {K}} \sum_ {i = 1} ^ {n} \sigma_ {i} \left(1 - \frac {\left(k _ {1} \left(a _ {i}\right) - k _ {2} \left(b _ {i}\right)\right) ^ {2}}{2}\right) \right] \\ = 0 + \frac {1}{n} \mathbb {E} _ {\sigma} \left[ \sup  _ {k _ {1}, k _ {2} \in \mathcal {K}} \sum_ {i = 1} ^ {n} \sigma_ {i} \frac {\left(k _ {1} \left(a _ {i}\right) - k _ {2} \left(b _ {i}\right)\right) ^ {2}}{2} \right] \\ = \frac {1}{2 n} \mathbb {E} _ {\sigma} \left[ \sup _ {k _ {1}, k _ {2} \in \mathcal {K}} \sum_ {i = 1} ^ {n} \sigma_ {i} L \left(k _ {1} (a _ {i}) - k _ {2} (b _ {i})\right) \right] \\ \end{array}
$$

where,

$$
L (x) = \left\{ \begin{array}{l} x ^ {2}, \text {i f} x \in [ - 2, 2 ] \\ 4, \text {o t h e r w i s e .} \end{array} \right.
$$

Since $L$ is 4-Lipschitz, by Talagrand’s Lemma, we have that, with probability at least $1 - \delta$ , for all $f \in \mathrm { c o n v } ( K _ { 2 } )$

$$
P [ y f (A, B) \leq \theta ] \leq \frac {1}{n} \sum_ {i = 1} ^ {n} {\bf 1} _ {[ y _ {i} f ^ {*} (a _ {i}, b _ {i}) \leq 2 \theta ]} + \frac {8}{\theta} \left(\mathfrak {R} _ {\mathcal {S A}, n} (\mathcal {K}) + \mathfrak {R} _ {\mathcal {S B}, n} (\mathcal {K})\right) + \sqrt {\frac {\log (1 / \delta)}{2 n}}.
$$

# B DATASET DETAILS

Below, we provide an overview of each database utilized in our research paper. In all the datasets, every entry includes both record information and a distinct record ID, which serves as a unique identifier for each entry across the entire database, encompassing all potential tables. Additionally, each entry is associated with an entity ID, allowing us to identify and group together entities that are identical, regardless of their location in different tables.

The code necessary to download and process each dataset can be accessed from our GitHub repository https: //github.com/thiagorr162/blockboost.

# abt_buy

This dataset contains name, description, manufacturer and price of product data from abt.com and buy.com. It is available at https://dbs.uni-leipzig.de/research/projects/object_matching/benchmark_datasets_ for_entity_resolution.

# amz_gg

This dataset contains name, description, manufacturer and price product data from Amazon and Google. It can be downloaded via https://dbs.uni-leipzig.de/research/projects/object_matching/benchmark_ datasets_for_entity_resolution.

# dblp_acm

This dataset contains title, authors, venue and year information of bibliographic data from DBLP and ACM. We downloaded the dataset from https://dbs.uni-leipzig.de/research/projects/object_matching/ benchmark_datasets_for_entity_resolution.

# dblp_sch

This dataset contains title, authors, venue and year information of bibliographic data from DBLP and Google Scholar. It is available at https://dbs.uni-leipzig.de/research/projects/object_matching/benchmark_ datasets_for_entity_resolution.

# musicbrainz

This dataset contains number, title, length, artist, album, year and language information of music data from MusicBrainz. This dataset can be found at https://dbs.uni-leipzig.de/research/projects/object_ matching/benchmark_datasets_for_entity_resolution.

# restaurant

This dataset contains name, address, location and cuisine type of restaurants data. It can be downloaded from https://github.com/cleanzr/restaurant.

# rldata

These datasets contain individuals’ first and last names, as well as their birth year, birth month, and birth day. It is available at https://github.com/cran/RecordLinkage/.

wm_amz

This dataset comprises a wide range of product information from both Walmart and Amazon. It includes details such as brand, title, shelf description, short and long descriptions, model number, weight, and various other relevant attributes. It can be found at https://github.com/anhaidgroup/deepmatcher/blob/master/ Datasets.md.

# C REDUCTION RATIO AND RECALL VALUES

In this section, we show the Recall, RR and H(Recall, RR) values for all models across all datasets. The results are sorted in descending order based on the H(Recall, RR) value.

# C.1 Dataset abt_buy

<table><tr><td>Model</td><td>Recall</td><td>RR</td><td>H(Recall, RR)</td></tr><tr><td>blockboost</td><td>0.8967</td><td>0.9265</td><td>0.9113</td></tr><tr><td>ctt</td><td>0.8401</td><td>0.9868</td><td>0.9076</td></tr><tr><td>hybrid</td><td>0.7046</td><td>0.9868</td><td>0.8222</td></tr><tr><td>ae</td><td>0.6978</td><td>0.9868</td><td>0.8175</td></tr><tr><td>canopy</td><td>0.7199</td><td>0.8072</td><td>0.7610</td></tr><tr><td>tlsh</td><td>0.5058</td><td>0.8209</td><td>0.6259</td></tr><tr><td>agrapher</td><td>0.4770</td><td>0.5325</td><td>0.5032</td></tr><tr><td>klsh</td><td>0.2283</td><td>0.9175</td><td>0.3656</td></tr><tr><td>spectral</td><td>0.1558</td><td>0.8434</td><td>0.2631</td></tr></table>

# C.2 Dataset amz_gg

<table><tr><td>Model</td><td>Recall</td><td>RR</td><td>H(Recall, RR)</td></tr><tr><td>blockboost</td><td>0.8312</td><td>0.9293</td><td>0.8775</td></tr><tr><td>hybrid</td><td>0.7515</td><td>0.9779</td><td>0.8499</td></tr><tr><td>ae</td><td>0.7515</td><td>0.9779</td><td>0.8499</td></tr><tr><td>ctt</td><td>0.6856</td><td>0.9912</td><td>0.8106</td></tr><tr><td>canopy</td><td>0.6284</td><td>0.5836</td><td>0.6052</td></tr><tr><td>agrapher</td><td>0.5120</td><td>0.5706</td><td>0.5397</td></tr><tr><td>spectral</td><td>0.5165</td><td>0.5199</td><td>0.5182</td></tr><tr><td>klsh</td><td>0.3933</td><td>0.7465</td><td>0.5152</td></tr><tr><td>tlsh</td><td>0.1669</td><td>0.8940</td><td>0.2813</td></tr></table>

# C.3 Dataset dblp_acm

<table><tr><td>Model</td><td>Recall</td><td>RR</td><td>H(Recall, RR)</td></tr><tr><td>hybrid</td><td>0.9994</td><td>0.9969</td><td>0.9981</td></tr><tr><td>ae</td><td>0.9968</td><td>0.9969</td><td>0.9968</td></tr><tr><td>ctt</td><td>0.9910</td><td>0.9969</td><td>0.9939</td></tr><tr><td>blockboost</td><td>0.9883</td><td>0.9969</td><td>0.9926</td></tr><tr><td>klsh</td><td>0.8160</td><td>0.9923</td><td>0.8956</td></tr><tr><td>tlsh</td><td>0.7691</td><td>0.9784</td><td>0.8612</td></tr><tr><td>canopy</td><td>1.0000</td><td>0.7397</td><td>0.8504</td></tr><tr><td>agraphisher</td><td>0.7230</td><td>0.6723</td><td>0.6967</td></tr><tr><td>spectral</td><td>0.7706</td><td>0.5813</td><td>0.6627</td></tr></table>

# C.4 Dataset dblp_sch

<table><tr><td>Model</td><td>Recall</td><td>RR</td><td>H(Recall, RR)</td></tr><tr><td>ctt</td><td>0.9836</td><td>0.9989</td><td>0.9912</td></tr><tr><td>blockboost</td><td>0.9840</td><td>0.9944</td><td>0.9891</td></tr><tr><td>hybrid</td><td>0.9673</td><td>0.9999</td><td>0.9833</td></tr><tr><td>ae</td><td>0.9623</td><td>0.9999</td><td>0.9807</td></tr><tr><td>canopy</td><td>0.8670</td><td>0.9179</td><td>0.8917</td></tr><tr><td>klsh</td><td>0.5292</td><td>0.9989</td><td>0.6919</td></tr><tr><td>aghasher</td><td>0.7031</td><td>0.6404</td><td>0.6703</td></tr><tr><td>spectral</td><td>0.4742</td><td>0.8264</td><td>0.6026</td></tr><tr><td>tlsh</td><td>0.3730</td><td>0.9990</td><td>0.5431</td></tr></table>

# C.5 Dataset musicbrainz

<table><tr><td>Model</td><td>Recall</td><td>RR</td><td>H(Recall, RR)</td></tr><tr><td>ctt</td><td>0.9908</td><td>0.9979</td><td>0.9943</td></tr><tr><td>hybrid</td><td>0.9869</td><td>0.9979</td><td>0.9924</td></tr><tr><td>ae</td><td>0.9865</td><td>0.9979</td><td>0.9922</td></tr><tr><td>blockboost</td><td>0.9866</td><td>0.9952</td><td>0.9909</td></tr><tr><td>tlsh</td><td>0.9053</td><td>1.0000</td><td>0.9503</td></tr><tr><td>klsh</td><td>0.8946</td><td>1.0000</td><td>0.9444</td></tr><tr><td>aghasher</td><td>0.7280</td><td>0.7477</td><td>0.7377</td></tr><tr><td>spectral</td><td>0.6162</td><td>0.7161</td><td>0.6624</td></tr><tr><td>canopy</td><td>0.0532</td><td>1.0000</td><td>0.1010</td></tr></table>

# C.6 Dataset restaurant

<table><tr><td>Model</td><td>Recall</td><td>RR</td><td>H(Recall, RR)</td></tr><tr><td>ctt</td><td>1.0000</td><td>0.9952</td><td>0.9976</td></tr><tr><td>hybrid</td><td>1.0000</td><td>0.9944</td><td>0.9972</td></tr><tr><td>ae</td><td>1.0000</td><td>0.9944</td><td>0.9972</td></tr><tr><td>blockboost</td><td>1.0000</td><td>0.9778</td><td>0.9888</td></tr><tr><td>klsh</td><td>0.9038</td><td>0.9744</td><td>0.9378</td></tr><tr><td>tlsh</td><td>0.7308</td><td>0.9822</td><td>0.8380</td></tr><tr><td>canopy</td><td>0.7660</td><td>0.8057</td><td>0.7854</td></tr><tr><td>agrapher</td><td>0.6410</td><td>0.8428</td><td>0.7282</td></tr><tr><td>spectral</td><td>0.3590</td><td>0.9416</td><td>0.5198</td></tr></table>

# C.7 Dataset rldata500

<table><tr><td>Model</td><td>Recall</td><td>RR</td><td>H(Recall, RR)</td></tr><tr><td>blockboost</td><td>0.9857</td><td>0.9993</td><td>0.9925</td></tr><tr><td>tlsh</td><td>0.9857</td><td>0.9790</td><td>0.9823</td></tr><tr><td>klsh</td><td>0.9714</td><td>0.9681</td><td>0.9697</td></tr><tr><td>ctt</td><td>0.9429</td><td>0.9921</td><td>0.9669</td></tr><tr><td>ae</td><td>0.9429</td><td>0.9906</td><td>0.9662</td></tr><tr><td>hybrid</td><td>0.9429</td><td>0.9906</td><td>0.9661</td></tr><tr><td>canopy</td><td>0.8429</td><td>0.8168</td><td>0.8296</td></tr><tr><td>agraphasher</td><td>0.6571</td><td>0.8267</td><td>0.7322</td></tr><tr><td>spectral</td><td>0.6571</td><td>0.7298</td><td>0.6916</td></tr></table>

# C.8 Dataset rldata10000

<table><tr><td>Model</td><td>Recall</td><td>RR</td><td>H(Recall, RR)</td></tr><tr><td>blockboost</td><td>0.9982</td><td>0.9995</td><td>0.9988</td></tr><tr><td>tlsh</td><td>0.9811</td><td>0.9940</td><td>0.9875</td></tr><tr><td>ctt</td><td>0.9214</td><td>0.9956</td><td>0.9571</td></tr><tr><td>canopy</td><td>0.8964</td><td>0.9642</td><td>0.9291</td></tr><tr><td>ae</td><td>0.8700</td><td>0.9962</td><td>0.9288</td></tr><tr><td>klsh</td><td>0.8768</td><td>0.9820</td><td>0.9264</td></tr><tr><td>hybrid</td><td>0.8643</td><td>0.9981</td><td>0.9264</td></tr><tr><td>agraphasher</td><td>0.7486</td><td>0.8601</td><td>0.8005</td></tr><tr><td>spectral</td><td>0.6829</td><td>0.8462</td><td>0.7558</td></tr></table>

# C.9 Dataset wm_amz

<table><tr><td>Model</td><td>Recall</td><td>RR</td><td>H(Recall, RR)</td></tr><tr><td>ctt</td><td>0.9012</td><td>0.9903</td><td>0.9436</td></tr><tr><td>blockboost</td><td>0.9301</td><td>0.9571</td><td>0.9434</td></tr><tr><td>hybrid</td><td>0.8995</td><td>0.9903</td><td>0.9427</td></tr><tr><td>ae</td><td>0.8492</td><td>0.9968</td><td>0.9171</td></tr><tr><td>spectral</td><td>0.4992</td><td>0.6845</td><td>0.5773</td></tr><tr><td>agrapher</td><td>0.5126</td><td>0.6138</td><td>0.5586</td></tr><tr><td>klsh</td><td>0.3476</td><td>0.8634</td><td>0.4956</td></tr><tr><td>canopy</td><td>0.0088</td><td>0.9903</td><td>0.0174</td></tr><tr><td>tlsh</td><td>0.0028</td><td>0.9987</td><td>0.0056</td></tr></table>

# D BLOCKING AT SCALE

In this section, we explore how BlockBoost performs for very large datasets. In this case, several of the benchmarking blocking algorithms either become out of time (over 11 hours) or out of memory (over 32GB). For this reason, these experiments were not included in the main paper, but we include them in the Supplementary Material as they give further evidence of BlockBoost’s scalability.

In small datasets, increasing the size of the training set by choosing random pairs as non-matches is a good unsupervised strategy to improve the predictive performance, since the chance of picking matches is very low. However, using a training dataset larger than a couple of million entries yields diminishing returns and can hinder scalability. Moreover, very large sets of candidate sets might not be desirable in some practical applications. To represent the two aforementioned scenarios, i.e. small and large datasets respectively, two experiments were included:

• BlockBoost: For each matching pair in the training dataset, select 16 non-matches. The best maximum hamming distance is selected using a validation dataset.   
• BlockBoost-1bi: For each matching pair in the training dataset, select 1 non-match. The maximum hamming distance is also selected using a validation dataset, but the possibilities are restricted to values that produce a set of candidate pairs with less than 1 billion entries.

As shown in Figure 1, the price in recall of limiting the size of the set of candidate pairs is low in the musicbrainz_2m dataset, and this restriction prevents IO bottlenecks. All of the benchmarks ran on an Intel(R) Core(TM) i7-10700 CPU @ 2.90GHz, with 32 GB of DDR4 – 2666 MT/s.

# D.1 Predictive Performance – musicbrainz_2m

In this subsection, we analyze the predictive performance of blocking benchmarks, as well as BlockBoost, over the data set musicbrainz_2m, with 2 million entries, which can be downloaded in https://dbs.uni-leipzig. de/research/projects/object_matching/benchmark_datasets_for_entity_resolution. As claimed in the

main text, we find that BlockBoost can scale to this size and still maintain a competitive performance in terms of recall and reduction ratio.

# D.1.1 Recall and Reduction Ratio

<table><tr><td>Model</td><td>Recall</td><td>RR</td><td>H(Recall, RR)</td></tr><tr><td>ctt</td><td>OOM</td><td>OOM</td><td>OOM</td></tr><tr><td>hybrid</td><td>OOM</td><td>OOM</td><td>OOM</td></tr><tr><td>ae</td><td>OOM</td><td>OOM</td><td>OOM</td></tr><tr><td>blockboost</td><td>0.9848</td><td>0.9941</td><td>0.9895</td></tr><tr><td>blockboost-1bi</td><td>0.9791</td><td>0.9984</td><td>0.9887</td></tr><tr><td>tlsh</td><td>OOT</td><td>OOT</td><td>OOT</td></tr><tr><td>klsh</td><td>OOT</td><td>OOT</td><td>OOT</td></tr><tr><td>canopy</td><td>OOM</td><td>OOM</td><td>OOM</td></tr></table>


[ImageDescription]
- source: images/60c7a9c04055c89b214c200d856abe4992c3bd89021fefe614ebfea3bcee2c3d.jpg
- alt: (no-alt)
- description: Image found in markdown. Detailed vision caption is unavailable in this runtime.
  
Figure 1: Recall as a function of the size of the set of candidate pairs in the musicbrainz_2m dataset, with proportion of non-matches of 1. Note the the maximum value of $x$ is 1/100th of the total number of possible pairs. Computing the reduction ratio and recall for all of the possible maximum hamming distances takes 113.3 seconds on this dataset.

# D.1.2 Bit Compression

Here we show the compression (in bits) achieved by BlockBoost’s hashes over the original set of features.

• BlockBoost: 11x   
• BlockBoost-1bi: 12x

# D.2 Time

In this section, we compare BlockBoost against other benchmark models on the musicbrainz_20 dataset with 20,000 entries, musicbrainz_200 dataset with 200,000 entries, and musicbrainz_2m dataset with 2 million entries, showcasing the execution time in each case. Each of these datasets can be downloaded from https://dbs. uni-leipzig.de/research/projects/object_matching/benchmark_datasets_for_entity_resolution.

• musicbrainz_20k

```txt
klsh: 14 min 41 sec  
tlish: 2 min 18 sec  
canopy: 14 min 10 sec  
DeepBlocker: 12 min 35 sec  
blockboost: 4.3 sec (size of train = 112234)  
blockboost-1bi: .55 sec (size of train = 13204) 
```

```txt
- musicbrainz_200k
    - klsh: OOT
    - tlsh: 52 min 23 sec
    - canopy: OOM
    - DeepBlocker: 2 hrs 17 min 57 sec
    - blockboost: 45.4 sec (size of train = 1112820)
    - blockboost-1bi: 5.61 sec (size of train = 130920) 
```

```python
- musicbrainz_2m
    - klsh: OOT
    - tlsh: OOT
    - canopy: OOM
    - DeepBlocker: OOM
    - blockboost: 14 min 76 sec (size of train = 11032354)
    - blockboost-1bi: 1 min 55 sec (size of train = 1297924) 
```

# E BLOCKING VS LEARNING-TO-HASH

In this section, we discuss important differences between the fields of blocking and learning-to-hash (as well as some similarities).

Blocking involves grouping together items that are considered similar based on a specific metric so that it is possible to forgo a quadratic number of comparisons and only focus on comparing entries within the same block,as we detail in Sections 1 and 2 in the main paper. For instance, if the goals it to match customer purchase records to customer accounts, blocks can be created using attributes such as last name or ZIP code of the billing address, or a combination of these. Well-designed blocks can greatly enhance the speed and efficiency of the matching process. One way to create such blocks is via hashing (although there are important alternatives). An effective hash code ensures that similar items are grouped together by mapping them to the same hash code, while dissimilar items are assigned different hash codes.

While the field of learning-to-hash also deals with developing effective hash functions, it is typically concerned with the nearest neighbors problem, rather than blocking. The nearest neighbor problem focuses on finding the most similar data points to a given query point within a dataset regardless of their specific identities or relationships. In contrast, blocking is a technique used to group together similar items based on shared attributes or criteria.

This difference gives more structure to blocking problems, which are typically exploited by benchmark algorithms. For instance, in entity matching, which is a typical application for blocking, there exists a specific notion of similarity, i.e., duplicated records that represent the same entity. In this context, the occurrence of duplicate entries often follows specific patterns, such as typos or minor textual variations. Understanding these patterns is crucial as it allows us to develop tailored techniques that improve the blocking process. Examples of such techniques include the use of shingling and minhash vectorization [5, 4], as well as the construction of artificial training sets, which is the case of DeepBlocker [6]. These approaches depend significantly on these specific textual heuristics to achieve successful outcomes, and it is not immediately obvious how to apply them to other learning-to-hash problems.

It is also important to highlight that the research tools employed in the literature vary between blocking and learning-to-hash. These two approaches have different focuses and evaluation criteria, leading to the use of distinct benchmark models and databases for assessing their performance. For instance, in blocking, in addition to recall, we also consider the reduction ratio metric. If we apply this measure to a widely used datasets in learningto-hash literature like MNIST or CIFAR-10, we observe that since there are only 10 classes of objects, the best possible reduction ratio for each class would be 90%. This would represent a perfect blocking scenario, but even then, we would still need to perform numerous comparisons within each block (e.g. $( 1 0 \% \times 6 0 , 0 0 0 ) ^ { 2 } = 3 6 , 0 0 0 , 0 0 0$ in CIFAR-10). This is not the case for most blocking problems, where we usually have just a few duplicated entries, easily leading to reduction ratio metrics close to 99.9% (as is the case for BlockBoost in some of the datasets in the paper).

In spite of these differences, one can use still apply learning-to-hash for blocking. As argued above, we expect that they result in suboptimal performance; that is indeed what happens in the paper with two traditional learning-to-hash algorithms (Aghasher and Spectral Hashing). In all of the datasets considered, BlockBoost enjoyed better performance and speed. Conversely, as alluded to in the paper’s conclusion, one could adapt BlockBoost to solve typical learning-to-hash problems; that is an avenue for future work.

# F BLOCKBOOST WITH LSH-INSPIRED BLOCKING

In this section, we present an alternative to the weighted hamming distance hashing step in the original paper based on the theory of Locality-sensitive hashing (LSH). We believe that this alternative solution can be used to solve classic problems in the learning-to-hash field.

The algorithm: The algorithm produces $k$ -bit hash functions $g _ { 1 } , \ldots , g _ { L }$ such that

$$
g _ {1, j} = k _ {j} ^ {*} \text {w i t h p r o b a b i l i t y} \alpha_ {j} ^ {*}. \tag {1}
$$

Items $( A , B )$ will be part of the same block if there exists at least one hash function $g _ { i }$ with $g _ { i } ( A ) = g _ { i } ( B )$

Algorithm 1 LSH inspired hashing   
Input: $k,L\in \mathbb{N}$ , convex weights $(\alpha_{t}^{*})_{t = 1}^{T}$ , classifiers $(k_{t}^{*})_{t = 1}^{T}$ 1: for $i\gets 1$ to $L$ do   
2: for $j\gets 1$ to $k$ do   
3: $g_{i,j}\gets k_t^*$ with probability $\alpha_{t}^{*}$ 4: end for   
5: $g_{i}\gets (g_{i,1},\dots ,g_{i,k})$ 6: end for   
Output: $(g_{1},\ldots ,g_{L})$

Why it works: To understand why this hashing technique works, note that, given $f ^ { * }$ as in (2), for any $( A , B ) \in { \mathcal { A } } \times B$ , and any function $g _ { i , j }$ constructed as before, since $g _ { i , j }$ is $\{ - 1 , + 1 \}$ -valued and $g _ { i , j } = k _ { t } ^ { * }$ with probability $\alpha _ { t }$ for each $t \in [ T ]$ , it is easy to show that

$$
\mathbb {P} _ {g _ {i, j}} \left[ g _ {i, j} (A) = g _ {i, j} (B) \right] = \mathbb {E} _ {g _ {i, j}} \left[ \frac {1 + g _ {i , j} (A) g _ {i , j} (B)}{2} \right] = \frac {1}{2} + \frac {f ^ {*} (A , B)}{2}.
$$

This implies that, if $f ^ { * } ( A , B )$ properly approximates the similarity notion $A \sim _ { R } B$ , it is expected that $1 \approx$ $\mathbb { P } _ { g _ { i , j } } \left[ g _ { i , j } ( A ) = g _ { i , j } ( B ) \right] \geq p _ { 1 } > 1 / 2$ for most similar pairs, whereas $0 \approx \mathbb { P } \left[ g _ { i , j } ( A ) = g _ { i , j } ( B ) \right] \leq p _ { 2 } < 1 / 2$ for most dissimilar pairs. This intuition can be made precise (see Supplementary Material, where values $p _ { 1 }$ and $p _ { 2 }$ will be derived from a margin property of the function $f ^ { * }$ ). Here, $k$ and $L$ are hyperparameters used to amplify the gap between the values $p _ { 1 }$ and $p _ { 2 }$ .

Lemma F.1. Let $f ^ { * }$ be as in (2). Then for any $( A , B ) \in { \mathcal { A } } \times B$ , and any function $g _ { i , j }$ as in Algorithm 1,

$$
\mathbb {P} _ {g _ {i, j}} \left[ g _ {i, j} (A) = g _ {i, j} (B) \right] = \frac {1 + f ^ {*} (A , B)}{2},
$$

where the probability is over the choice of $g _ { i , j }$

Proof. Since $g _ { i , j }$ is $\{ - 1 , + 1 \}$ -valued,

$$
\mathbb {P} _ {g _ {i, j}} \left[ g _ {i, j} (A) = g _ {i, j} (B) \right] = \mathbb {E} _ {g _ {i, j}} \left[ \frac {1 + g _ {i , j} (A) g _ {i , j} (B)}{2} \right].
$$

Now recall that $g _ { i , j } = k _ { t } ^ { * }$ with probability $\alpha _ { t }$ for each $t \in [ T ]$ .

Theorem F.2 (Performance of the LSH inspired hashing). Consider databases $\mathcal { A }$ and $\boldsymbol { B }$ such that $| \mathcal { A } | = N _ { \mathcal { A } }$ and $| B | = N _ { B }$ M be the set of matching pairs. For given $\theta > 0$ and $\gamma \in ( 0 , 1 )$ , if the output $f ^ { * }$ of Algorithm 1 satisfies the $\theta$ -margin condition and we set:

$$
\rho := \frac {\log \left(\frac {2}{1 + \theta}\right)}{\log \left(\frac {2}{1 - \theta}\right)} \in [ 0, 1),
$$

$$
k := \left\lceil \log_ {\frac {2}{1 + \theta}} N _ {\mathcal {A}} \cdot N _ {\mathcal {B}} \right\rceil ,
$$

$$
L := \left\lceil \frac {2 (N _ {\mathcal {A}} \cdot N _ {\mathcal {B}}) ^ {\rho} \log (1 / \gamma)}{1 + \theta} \right\rceil ,
$$

then the LSH inspired hashing method achieves

$$
\mathbb {E} [ \text {R e c a l l} ] \geq (1 - \gamma) (1 - \eta)
$$

$$
\mathbb {E} \left[ \mathrm {R R} \right] \geq \left(1 - \frac {| \mathcal {M} | + L}{N _ {\mathcal {A}} \cdot N _ {\mathcal {B}}}\right) (1 - \eta),
$$

where expectations are with respect to the randomness in the hash code, and Recall and RR are defined in (5) and (6).

Proof. This proof is an adaptation of [2]. Since $f ^ { * }$ satisfies Condition 4.1 for $\theta > 0$ , we know by Lemma F.1 that for all $A , B$ in a set $\varepsilon$ of $P$ -measure $\geq 1 - \eta$

$$
\text {i f} A \sim_ {R} B, \text {t h e n} \mathbb {P} _ {g _ {i, j}} [ g _ {i, j} (A) = g _ {i, j} (B) ] \geq \frac {1 + \theta}{2} = p _ {1}
$$

$$
\text {i f} A \not \sim_ {R} B, \text {t h e n} \mathbb {P} _ {g _ {i, j}} [ g _ {i, j} (A) = g _ {i, j} (B) ] \leq \frac {1 - \theta}{2} = p _ {2},
$$

where $P$ is as described in Section 4. For our next calculations assume we are conditioned to this event. Fix $k = \lceil \log _ { 1 / p _ { 2 } } N _ { \mathcal { A } } \cdot N _ { \mathcal { B } } \rceil$ and let $\mathcal { M }$ be the set of matching pairs as defined in (7). We split the proof in the following steps:

• Probability of finding correct matches. Suppose that $A \sim _ { R } B$ and $( A , B ) \in \mathcal { E }$ . By independence, for $i \in [ L ]$

$$
\begin{array}{l} \mathbb {P} _ {g _ {i}} [ g _ {i} (A) = g _ {i} (B) ] = \mathbb {P} _ {g _ {i, j}} [ g _ {i, j} (A) = g _ {i, j} (B) ] ^ {k} \\ \geq p _ {1} ^ {k} \\ \geq p _ {1} ^ {\log_ {1 / p _ {2}} \left(N _ {\mathcal {A}} \cdot N _ {\mathcal {B}}\right) + 1} \\ = p _ {1} p _ {1} ^ {\log_ {1 / p _ {2}} \left(N _ {\mathcal {A}} \cdot N _ {\mathcal {B}}\right)} \\ = p _ {1} \left(N _ {\mathcal {A}} \cdot N _ {\mathcal {B}}\right) ^ {- \rho}, \\ \end{array}
$$

where in the last equality we used a simple logarithm change of basis. That is,

$$
\mathbb {P} _ {g _ {i}} \left[ g _ {i} (A) \neq g _ {i} (B) \right] \leq 1 - p _ {1} \left(N _ {\mathcal {A}} \cdot N _ {\mathcal {B}}\right) ^ {- \rho},
$$

Thus, the probability of finding the correct match is

$$
\begin{array}{l} \mathbb {P} [ \exists i \in \{1, \dots , L \}, g _ {i} (A) = g _ {i} (B) ] = 1 - \mathbb {P} [ \forall i \in \{1, \dots , L \}, g _ {i} (A) \neq g _ {i} (B) ] \\ = 1 - \mathbb {P} _ {g _ {i}} \left[ g _ {i} (A) \neq g _ {i} (B) \right] ^ {L} \\ \geq 1 - \left(1 - p _ {1} \left(N _ {\mathcal {A}} \cdot N _ {\mathcal {B}}\right) ^ {- \rho}\right) ^ {L} \\ \end{array}
$$

hence, by setting $\begin{array} { r } { L = \frac { \log ( 1 / \gamma ) ( N _ { \mathcal { A } } \cdot N _ { \mathcal { B } } ) ^ { \rho } } { p _ { 1 } } } \end{array}$ for $\gamma \in ( 0 , 1 )$ , we have that

$$
\begin{array}{l} \mathbb {P} \left[ \exists i \in \{1, \dots , L \}, g _ {i} (A) = g _ {i} (B) \right] \geq 1 - \left(1 - p _ {1} \left(N _ {\mathcal {A}} \cdot N _ {\mathcal {B}}\right) ^ {- \rho}\right) ^ {L} \\ > 1 - e ^ {- \log (1 / \gamma)} \\ = 1 - \gamma . \\ \end{array}
$$

• Expected Recall. By the previous item, we have

$$
\begin{array}{l} \mathbb {E} \left[ \text {R e c a l l} \right] \geq \frac {1}{| \mathcal {M} |} \sum_ {(\ell , r) \in \mathcal {M}} \mathbb {P} [ \exists i \in \{1, \dots , L \}, g _ {i} (A _ {\ell}) = g _ {i} (B _ {r}) | (A, B) \in \mathcal {E} ] P [ \mathcal {E} ] \\ \geq (1 - \gamma) (1 - \eta). \\ \end{array}
$$

• Probability of finding wrong matches. Suppose that $A \not \sim _ { R } B$ and $( A , B ) \in \mathcal { E }$ . Then, for $i \in [ L ]$

$$
\begin{array}{l} \mathbb {P} _ {g _ {i}} \left[ g _ {i} (A) = g _ {i} (B) \right] = \mathbb {P} _ {g _ {i, j}} \left[ g _ {i, j} (A) = g _ {i, j} (B) \right] ^ {k} \\ \leq p _ {2} ^ {k} \\ \leq \frac {1}{N _ {\mathcal {A}} \cdot N _ {\mathcal {B}}}, \\ \end{array}
$$

by our choice of $k$ .

• Expected number of wrong matches. By the previous item, conditioned to $( A , B ) \in \mathcal { E }$ , the random variable that counts the number wrong matches found by $g _ { i }$

$$
C \left(g _ {i}\right) = \sum_ {(\ell , r) \notin \mathcal {M}} \mathbf {1} _ {\left[ g _ {i} \left(A _ {\ell}\right) = g _ {i} \left(B _ {r}\right) \right]}
$$

follows a binomial distribution with parameter $\left( N _ { \mathcal { A } } \cdot N _ { \mathcal { B } } - | \mathcal { M } | , \frac { 1 } { N _ { \mathcal { A } } \cdot N _ { \mathcal { B } } } \right)$ , hence

$$
\mathbb {E} _ {g _ {i}} \left[ C (g _ {i}) \right] \leq 1,
$$

therefore the number of total wrong collisions for $g _ { i }$ is at most 1 and the number of total wrong collisions for all $g _ { i }$ for $i \in \{ 1 , \ldots , L \}$ is at most $L$ .

• Expected RR. By the previous item and the fact that Condition 4.1 holds with probability $\geq 1 - \eta$ , the expected number of comparisons is

$$
\begin{array}{l} \mathbb {E} [ \# \text {c o m p a r i s o n s} ] \leq \sum_ {(\ell , r) \in [ N _ {\mathcal {A}} ] \times [ N _ {\mathcal {B}} ]} \mathbb {P} [ \exists i \in \{1, \dots , L \}, g _ {i} (A _ {\ell}) = g _ {i} (B _ {r}) | (A, B) \in \mathcal {E} ] P [ (A, B) \in \mathcal {E} ] \\ + P [ (A, B) \notin \mathcal {E} ] \\ \leq \sum_ {(\ell , r) \in \mathcal {M}} \mathbb {P} [ \exists i \in \{1, \dots , L \}, g _ {i} (A _ {\ell}) = g _ {i} (B _ {r}) | (A, B) \in \mathcal {E} ] (1 - \eta) \\ + \sum_ {(\ell , r) \notin \mathcal {M}} \mathbb {P} [ \exists i \in \{1, \dots , L \}, g _ {i} (A _ {\ell}) = g _ {i} (B _ {r}) | (A, B) \in \mathcal {E} ] (1 - \eta) + \eta \\ \leq (| \mathcal {M} | + L) (1 - \eta) + \eta . \\ \end{array}
$$

Therefore, the expected RR satisfies

$$
\mathbb {E} \left[ \mathrm {R R} \right] \geq 1 - \eta - \left(\frac {| \mathcal {M} | + L}{N _ {\mathcal {A}} \cdot N _ {\mathcal {B}}}\right) (1 - \eta).
$$

The number of number of operations and the size of our data structure can be easily estimated using our previous calculations, but can also be found in [2], Theorem 3.4.


[ImageDescription]
- source: images/4d43aa474b00deedd598ef233ece14bdb051407e5bd04b45e5ed83aa605d30e5.jpg
- alt: (no-alt)
- description: Image found in markdown. Detailed vision caption is unavailable in this runtime.


Algorithmic complexity. The number of operations can be derived from the proof of Theorem F.2. It is possible to show that if $\xi : = ( N _ { A } \cdot N _ { B } ) ^ { \rho } / \log \left( 2 / ( 1 + \theta ) \right)$ , then the algorithm requires at most $\mathcal { O } ( \xi )$ distance computations/evaluations of hash functions and the data structure uses at most $\mathcal { O } ( \xi )$ words of space, in addition to the space needed to store the dataset.