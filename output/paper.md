# Credit Card Fraud Detection: A Deep Learning Approach

Sourav Verma

ABV-Indian Institute of

Information Technology and Management

Gwalior-474 010

Email: ipg 2013108@iiitm.ac.in

Joydip Dhar

ABV-Indian Institute of

Information Technology and Management

Gwalior-474 010

Email: jdhar@iiitm.ac.in

Abstract—Credit card is one of the most extensive method of installment for both online and offline mode of payment for electronic transactions in recent times. credit cards invention has provided significant ease in electronic transactions. However, it has also provided new fraud opportunities for criminals, which results in increased fraud rates. Substantial amount of money have been lost by many institutions and individuals due to fraudulent credit card transactions. Adapting improved and dynamic fraud recognition frameworks thus became essential for all credit card distributing banks to mitigate their losses.

In fact, the problem of fraudulent credit card transactions implicates a number of relevant real-time challenges, namely: Concept drift, Class imbalance, and Verification latency. However, the vast majority of current systems are based on artificial intelligence (AI), Fuzzy logic, Machine Learning, Data mining, Genetic Algorithms, and so on, rely on assumptions that hardly address all the relevant challenges of fraud-detection system (FDS).

This paper aims to understand & implement Deep Learning algorithms in order to obtain a high fraud coverage with very low false positive rate. Also it aims to implement an auto-encoder as an unsupervised (semi-supervised) method of learning common patterns.

Keywords: Credit card fraud, Fraud-detection system (FDS), Electronic transactions, Concept drift, Class imbalance, Verification latency, Machine Learning, Deep Learning

# 1. Introduction

Fraud refers to the intentional illegal exploitation of a system which results in loss of an oblivious entity. Credit card fraud can be defined as illegal use of both online and offline mode of credit card information for electronic transactions. Credit card fraud involves the exploitation of credit card systems which results in the loss of financial resources, the most prominent being monetary although other damages such as loss of integrity and authenticity are possible. Fraud, waste, and abuse in many financial

systems are estimated to result in billions of US dollars annually, thus has became a primary concern for financial institutions around the globe.

Furthermore, the rapid growth of the internet has exposed credit card systems to diverse fraudsters using different mechanisms to exploit financial systems. This provided an explode in attack patterns which rendered the once effective Artificial intelligent (AI), Machine Learning and Case-based fraud detection solutions no more effective as the computational complexity increases with each new detected fraud. More seriously, there is a higher tendency for first time frauds going undetected. The Case-based detection methods are also slow as a successful exploit could multiply if the solution took time to be integrated into the system. This problem can only be addressed with a refined and dynamic techniques capable of adapting to rapidly evolving fraudulent patterns.

Also of concern to credit card fraud detection solutions is, the recognition strength that indicates a Fraud detector’s ability to correctly identify both known and novel frauds. This is usually a direct function of how much fraud samples there are to model a solution. The emergence of Deep Learning algorithms has provided credit card fraud detection experts with verse amount of features and high dimentionality of data that will enhance the detection models. Such solutions that use Deep Learning algorithms to model, offer more efficient and adaptive solutions.

A complete credit card fraud detection model thus,

Must have the following properties:

1) It must be Adaptive: This refers to following abilities:

Ability to detect fraudulent patterns in a matter of moment (i.e quickly). This is also referred to as its alertness.   
Ability to detect first time fraudulent patterns with high accuracy and low false positive rate.

2) It must be Dynamic: This refers to following abilities:

Ability to detect all new instances of fraudulent activities with rapid alteration of false patterns.

3) It should be able to identify the fraudulent patterns accurately i.e. the true positive rate should be high.   
4) It should be able to detect the frauds quickly i.e time complexity of the system should be low.   
5) It should not predict a legit transaction as fraud i.e. the false positive rate should be low.

Also must address following relevant challenges, namely:

1) Concept drift: As customers’ habit changes, fraudsters change their approaches over time.   
2) Class imbalance: More legit transactions as compare to frauds (less than $0 . 5 \%$ ).   
3) Verification latency: Only a small set of transactions are timely checked by the authorities.

# 2. Related Works

# 2.1. Frauds and Fraudulent pattern detection

Fraud is a synonym for illicit use of a system to get some benefits, usually resulting in loss to another person. Frauds are as diverse as fraudulent patterns. Credit card fraud is fraud within the financial industry that usually cause monetary losses. The financial industries have been the principal victims of fraudulent activities in recent times. According to [1], substantial amount of money have been lost to insurance fraud. The growth of internet uses have made it easier for fraudsters to divulge and connect in order to target financial institutions from a distance, making it more diverse in nature. This further entangles the threats to credit card security systems, thus fraud detection and prevention are important concern to all financial institutions.

By many estimates, approximately $10 \%$ of insurance company payments are for fraudulent claims, and the global sum of these fraudulent payments, amounts to substantial amount of money.

Fraud detection refers to mechanisms to detect frauds when fraudulent activities occur, while Fraud prevention refers to all measures put in place to prevent frauds from happening, [2]. A necessary requirement for fraud preventive systems is their precision (i.e. predictive measure). Much concern is given to improving the precision of such systems. Fraud detection systems, conversely, need to adapt to the dynamism of threats. Hence in addition to possible predictiveness, Fraud detection systems need to be adaptive to the fraudulent patterns. A related concern usually classified under possible predictiveness is the time complexity of the system to detect fraudulent transactions. Certain systems require near real-time alertness on dubious transactions.

# 2.2. Outlier detection

An outlier is an observation that deviates so much from other observations as to evoke suspicion that it was generated by a different mechanism [3].

Unsupervised learning approaches are used to this model. Usually, the result of unsupervised learning is a new explanation or representation of the original observation data, which then lead to improved future responses or decisions. Unsupervised learning methods do not need prior knowledge of fraudulent and genuine transactions (i.e. labels) in historical databases, instead detect alteration in behavior or unusual transactions. These methods model a baseline distribution that represents normal behavior and then detect observations that show deviation from this norm. Outliers are a basic form of non-standard observation that can be used for fraud detection. In supervised methods, models are trained to discriminate between fraudulent and non-fraudulent behavior so that new observations can be assigned to different classes. Supervised methods require accurate identification of fraudulent transactions in historical databases and can only be used to detect frauds of a type that have previously occurred. An advantage of using unsupervised methods over supervised methods is that previously undiscovered types of fraud may also be detected. Supervised methods are merely trained to discriminate between legit transactions and previously known fraud.

[2] proposed unsupervised credit card fraud detection techniques, using behavioral outlier detection techniques. Anomalous spending behavior and frequency of transactions can be identified as outliers, which could be possible fraud cases.

# 2.3. Rule-based fraud detection

Rule-based methods consist all known fraudulent characteristics and use them to model the fraud detection system (FDS). They are classified as Supervised learning methods as they use previously known fraud to detect similar patterns. Such methods classify transactions using rules made out based on previously detected fraudulent transactions. The process used to adopt such models to evolving threats is manual, and thus such methods are not recommended for persistent threats of these days. An example of such methods include BAYES, RIPPER etc.

Although according to [4], Rule-based fraud analysis can be very tedious to administer because the proper layout of such rules require detailed, onerous, and prolonged programming for each credible fraud instance. The dynamic emergence of multiple new fraud types, demands that these rules be constantly adapted to include existing, emerging, and future fraud options. Moreover, it also presents a major hurdle to scalability. The more data the system must process, the more severe is the performance descents.

# 2.4. Statistical fraud detection

Statistical methods have been used to classify and detect frauds. The transactional data is known to follow a statistical distribution, and thus, transactional data points, that fall out of the normal distribution are considered dubious. Such methods include Linear Discriminant Analysis (LDA) and Logistic Regression [2]. Statistical methods can either be supervised or unsupervised. Supervised methods use known fraudulent cases to model the detector system.

A natural problem with the Statistical method is determining the most relevant distribution to fit a data set (best fit), and with increased dimension of the data, it becomes more difficult to approximate the distribution, [5].

# 2.5. Machine Learning based detection

Machine learning (ML) is the science of getting computers to act without being explicitly programmed. In the past decade, machine learning has given us self-driving cars, practical speech recognition, effective web search, and a vastly improved understanding of the human genome.

ML evolved primarily from Artificial Intelligence (AI) and Soft computing, and also from other fields including applied mathematics, pattern recognition and computational learning theory [6]. ML algorithms are mostly used to handle problems involving automatic data classification [7]. ML algorithms are capable of analyzing data and searching for hidden patterns in data. According to [8] ML algorithms aims to predict patterns from data based on learned experiences. ML algorithms are divided into different classes, namely: supervised learning, unsupervised learning, semisupervised learning, reinforcement learning, transduction and learning to learn [8]. Most of the proposed credit card fraud detection techniques are based on supervised learning and few are based on semi-supervised learning. Some of these techniques are discussed next:

2.5.1. Hidden Markov Model. [9] proposed a technique based on Hidden Markov Model (HMM). In the study, authors used HMM to model a sequence of credit card transactions and divide the transactions into three price ranges (clusters): low (l), medium (m), and high (h). The type of each transaction is linked to the line of business of the corresponding merchant. Afterwards, determine the three probability matrices so that representation of the HMM is complete. These three model parameters are determined in a training phase using the Baum-Welch algorithm [10]. Authors considered the special case of fully connected HMM in which every state of the model can be reached in a single step from every other state.

An HMM is initially trained with the normal behaviour of the cardholder. Thereafter, authors constructed sequences from training data-set and trained the model. In the testing and validation phase, if an incoming credit card transaction is not accepted by the trained HMM with sufficiently high probability, it is considered to be fraudulent. authors also

claim to minimize the true positive rates.

Another proposed technique based on HMM is proposed by [10]. In the study also, authors used HMM to model a sequence of credit card transactions and used K-mean clustering algorithm to cluster the transactions into three price ranges (clusters): low (l), medium (m), and high (h), as proposed by [9] Afterwards, incoming transactions were tested into the trained model and authorized if accepted with sufficiently high probability. Otherwise, transaction will be terminated and IP address of the merchant to be defrauded will be traced using HMM. A notification will be sent to both the merchant system’s administrator and cardholder via mobile communications. Also as per the authors, the HMM was trained with Baum-Welch algorithm.

# 2.5.2. Support vector machines (SVM) based techniques.

[11] performed a comparative study between SVM and decision tree based credit card fraud detection system (FDS). Firstly, authors divided the data-set used into three groups with the ratio of fraudulent transactions to the legit ones in 1:1, 1:4, 1:9 respectively, during the implementation. As usual training to testing data-set was divided by $70 \%$ to $30 \%$ . Authors used four kernels for SVM in the setup. Also they developed seven SVM-based and decision tree based models and tested each of them. Results from experiments revealed that the Decision tree based model outperformed SVM model. The models achieved classification accuracy between the range of 83.02 to $9 4 . 7 6 \%$ .

2.5.3. Frequent item-set mining. [12] proposed a technique based on frequent item-set mining, called Fraud-Miner. Authors separated each customer’s transaction from the whole transactional database and from each customer’s transactions again separated their legit and fraud transactions. Afterwards, applied Apriori algorithm to both the sets of legit and fraud transactions on each customer’s transactions which returns a set of frequent item-sets of both legit and fraud transactions. for testing, authors propose a matching algorithm which spans the legit and fraudulent pattern databases of each customers for a match with the incoming transaction to detect fraud. If a convenient match is found with legit pattern of the corresponding customer, it matches, otherwise not. The experimental result shows that the FraudMiner outperformed the existing solutions eg. SVM, Naive Bias (NB), KNN etc.

2.5.4. Ensemble based technique. [13] proposed a credit card fraud detection model based on bagging ensemble classifier. The primary objective of study was to compare the performance of three different advanced data mining techniques namely: SVM, NB and KNN to bagging ensemble classifier based on decision tree. To counter class imbalance problem authors divided the data-set used into four groups with fraud rates approximately $20 \%$ , $1 5 \%$ , $10 \%$ , $3 \%$ respectively also authors used 10 fold cross validation technique. To compare the results authors weigh the performance of SVM, NB and KNN and compared

with the result obtained by bagging ensemble classifier. The experimental result revealed that bagging ensemble classifier achieved better fraud detection rate and an improved false positive rate.

[14] proposed a credit card fraud detection techniques based on Ensemble classification and extended feature selection. Authors considered both the feature selection and the prediction (decision) cost for accuracy enhancement of the FDS. After selecting best features using an extended wrapper method, an ensemble classification is performed. Authors performed the ensemble classification using cost sensitive decision tree in a decision forest framework. The experimental result revealed that considering the F-measure as the evaluation metric, the proposed approach achieves 1.8 to $2 . 4 \%$ performance improvement compared to the other classifiers.

# 2.6. Nature inspired (NI) based techniques

Nature inspired based methods refers to algorithms inspired by nature’s problem solving ability [15]. In other words, Nature is the source of inspiration to Nature Inspired algorithms. For example, Ant Colony Optimization is inspired by the methods used by ants to seek for pathways between their colony and a food source, Bat algorithm was inspired by the echolocation behaviour of micro-bats, with varying pulse rates of emission and loudness and Genetic Algorithm is inspired by the process of natural selection, that belongs to the larger class of evolutionary algorithms (EA). [16]. NI algorithms are designed to handle complex real world classification and optimization related problems, such as timetabling problem, travelling salesman problems (TSP) and hostel allocation problems [15]. Generally, NI algorithms are used for global optimization. Some NI-based techniques used to provide solution to credit card fraud detection are discussed next.

2.6.1. Genetic algorithm (GA) based techniques. [17] proposed a GA-based credit card FDS with the intent of detecting the fraud with minimum false positive rate. Instead of maximizing the correctly classified transactions authors prescribed an objective function with variable misclassification cost. The objective function intents at minimizing false positive rate. During classification, authors extracted credit card transaction from the database and standardize the data. Afterwords, calculated critical values for each transaction present in the database. Authors also extracted the frequent item-set for credit card usage, indigence, location where the credit card was used, balance on the account linked to credit card, average spending pattern of the credit cardholder from each transaction. Furthermore, authors used Genetic Algorithm (GA) to generate new critical values. Finally, the new critical values were then used for classification.

2.6.2. Artificial neural network (ANN) based techniques. [18] used Simulated annealing (SA) and Back-propagation algorithm (BPA) for Feed-forward Neural Network (FFNN)

to develop a credit card FDS with the intent of hybridizing SA and BPA for FFNN, which can join the symbolic global searching capability of SA with the precise local searching element of back-propagation FFNNs to improve the initial weights of a neural network toward getting a better result for detection fraud. Authors suggest of identifying fraud and legit transactions based on following critical values, namely: credit card usage frequency, number of locations of credit card usage, average credit card indigence and credit card book balance.

Authors randomly initialized the weights of FFNN and evaluated weights using SA, following a temperature annealing schedule with the algorithm. While first temperature value is less than or equal to the minimum error authors selected the best solution. Furthermore, for training, authors initialize the parameters of BP learning algorithm. While the threshold epoch not reached, authors update the weights of BP to minimize the error with training data. Finally, Authors assess the execution of classification with test data to validate the study. The experimental result revealed that BPFFNN with applied SA yielded better false positive rate as compare to the simple BPFFNN algorithm.

# 3. Proposed Methodology

To address the major concerns of a Fraud detection system, the following infrastructural design choices were made.

# 3.1. Deep Learning Model

A deep learning computation model will be used to model FDS. Deep learning models are appropriate here as they serve well for data sets having large amount of data with large no. of features. The ability of such models to learn feature hierarchy composing lower level features into higher level abstractions influenced its choice here. It has the potential to discover sophisticated patterns in large data sets through its self-adjusting back-propagation algorithm.

Furthermore, Deep Learning models address some of the big challenges posed by Big Data computation and analysis, thereby providing efficient means to the use of Big Data.

For the purpose of this task, two Deep Learning models are used to model the credit-card fraud data and to predict fraudulent transactions. The first model will be a Multi-layer feed-forward neural network system. It is built based on the neuron units. The model works by feeding the input data into the first layer. Subsequent layers learn more concrete features from previous layers through non-linear transformations.

# Experiment I: Multi-layer feed-forward supervised learning

This experiment involves tuning a feed-forward deep learning network to model financial data. Parameters such as activation function, number of epochs, hidden layers size will be adjusted until a suitable model is reached. The aim

of the experiment is to find recommendations for Fraud detection system designs.

Choices for activation function include: The Tanh function & Logistic sigmoid function with min-max or z-score normalization. After Using validation set to determine the data standardization approach and the best fit activation functions, According to experiments, It’s found that Tanh performs better than Logistic sigmoid, when using it with z-score normalization. Hence, choosing tanh followed by zscore is the best option. the Epoch numbers will be increased by factors of 10 (1, 10, 100 etc.). Hidden layers will vary between 1 to 50. The mean-squared error (MSE) function or reconstruction error is used as the loss function.

The multi-layer feed-forward Deep Learning model is used to conduct supervised learning with the training data set, which is the majority of the data, splitted in the ratio of 3:1. Design parameters will be varied to determine a recommended design for the FDS on the data set.

# 3.2. Anomaly Detection

The anomaly detection model used here is Deep Autoencoders (DAE) to detect fraudulent pattern in the data-set which is an unsupervised model.

Unsupervised means the model will be trained for both fraud and non-fraud data without feeding the labels. Since the class imbalance is very high in the credit cards, It is expected from the model to learn and memorize the patterns of legit ones after the unsupervised training, and should be able to give a credible score for any transaction as being an outlier. And this unsupervised training would be quite handy in practice especially when we don’t have enough labeled data set. Deep Auto-encoders can be used to pre-train the model before a supervised training.

# Experiment II: Deep Auto-encoders for detecting fraudulent patterns

This experiment involves designing a Deep Auto-encoder to detect anomaly in the credit card data-set. The Deep Auto-encoder learns the pattern in the data-set through non-linear transformations of layers.

To test for anomaly, it reconstructs the test data, anomalous data will deviate a lot from the legit ones and thus will have high error. A deep learning auto-encoder will be trained on $7 5 \%$ of the data set, the remaining $2 5 \%$ will be used to test for the model’s predictions. The mean-squared error (MSE) function or reconstruction error function is used as the loss function.

Reconstruction Error Function:

$$
L \left(x, x ^ {\prime}\right) = \left| \left| x - x ^ {\prime} \right| \right| ^ {2} \tag {1}
$$

# Experiment III: Optimization

This experiment involves optimizing the performance of the proposed deep learning fraud detection model. For

optimizing the performance of the FDS, Nature Inspired Bat Algorithm is used. The aim is to mitigate the training cost and complexity of the FDS. Also to enhance the overall performance of the model.

# Experiment IV: Comparison with different Scikit learn algorithms

This experiment involves comparing different Scikitlearn methods/algorithms to classify fraudulent patterns. These comparisons are done using different performance measures e.g.: AUC scores, confusion matrices and precision-recall curves.

# 4. Implementation & Execution

We trained and tested our proposed FDS using Kaggle’s Credit Card Fraud Detection data-set. The summary of data-set:

<table><tr><td colspan="2">Kaggle: Credit Card Fraud Detection Data-set</td></tr><tr><td>Data Set Characteristics</td><td>Multivariate</td></tr><tr><td>Number of Attributes</td><td>31</td></tr><tr><td>Number of Instances</td><td>284807</td></tr><tr><td>Attribute Characteristics</td><td>Categorical, Float64</td></tr></table>

# 4.1. Data Standardization & Activation Function

Two types of data standardization function is considered here: z-score & min-max normalization. 1) z-score normalization will normalize every column such that the resultant columns will have mean of zero and standardization of ones. And this will be a good choice if we are using Tanh activation function. This will output values on both sides of zero. Furthermore, Tanh activation function will leave values that are too extreme to still keep some outliers left after the normalization process. This might be useful to detect some extremeness in this case.

2) min-max normalization will assure all values to be in the range [0, 1]. min-max is the default scaling approach if we are using sigmoid as our output activation function.

Z-Score Standardization Function:

$$
z - s c o r e = \frac {x - \mu}{\sigma} \tag {2}
$$

Tanh Activation Function:

$$
\tanh  = \frac {e ^ {2 x} - 1}{e ^ {2 x} + 1} \tag {3}
$$

# 4.2. Modeling Auto-encoder as unsupervised learning

• Parameters:

– learning rate $= 0 . 0 1$   
– training epochs $= 6 0$ (Optimal)   
– batch size $= 2 5 6$

– display step $= 1$

• Network Parameters:

– n hidden $. 1 = 1 5$   
– n hidden $2 = 1 5$

• For FC layers:

hidden size $= 4$ (Best hidden size based on validation)   
– output size $= 2$ (classes: 1 & 0)

# 4.3. Optimization

Optimization is done using Nature Inspired Bat Algorithm.

4.3.1. Binary Bat Algorithm. The binary bat algorithm has been inspired by the echolocation behaviour of bats [19]. The characteristics of bats for finding its pray are being used in this algorithm. Bats tend to decrease the loudness and increase the rate of emitting ultrasonic waves, when they chase pray.

In binary bat algorithm each artificial bat has a position vector, a velocity vector and a frequency vector. The position of the bats in binary bat algorithm is either 0 or 1. The velocity can be updated using the following equations:

$$
V _ {i} (t + 1) = V _ {i} (t) + \left(X _ {i} (t) - X ^ {*}\right) F _ {i} \tag {4}
$$

Where Vi , Xi and Fi are the velocity, position and frequency of ith bat. $X ^ { * }$ is the current global best location.

The frequency of the ith bat can be updated using the following formula:

$$
F _ {i} = F _ {\min } + \left(F _ {\max } - F _ {\min }\right) \beta \tag {5}
$$

Where $F _ { m i n }$ is the minimum frequency and $F _ { m a x }$ is the maximum frequency. $\beta$ represents a random number which lies between 0 and 1.

The position of bats can be updated based on following function:

$$
X _ {i} (t + 1) = X _ {i} (t) + V _ {i} (t) \tag {6}
$$

The loudness and pulse rate of binary bat algorithm is A and r. These two variables can be updated as follows:

$$
A _ {i} (t + 1) = \alpha A _ {i} (t) \tag {7}
$$

$$
r _ {i} (t + 1) = r _ {i} (0) [ 1 - \exp (- \gamma t) ] \tag {8}
$$

Where $\alpha$ and $\gamma$ are constants. The loudness and the pulse rate are updated when we optimize the new solutions to ensure that the bats are moving toward the best solutions.

4.3.2. Optimization using Binary Bat Algorithm. By using Nature Inspired Bat Algorithm for feature selection process, It is found that some features can be dropped to mitigate the training cost and the complexity of the entire system. Also it enhances the test AUC score significantly. Features dropped were:

’V28’, ’V27’, ’V26’, ’V25’, ’V24’, ’V23’, ’V22’, ’V20’, ’V15’, ’V13’, ’V8’

Algorithm 1: Pseudo code for Binary Bat Algorithm:   
1 Initialize the bat's position and velocity, $X_{i}$ and $V_{i}$ $(\mathrm{i} = 1,2,3,\dots \mathrm{n})$ 2 Initialize frequency $(F_{i})$ pulse rate $(r_i)$ and loudness $(A_{i})$ 3 while iteration $<$ max_iteration do   
4 Generate new solutions by adjusting frequencies   
5 Update velocity and locations for each solution   
6 if rand $>r_i$ then   
7 Select a solution among the best solution   
8 Generate a local solution among the selected best   
9 end   
10 Generate a new solution by flying (around) randomly   
11 if rand $<  A_{i}$ and $F(X_{i}) <   F(X^{*})$ then   
12 Accept the new solution   
13 increase $r_i$ and reduce $A_{i}$ 14 end   
15 Rank the bats and find current best X\*   
16 end

# 5. Result & Analysis

This section presents the outcomes of the experiments designed to provide Fraud detection systems.

# 5.1. Performance Measures

The following standard evaluation measures were used in describing the predictive Fraud detection models:

1) Mean Square Error (MSE): It is used to assess the quality of a predictor models. It takes values in the range [0, 1]. The goal is to minimize MSE value. A ideal model will have MSE value $= 0$ .   
2) Accuracy: Accuracy is the simplest performance measure that refers to the closeness of a measured value to a standard or known value. Mathematically, it is the ratio between the number of correct predictions and the total number of predictions.

$$
A c c u r a c y = \frac {\# C o r r e c t}{\# P r e d i c t i o n s} \tag {9}
$$

3) Confusion Matrix: A confusion matrix (or confusion table) is a table that is often used to describe the performance of a prediction model on a set of test data for which the true values are known. In other words, it is a matrix between Real Classes and Predicted Classes.

Confusion Matrix has following terms:

True Positive (TP): Means no of positive cases which are predicted positive.

False Positive (FP): Means no of negative cases which are predicted positive.   
True Negative (TN): Means no of negative cases which are predicted negative.   
False Negative (FN): Means no of positive cases which are predicted negative.

4) Precision & Recall (PR): Precision is a measure of classifier’s exactness. Whereas, Recall is a measure of classifier’s exactness.

In other words, Precision refers to how many selected items are relevant? Whereas, Recall refers to how many relevant items are selected?

$$
P r e c i s i o n = \frac {T P}{T P + F P} \tag {10}
$$

$$
\text {R e c a l l} = \frac {T P}{T P + F N} \tag {11}
$$

5) F-scores (F1): F1 score is simply the harmonic mean of precision and recall values.

$$
F 1 = 2 * \frac {\text {P r e c i s i o n} * \text {R e c a l l}}{\text {P r e c i s i o n} + \text {R e c a l l}} \tag {12}
$$

6) Area Under Curve (AUC): Also known as ”Area Under the Receiver Operating Characteristic curve (AUROC).” It is used to observe the usefulness of a model. AUROC curve is plotted between the true positive rate and the false positive rate at different threshold values.

# 5.2. Experiment I & II: Result & Analysis

The first & second experiment involves the use of Multilayer feed-forward networks to model Credit Card Fraud data and then to use the proposed deep auto-encoder to classify the fraudulent patterns with greater accuracy.

To perform this, TensorFlow software library is used. It is an open-source library used for data-flow programming across a range of tasks.

Test AUC Score: $9 5 . 3 3 \%$


[ImageDescription]
- source: images/2a37994bd2faa07532e7fc22d321023e1adabfa98470bad6dbc50dec4fb1b639.jpg
- alt: (no-alt)
- description: Image found in markdown. Detailed vision caption is unavailable in this runtime.
  
Figure 1. Auto Encoder confusion matrix


[ImageDescription]
- source: images/6e03b94223aaa6f27cb2c6cc096e21c0418115028cdcb84af84ca94274c63f30.jpg
- alt: (no-alt)
- description: Image found in markdown. Detailed vision caption is unavailable in this runtime.
  
Figure 2. Auto Encoder AUROC Curve


[ImageDescription]
- source: images/a0fa09086bf3dc30dcf1a1ed96c782769cc8102cc752e4a03914f9c08c66874d.jpg
- alt: (no-alt)
- description: Image found in markdown. Detailed vision caption is unavailable in this runtime.
  
Figure 3. Show distribution of all MSE


[ImageDescription]
- source: images/2c02a4548d00b1bde5f5d8715987256a9a7eece14e69f438ab5c91a59372a8ba.jpg
- alt: (no-alt)
- description: Image found in markdown. Detailed vision caption is unavailable in this runtime.
  
Figure 4. Display only fraud cases

# 5.3. Experiment III: Result & Analysis

The third experiment involves Nature Inspired Bat Algorithm for optimizing the FDS implemented using the first and the second experiment. The optimization is done using binary bat algorithm in the feature selection stage. It is found that this optimization can mitigate the training cost and the complexity of the entire system. Also it enhances the test AUC score significantly.

Test AUC Score: $9 6 . 2 1 \%$


[ImageDescription]
- source: images/3a09172fca95737af9f55ed8fc1d9b333cf22cb9472724012be4232ed16a2ef7.jpg
- alt: (no-alt)
- description: Image found in markdown. Detailed vision caption is unavailable in this runtime.
  
Figure 5. Auto Encoder confusion matrix


[ImageDescription]
- source: images/73312c453dfb87c1c694b2ddb9c34cc5bfbd990af9c1781bd75c539ed2c48579.jpg
- alt: (no-alt)
- description: Image found in markdown. Detailed vision caption is unavailable in this runtime.
  
Figure 6. Auto Encoder AUROC Curve

# 5.4. Experiment IV: Result & Analysis

The forth and final experiment involves Comparing different Scikit-learn methods/algorithms to classify fraudulent patterns. These comparisons are done using different performance measures e.g.: AUC scores, confusion matrices and precision-recall curves.

# 5.4.1. FDS using different Scikit learn methods with Under-sampling. This experimentation uses Undersampling to handle class imbalance problem.

Under-sampling intends to balance class distribution by randomly eliminating majority class observations. This is practised until the majority and minority class instances are balanced out.


[ImageDescription]
- source: images/1f0663fa831369493c6a15d46875885824dd4dcc2636b5ce6635c3e7b0ecd960.jpg
- alt: (no-alt)
- description: Image found in markdown. Detailed vision caption is unavailable in this runtime.
  
Figure 7. AUCROC Curve with Under-sampling

We can see that XGB performs better than any other Scikit learn algorithms but the proposed FDS and the optimization outperforms it.

# 5.4.2. FDS using different Scikit learn methods with Over-sampling. This experimentation uses Synthetic Minority Over-sampling Technique (SMOTE) to handle class imbalance problem.

SMOTE is used to avoid over-fitting which occurs when exact replicas of minority class instances are added to the main data-set.

A subset of data is taken from the minority class distribution as an example and then new, synthetic similar instances are created. These synthetic instances are then added to the original data-set. The new data-set is used as a sample to train the classification models.


[ImageDescription]
- source: images/fc26e2d5ab9e597311551517f1a04b1b61e4535811c8c03de7d054fcc8722527.jpg
- alt: (no-alt)
- description: Image found in markdown. Detailed vision caption is unavailable in this runtime.
  
Figure 8. AUCROC Curve with Over-sampling

In this case Random Forest is the brightest performer, in fact it turns out to be almost an ideal classifier. But it might be the case that after applying SMOTE, the model gets over-fitted that means its credibility is not guaranteed.

5.4.3. FDS using different Scikit learn methods with Stratified 3-fold sampling. This experimentation uses different Scikit learn method to implement FDS on the given data-set. Also in this experiment Stratified 3-fold sampling is used to demonstrate the different performance measures of these algorithms at three different folds.

Below is the best fold demonstration for different Scikit learn algorithms:


[ImageDescription]
- source: images/715f7eb6c202a20edbe7f5d298eae36dd1dc62caa453c840949257da37a20a50.jpg
- alt: (no-alt)
- description: Image found in markdown. Detailed vision caption is unavailable in this runtime.
  
• Random Forest (RF):   
Figure 9. RF Confusion Matrix


[ImageDescription]
- source: images/d19a5779ab2ed6301b926c00b9506c08f7c238055a306a46e4fb9ef482855eee.jpg
- alt: (no-alt)
- description: Image found in markdown. Detailed vision caption is unavailable in this runtime.
  
Figure 13. DT with entropy Classification Report

Figure 10. RF Classification Report

# • Linear Regression (LR):

<table><tr><td colspan="4">Coefficient:</td></tr><tr><td>[-5.13666891e-09</td><td>-1.94003929e-03</td><td>2.85955120e-03</td><td>-5.12566968e-03</td></tr><tr><td>3.82127084e-03</td><td>-2.34242973e-03</td><td>-1.70713138e-03</td><td>-6.92689998e-03</td></tr><tr><td>9.84500022e-04</td><td>-3.61701850e-03</td><td>-8.06975037e-03</td><td>6.22972168e-03</td></tr><tr><td>-1.08467353e-02</td><td>-1.87312931e-04</td><td>-1.31613033e-02</td><td>-2.09226924e-04</td></tr><tr><td>-9.36694484e-03</td><td>-1.61326157e-02</td><td>-5.64633721e-03</td><td>1.94153822e-03</td></tr><tr><td>2.57593383e-04</td><td>2.08422229e-03</td><td>1.77236969e-04</td><td>1.01074257e-04</td></tr><tr><td>-5.02834707e-04</td><td>3.00092273e-04</td><td>3.43306751e-04</td><td>1.44082788e-03</td></tr><tr><td>1.17774065e-03</td><td>7.15060808e-06]</td><td></td><td></td></tr><tr><td colspan="4">Intercept:</td></tr><tr><td colspan="4">0.0015802378973</td></tr><tr><td colspan="4">Accuracy = 0.525421124788</td></tr></table>

Figure 11. LR Classification Report

# • Logistic Regression (LOR):

<table><tr><td colspan="2">Coefficient: 
[[-7.07873101e-05 3.10375906e-01 -4.39220249e-01 -8.03621195e-01 
1.02051153e-01 -1.18303954e-02 -5.80886115e-02 2.97149895e-01 
-3.13064087e-01 -3.61117841e-01 -2.02536358e-01 -2.81515103e-01 
2.63285979e-02 -2.93581993e-01 -6.54243726e-01 -4.33720511e-01 
-2.92949221e-01 -4.72907602e-01 1.94575437e-02 2.64922477e-02 
1.05852994e-01 2.52719436e-01 3.49222013e-01 1.08728798e-01 
-3.89816931e-02 -3.16207443e-01 5.00063890e-02 -9.07856182e-02 
2.71164375e-02 -5.99714346e-03]]</td></tr><tr><td>Intercept: 
[-1.59521329]</td><td></td></tr><tr><td colspan="2">Accuracy = 0.999120460099</td></tr></table>

Figure 12. LOR Classification Report

# Decision Tree Classifier using entropy criterion (DT):

<table><tr><td colspan="5">---Classification Report---</td></tr><tr><td></td><td>precision</td><td>recall</td><td>f1-score</td><td>support</td></tr><tr><td>0</td><td>1.00</td><td>1.00</td><td>1.00</td><td>94772</td></tr><tr><td>1</td><td>0.72</td><td>0.78</td><td>0.75</td><td>164</td></tr><tr><td>avg / total</td><td>1.00</td><td>1.00</td><td>1.00</td><td>94936</td></tr><tr><td colspan="5">Accuracy = 0.999104659981</td></tr></table>


[ImageDescription]
- source: images/d218a06abd0e497c47f7669c77729034e0a28dbfb5952d4e79be00acc1dea6b0.jpg
- alt: (no-alt)
- description: Image found in markdown. Detailed vision caption is unavailable in this runtime.
  
Figure 14. DT with entropy Confusion Matrix


[ImageDescription]
- source: images/493bd94128e391c96f1083a4b81dadcfce23e0acf1443792f7d6fdb53b622240.jpg
- alt: (no-alt)
- description: Image found in markdown. Detailed vision caption is unavailable in this runtime.
  
• Decision Tree Classifier using gini criterion (DT):   
Figure 15. DT with gini Confusion Matrix


[ImageDescription]
- source: images/d5709ced92b68b2d18bf8728e4453c2a457d73981c5304d873fd9fb0e1e1eeba.jpg
- alt: (no-alt)
- description: Image found in markdown. Detailed vision caption is unavailable in this runtime.
  
Figure 16. DT with gini Classification Report


[ImageDescription]
- source: images/1d629e237a7b5e91b21c38a3e8da3e70d744c3698772397056441e585af4887e.jpg
- alt: (no-alt)
- description: Image found in markdown. Detailed vision caption is unavailable in this runtime.
  
• Gradient Boosting Machine (GBM):   
Figure 17. GBM Confusion Matrix


[ImageDescription]
- source: images/e5f01e01f74797b331343fceaa4d329574b2b2e828539ffdb7957cf60fb451dc.jpg
- alt: (no-alt)
- description: Image found in markdown. Detailed vision caption is unavailable in this runtime.
  
Figure 18. GBM Classification Report

# • XGBoost Classifier (XGB):

Figure 19. XGBoost Classification Report   

<table><tr><td colspan="5">---Classification Report---</td></tr><tr><td></td><td>precision</td><td>recall</td><td>f1-score</td><td>support</td></tr><tr><td>0</td><td>0.98</td><td>0.99</td><td>0.99</td><td>94771</td></tr><tr><td>1</td><td>0.99</td><td>0.98</td><td>0.99</td><td>94771</td></tr><tr><td>avg / total</td><td>0.99</td><td>0.99</td><td>0.99</td><td>189542</td></tr><tr><td colspan="5">Accuracy = 0.987680830634</td></tr></table>

# • ADABoost Classifier (ADAB):


[ImageDescription]
- source: images/0ae592f697e57cb6a70d6416bae19fa8bbff47a5780dd225bfd2334250f21e19.jpg
- alt: (no-alt)
- description: Image found in markdown. Detailed vision caption is unavailable in this runtime.
  
Figure 20. ADABoost Confusion Matrix

Figure 21. ADABoost Classification Report   

<table><tr><td colspan="5">---Classification Report---</td></tr><tr><td></td><td>precision</td><td>recall</td><td>f1-score</td><td>support</td></tr><tr><td>0</td><td>1.00</td><td>1.00</td><td>1.00</td><td>94772</td></tr><tr><td>1</td><td>0.92</td><td>0.82</td><td>0.87</td><td>164</td></tr><tr><td>avg / total</td><td>1.00</td><td>1.00</td><td>1.00</td><td>94936</td></tr><tr><td colspan="5">Accuracy = 0.999568130109</td></tr></table>

In this case again Random Forest is the brightest performer, as we can see its false positive rate. But looking at the recall value, its below our expectations. i.e we can again say that out proposed FDS model with the optimization outperforms this case as well.

# 6. Conclusion

The proposed methodology has made a useful contribution through the unsupervised Fraud detection method. In fact this can be a step towards more automation in the Fraud detection systems. This is important because it reduces human intervention in the whole process and therefore mitigates time and cost. Also it obtains a high fraud coverage with low false alarm rate.

The supervised Fraud detection approaches have been shown to be an effective classifier. But the proposed method is useful even when labeled data is not available.

The proposed Fraud detection approach can be a very effective method especially as the two fields (large amount of data and Deep learning) are rapidly evolving but more importantly because Deep learning is posited to be the most promising Machine learning method for Big Data analytic.

# References

[1] A. Dal Pozzolo, G. Boracchi, O. Caelen, C. Alippi, and G. Bontempi, “Credit card fraud detection: a realistic modeling and a novel learning strategy,” IEEE Transactions on Neural Networks and Learning Systems, 2017.   
[2] R. J. Bolton and D. J. Hand, “Statistical fraud detection: A review,” Statistical science, pp. 235–249, 2002.   
[3] E. Hung and D. W. Cheung, “Parallel algorithm for mining outliers in large database,” in Proc. 9th International Database Conference (IDC’99), Hong Kong, 1999.   
[4] Y. Kou, C.-T. Lu, S. Sirwongwattana, and Y.-P. Huang, “Survey of fraud detection techniques,” in Networking, sensing and control, 2004 IEEE international conference on, vol. 2. IEEE, 2004, pp. 749–754.   
[5] P.-N. Tan, M. Steinbach, and V. Kumar, “Introduction to data mining. 1st,” 2005.   
[6] A. O. Adewumi and A. A. Akinyelu, “A survey of machine-learning and nature-inspired based credit card fraud detection techniques,” International Journal of System Assurance Engineering and Management, pp. 1–17, 2016.   
[7] A. Bergholz, J. H. Chang, G. Paass, F. Reichartz, and S. Strobel, “Improved phishing detection using model-based features.” in CEAS, 2008.   
[8] T. O. Ayodele, “Types of machine learning algorithms,” in New advances in machine learning. InTech, 2010.   
[9] A. Srivastava, A. Kundu, S. Sural, and A. Majumdar, “Credit card fraud detection using hidden markov model,” IEEE Transactions on dependable and secure computing, vol. 5, no. 1, pp. 37–48, 2008.   
[10] M. Z. Khan, J. D. Pathan, and A. H. E. Ahmed, “Credit card fraud detection system using hidden markov model and k-clustering,” International Journal of Advanced Research in Computer and Communication Engineering, vol. 3, no. 2, pp. 5458–5461, 2014.   
[11] Y. G. S¸ahin and E. Duman, “Detecting credit card fraud by decision trees and support vector machines,” Newswood Limited, 2011.   
[12] K. Seeja and M. Zareapoor, “Fraudminer: a novel credit card fraud detection model based on frequent itemset mining,” The Scientific World Journal, vol. 2014, 2014.   
[13] M. Zareapoor and P. Shamsolmoali, “Application of credit card fraud detection: Based on bagging ensemble classifier,” Procedia Computer Science, vol. 48, pp. 679–685, 2015.   
[14] F. Fadaei Noghani and M. Moattar, “Ensemble classification and extended feature selection for credit card fraud detection,” Journal of AI and Data Mining, vol. 5, no. 2, pp. 235–243, 2017.

[15] G. Rozenberg, T. Bck, and J. N. Kok, Handbook of natural computing. Springer Publishing Company, Incorporated, 2011.   
[16] M. Mitchell, An introduction to genetic algorithms. MIT press, 1998.   
[17] R. D. Patel and D. K. Singh, “Credit card fraud detection & prevention of fraud using genetic algorithm,” International Journal of Soft Computing and Engineering, vol. 2, no. 6, 2013.   
[18] A. H. Awlla, “A hybrid simulated annealing and back-propagation algorithm for feed-forward neural network to detect credit card fraud,” UHD Journal of Science and Technology, vol. 1, no. 2, pp. 31–36, 2017.   
[19] X.-S. Yang, Nature-inspired optimization algorithms. Elsevier, 2014.