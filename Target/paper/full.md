# KANNADA-MNIST: A NEW HANDWRITTEN DIGITS DATASETFOR THE KANNADA LANGUAGE

Vinay Uday Prabhu

dig.mnist@gmail.com

August 6, 2019

# ABSTRACT

In this paper, we disseminate a new handwritten digits-dataset, termed Kannada-MNIST, for the Kannada script, that can potentially serve as a direct drop-in replacement for the original MNIST dataset[1]. In addition to this dataset, we disseminate an additional real world handwritten dataset (with $1 0 k$ images), which we term as the Dig-MNIST1 dataset that can serve as an out-of-domain test dataset. We also duly open source all the code as well as the raw scanned images along with the scanner settings so that researchers who want to try out different signal processing pipelines can perform end-to-end comparisons. We provide high level morphological comparisons with the MNIST dataset and provide baselines accuracies for the dataset disseminated. The initial baselines2 obtained using an oft-used CNN architecture $( 9 6 . 8 \%$ for the main test-set and $7 6 . 1 \%$ for the Dig-MNIST test-set) indicate that these datasets do provide a sterner challenge with regards to generalizability than MNIST or the KMNIST datasets. We also hope this dissemination will spur the creation of similar datasets for all the languages that use different symbols for the numeral digits.

# 1 Introduction

Kannada is the official and administrative language of the state of Karnataka in India with nearly 60 million speakers worldwide [3]. Also, as per articles 344(1) and 351 of the Indian Constitution, Kannada holds the status of being one of the 22 scheduled languages of India [4]. The language is written using the official Kannada script, which is an abugida of the Brahmic family and traces its origins to the Kadamba script (325-550 AD).

Distinct glyphs are used to represent the numerals 0-9 in the language that appear distinct from the modern Hindu-Arabic numerals in vogue in much of the world today. Unlike some of the other archaic numeral-systems, these numerals are very much used in day-to-day affairs in Karnataka, as in evinced by the prevalence of these glyphs on license-plates of vehicles captured in fig 1.

Fig 2 captures the evolution of the numerals through the ages. Modern Kannada scholars [5] posit that the emergence of these numeral-glyphs can be traced to the Gudnapur inscriptions [6], dating back to the $\mathbf { \bar { \boldsymbol { 6 } } } ^ { \dot { t } h }$ century AD when the Kadamba rulers held sway over the region[7]. The Kannada digits for 0-9 are shown in Fig 1 (Unicode: 0CE6 through to 0CEF) [9] .

Fig 4 captures the MNIST-ized renderings of the variations of the glyphs across the following modern fonts: Kedage, Malige-i, Malige-n, Malige-b, Kedage-n, Malige-t, Kedage-t, Kedage-i, Lohit-Kannada, Sampige and Hubballi-Regular.

# 1.1 The curious case of glyphs for 3,7 and 6

In this section, we focus on some idiosyncrasies with regards to the shapes of the glyphs used to represent numerals in Kannada. Three interesting observations emerge from Fig3 and Fig4.

The first observation is that the glyph for 0 is the same as in the Hindu-Arabic system. Secondly, the shapes of the digits for 3 and 7 in Kannada look rather similar to the glyph for 2 in the modern Hindu-Arabic numeral system (Fig 5).

![](images/caa48fd4af82337fccb49195f51a890900482a7bf5c6f5f6835abd8bd360126a.jpg)  
Figure 1: Usage of the Kannada numerals on vehicular license plates

These will be leveraged during our dataset curation procedure as a sanity check for scanned and segmented digits using a pre-trained MNIST-digits classifier.

The third observation is related to the peculiar intra-class variation of the representation (refer to fig 4 and fig 2) for 6 across different fonts and different eras. The modern day deformations are represented in Fig 6, where we observe the deviation from the puritanical textbook representation of the symbol (as seen in the unicode-derived image on the extreme left) and the more colloquial usage which looks like a mirror image of 3 in the Hindu-Arabic system. As will be seen in the upcoming section, many of the volunteers who helped curate the dataset used one or both of these glyphs, resulting in high intra-class variation.

# 1.2 Related work

There have been some nascent attempts made towards Kannada handwritten digit classification, albeit at a smaller scale. In [10], the authors used the chain code histogram idea to achieve $98 \%$ accuracy on a dataset of 2300 digit-images. In [11], the authors used a nearest neighbor classifier to achieve $91 \%$ accuracy of 250 test numerals. Support Vector Machines (SVMs) were used to achieve $98 \%$ accuracy on a small dataset of $5 0 0 0 4 0 \times 4 0$ numeral-images in [12]. The largest dataset currently used in academic literature that contains Kannada characters is the Chars74k dataset [13] that contains 657 characters of the Kannada script collected using a tablet PC, albeit with a mere 25 samples per-number. In [14], the authors harnessed standard augmentation techniques to create an augmented dataset of 18000 digit images harnessing the Chars74k dataset and trained Convolutional Neural Networks (CNNs) and Deep Belief Networks (DBNs) to obtain $9 8 \%$ test accuracy. Earlier this year, we proposed a Seed-Augment-Train/Transfer (SAT) framework that contains a synthetic seed image dataset generation procedure for languages with different numeral systems using freely available open font file datasets (Lohit to be more specific). This seed dataset of images was then augmented to create a purely synthetic training dataset, using which we trained a deep neural network and tested on held-out real world small-sized handwritten digits dataset spanning five Indic scripts, Kannada, Tamil, Gujarati, Malayalam, and Devanagari, containing 1280 digits each.

Through this paper, we hope to address this paucity of an MNIST-sized dataset for the Kannada language.

# 1.3 Main contributions of the paper

The main contributions are:

1. Contributing a real world handwritten Kannada-MNIST dataset that was collected in Bangalore, India, that can potentially serve as a direct drop-in replacement for the original MNIST dataset[1].   
2. Contributing an additional 10k real world handwritten Dig-MNIST dataset that was collected in Redwood City, CA, that can serve as an out-of-domain test dataset.   
3. Open sourcing all the code required to generate such datasets for other languages.

![](images/4272793f28ecb5ad05e2fba83bfab5d78952e42b2fa30e1e77e69d8d77e21d78.jpg)  
Figure 3: The character code tables for Kannada-MNIST from the Unicode Standard, Version 12.1

Figure 2: Evolution of the Kannada numerals through the ages ([8])

The Unicode Standard 12.1   

<table><tr><td>0CE6</td><td>0</td><td colspan="3">KANNADA DIGIT ZERO</td><td>0CEB</td><td>8</td><td colspan="3">KANNADA DIGIT FIVE</td></tr><tr><td>0CE7</td><td>0</td><td colspan="3">KANNADA DIGIT ONE</td><td>0CEC</td><td>2</td><td colspan="3">KANNADA DIGIT SIX</td></tr><tr><td>0CE8</td><td>9</td><td colspan="3">KANNADA DIGIT TWO</td><td>0CED</td><td>2</td><td colspan="3">KANNADA DIGIT SEVEN</td></tr><tr><td>0CE9</td><td>2</td><td colspan="3">KANNADA DIGIT THREE</td><td>0CEE</td><td>6</td><td colspan="3">KANNADA DIGIT EIGHT</td></tr><tr><td>0CEA</td><td>8</td><td colspan="3">KANNADA DIGIT FOUR</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>○</td><td>○</td><td>○</td><td>○</td><td>○</td><td>○</td><td>○</td><td>○</td><td>○</td><td>○</td></tr></table>

4. Open-sourcing the raw scanned images along with the scanner settings so that researchers who want to try out different signal processing pipelines can perform end-to-end comparisons.   
5. Performing high level morphological comparisons with the MNIST dataset and providing baselines accuracies for the dataset disseminated.   
6. Open sourcing the code-templates for generating synthetic seed images for various modern Kannada fonts.

The rest of the paper is organized as follows: Section-2 covers the dataset preparation process, Section-3 details the comparisons vis-a-vis the standard MNIST dataset. In Section-4, we present the classification baseline results obtained using an off-the-shelf CNN, and Section-5 concludes the paper.

# 2 Dataset Creation

In order to avoid the kind of uncertainties, folklore and trivia surrounding MNIST (as evinced in [15]), we have decided to detail and open source all aspects of the data collection process. Further, we have also decided to open-source the

![](images/0eeae459cd5941105b797569207a2b4068c6cd3df9a2ec0989e63b4ecfacef77.jpg)

![](images/2cfd260a8c506391ecefe3d52e7d801f3b48d506dd5affa5da7eb2669e33f960.jpg)  
Figure 4: MNIST-ized renderings of the 0-9 Kannada numerals in 11 modern fonts   
Unicode-0CE9 (Lohit-font)

![](images/88b3706dba2a85dff695859f05d72f93e81b17c3c36f67d0439f014c94f464c2.jpg)  
Mean (class-3)

![](images/197da7c25204f17a61299cbc9c83dd29600734b4e8789c9b8583cc17f3c36563.jpg)  
Unicode-0CED (Lohit-font)

![](images/aa0835b814b17675675cc30b8d248c693af372adc5fd9be14ef11b443cd2aa28.jpg)  
Mean (class-7)   
Figure 5: The similarity between the glyphs for 3 and 7 in Kannada

raw scan images to facilitate end-to-end experimentation with disparate signal processing pipelines. In this section, we will cover the details of creating the following two datasets:

1. The main Kannada-MNIST dataset that consists of a training set of $6 0 0 0 0 \ : 2 8 \times 2 8$ gray-scale sample images and a test set of 10000 sample images uniformly distributed across the 10 classes. This dataset is based off of the efforts of 65 volunteers from Bangalore, India, who are native speakers and users of the Kannada language and the script. This was curated to serve as a direct one-to-one drop-in replacement for the original MNIST dataset (akin to Fashion-MNIST [16] and K-MNIST [17] datasets).   
2. The Dig-MNIST dataset that consists of $1 0 2 4 0 2 8 \times 2 8 \time 1 0 0 0 0$ gray-scale images that was curated with the purpose of providing a more challenging test dataset that was curated in Redwood City, CA, with the help of volunteers

![](images/540bd581ec7617368774353500c979cb026d58a96c426e2a172900af14aa0eb7.jpg)  
Unicode-0CEC (Lohit-font)

![](images/eabb1df0f76f17318570e0ee597084cae198959bbf580df1134f0d4940c50134.jpg)  
Volunteer-1's instance

![](images/66ec543a165f0bbcc9c687eaa1f0b1bc992dd927e81a6b75b642d6cedcbb40c5.jpg)  
Mean over the dataset   
Figure 6: Deformations in the numeral glyphs for 6

many of whom were encountering the Kannada script for the first time and had fair difficulty in replicating the shape of the glyphs. This test dataset, we hope will facilitate domain adaptation experiments.

# 2.1 Main dataset

Fig 7 presents the work-flow followed to curate the main Kannada-MNIST dataset. The whole process was split into four phases: Data-gathering, pre-processing and slicing, Sanity-check, Train-test split. In the following subsections, we will cover each of these in detail.

# 2.1.1 Data-gathering

65 volunteers were recruited in Bangalore, India, who were native speakers of the language as well as day-to-day users of the numeral script. Each volunteer filled out an A3 sheet containing a $3 2 \times 4 0$ grid. This yielded filled-out A3 sheets containing 128 instances of each number which we posit is large enough to capture most of the natural intra-volunteer variations of the glyph shapes. All of the sheets thus collected were scanned at 600 dots-per-inch resolution using the Konica Accurio-Press-C6085 scanner that yielded $6 5 ~ 4 9 6 3 \times 3 5 0 9$ png images.

# 2.1.2 Pre-processing and slicing

In this sub-phase, each of the $4 9 6 3 \times 3 5 0 9$ sized scanned $3 2 \times 4 0$ grid png images were passed through two preprocessing stages3 as detailed in [18]. This approach was originally used as an extraction framework to eke out the digits for a Sudoku-solver. The first pre-processing stage entails:

1. Applying a Gaussian-blur filter of kernel-size $9 \times 9$   
2. Performing adaptive thresholding using 11 nearest neighbour pixels   
3. Applying a bitwise-NOT operator to perform colour inversion to ensure that the target gridlines have non-zero pixel values

The second pre-processing phase entailed two operations. The first was to estimate the corners of the largest polygon that was then harnessed to crop and warp the $3 2 \times 4 0$ grid-image. The intermediate images after these phases are as shown in Fig 7.

The cropped-and-warped image thus obtained was then segmented into 1280 slices to yield the constituent individual digit images which were then MNIST-ized4. For this, we followed the procedure in [19] that entails pixel-thresholding, row-column padding and finally inflicting a Best-shift transformation to drag the current center-of-mass of the digit image to the center of the target MNIST-ized $2 8 \times 2 8$ image. At the end of this phase, we had a $1 2 8 \times 1 0 \times 2 8 \times 2 8$ image tensor per scanned image. The class-label associated with each image was obtained using the row index of the image in the $3 2 \times 4 0$ grid.

# 2.1.3 Sanity-check

One natural question that emerged during our new dataset curation was how to ensure the MNIST-compatibility of the same? In order to assuage these concerns, we decided to literally use a CNN pre-trained on the original MNIST digits

and perform inference on the newly created digit images targeting classes in Kannada that looked similar to the MNIST digits.This formed an integral part of a series of sanity checks we performed that are as shown in Fig 7 and listed below:

1. Firstly, we perform class-wise checks by looking at the histogram of counts of the labels as well as eye-balling out the class-wise mean images of the $1 2 8 \times 1 0$ digits array and visually verifying that they indeed look like their archetypal glyphs.   
2. Secondly, using the observations made in Section 1.1, we perform 3 sets of classifications on the MNIST-ized digit images. We use a high accuracy $( 9 9 . 4 \%$ test-set accuracy) $\mathrm { C N N } ^ { 5 }$ pre-trained on MNIST digits to classify the images belonging to class zero(as the glyph for zero is the same), three and seven (as the glyphs look very similar to 2 in MNIST). This produces a triple of accuracies which are used to ascertain the quality and MNIST-likeness of the images produced by the parsing procedure we’ve deployed. With regard to Fig 7, $9 5 \%$ of the 128 zero-images were classified by the MNIST-CNN as zero, $9 6 \%$ of the 128 three-class images were classified by the MNIST-CNN as class-2 and $8 3 \%$ of the 128 seven-class images were classified by the MNIST-CNN as class-2.

We have duly shared the implementation of the step-wise procedure described above as a colab notebook accessed here: https://github.com/vinayprabhu/Kannada_MNIST/blob/master/colab_ notebooks/0)Scan_parse_example_main.ipynb

# 2.1.4 Train-test split: Worst cohort selection

Using the curation procedure described above, we were able to collect $6 5 \times 1 2 8 0 = 8 3 2 0 0$ images. In order to ensure that our dataset would serve as drop-in replacement to the MNIST dataset, we had to select 60000 images for the final training set and 10000 images for the final test dataset. Upon random selection, we were able to hit similar levels of accuracy $( > 9 9 \% )$ that is achieved for the MNIST dataset. This could very well be attributed to the fact that random sampling based train-test data segmentation essentially allows the CNN to cover the span of the possible glyph-modes used by a user to represent a numeral whereas the real generalization challenge lies in being able to model the possible deviations that the glyphs shape might take across unseen users. Hence, we first sorted the users according to their difficulty scores. The difficulty score for a user was computed by taking the mean of the elements of the proxy-score-vector, which represents the probabilities that the user’s 0, 3 and 7 representations in Kannada would be classified as 0, 2 and 2 by the MNIST-CNN classifier as explained in Section 2.1.3. We then picked the top/worst 8 users into a test cohort and sampled 10000 digits to form the final test dataset. We then picked the next 47 users and sampled 60000 digits to form the final train dataset. This implies that any test-accuracy achieved by a machine learning classifier model will actually map to the ability of the classifier to predict the digit-classes from images emanating from users hitherto unseen during the training phase. The triple of 0-3-7 class accuracy-vectors of the train and test datasets thus formed were [0.943, 0.962, 0.9575] and [0.825, 0.87, 0.743] respectively.

We have shared the implementation of this procedure through a colab notebook shared here: https: //github.com/vinayprabhu/Kannada_MNIST/blob/master/colab_notebooks/1b)_Main_ dataset_tensor_generation_worst_cohort.ipynb

The class-wise mean images of the train set (top-row) , the test set (second-row) and the difference between the train and test classwise-means is shown in Fig 8

# 2.2 10k Dig-MNIST dataset

As stated above, we also disseminate an additional more challenging 10k Dig-MNIST dataset in this paper that was collected using volunteers in Redwood City, many of whom were, in fact, seeing the numeral glyphs for the first time and trying their best to reproduce the shapes. This sampling-bias, combined with the fact that we used a completely different writing sheet dimension and scanner settings, resulted in a dataset that would turn out to be far more challenging than the easy test dataset curated in the above sub-section. The rest of this sub-section details the specifics of the procedure and the companion colab notebook can be obtained at: https://github.com/vinayprabhu/Kannada_MNIST/ blob/master/colab_notebooks/2)_Kannada_MNIST_10k_RWC.ipynb.

# 2.3 Dataset curation

8 volunteers aged 20 to 40 were recruited to generate a $3 2 \times 4 0$ grid of Kannada numerals (akin to 2.1), all written with a black ink Z-Grip Series | Zebra Pen on a commercial Mead Cambridge Quad Writing Pad, 8-1/2" x 11", Quad Ruled,

White, 80 Sheets/Pad book. We then scan the sheet(s) using a Dell - S3845cdn scanner (See Fig 9)with the following settings:

• Output color: Grayscale   
• Original type: Text   
• Lighten/Darken: Darken $^ { + 3 }$   
• Size: Auto-detect

The reduced size of the sheets used for writing the digits (US-letter vis-a-vis A3) resulted in smaller scan (.tif) images that were all approximately $1 6 0 0 \times 2 0 0 0$ . The Darken $^ { + 3 }$ scanner option resulted in the grid lines being visible enough that it allowed us to use the same signal processing pipeline detailed in Section 2.1 built on the sudoku-digit extraction idea. Fig 10 captures the user-wise class-wise mean-images for the dataset thus curated.

# 3 Comparisons with the MNIST dataset

In the section, we provide some qualitative and quantitative comparisons between the MNIST and the Kannada-MNIST datasets6

# 3.1 Morphological comparisons

In Fig 11, we provide a comparison of the mean pixel-wise intensities between the MNIST and the Kannada-MNIST datasets. As seen, the Kannada-MNIST dataset is much less peaky with a maximal mean pixel-intensity of $\sim 0 . 3$ as compared to the MNIST dataset, that has a few pixel indices with mean pixel-intensities of $\sim 0 . 6$ . In Fig 12, we used the Morpho-MNIST framework [20] to generate the statistics of morphological traits such as length, thickness, slant, width and height of the handwritten digits for the two datasets. As seen, the bi-modality of length as well as width is less pronounced for the Kannada digits. The slant-to-width joint-scatter-plots were visibly different between the two datasets as well.

# 3.2 Dimensionality reduction comparisons

To begin with, we used the Uniform Manifold Approximation and Projection (UMAP) [21] technique to visualize a two-dimensional projection of the two datasets. As seen in Fig 14, the two sub-plots paint a very different picture of the lower dimensional representations for the two datasets. We also performed dimensionality reduction analysis using PCA to understand the variation of explained variance across the PCA components. As seen in Fig 13,the top-50 PCA components explain $8 3 \%$ of the total variance for the MNIST dataset and only $6 3 \%$ for Kannada-MNIST.

# 4 Classification results

In this section, we present the classification results obtained by training an off-the-shelf $\mathrm { C N N } ^ { 7 }$ (See Fig 15) using Adadelta optimizer with learning-rate $^ { : = 1 }$ .0 and $\rho = 0 . 9 5$ . For the main dataset, with 60, 000 − 10, 000 train-test split, we achieved $9 7 . 1 3 \%$ top-1 accuracy. The classification report is as shown in Table 1 and the epoch-wise accuracy and loss plots are as shown in Fig 17. In Fig 16, we have the confusion matrix. This pre-trained CNN achieved $7 6 . 2 \%$ top-1 accuracy on the dig-10k dataset. In terms of precision, class-2 $( 6 3 . 9 \% )$ and class-6 $( 5 5 . 1 \% )$ were the most challenging. In terms of recall, classes 0,3 and 7 were all at the sub- $6 1 \%$ level. This showcases the fragile nature of the CNN’s ability to truly generalize across author-cohorts and provides for an interesting challenge to the machine learning community at large. Fig 18 and Table 2 provide the confusion matrix and the per-class classifcation report for the dig dataset.

Lastly, for the single-author 1280 digits dataset used in [22], we achieved $8 3 . 6 7 \%$ top-1 accuracy. Again, as seen in Table 3 the CNN did struggle to achieve good precision class-6 $( 6 0 \% )$ and good recall for classes -0 and 7 $( 6 0 - 6 3 \% )$ Figure 19 provides the confusion matrix for the same. Given the smaller size of the test dataset, we did dig in to visualize the images of the digits that the CNN classified for classes 0 (Fig 20), 7(Fig 21) and $8 ( \mathrm { F i g } 2 2 )$ . The title of each of the plots represents the predicted class by the CNN. As seen, for a human eye, most of the images do look like the glyphs. But, upon closer inspection, we observe the presence of rogue non-glyph pixels (akin to naturally occurring

Table 1: Classification report for the Kannada MNIST dataset   

<table><tr><td>class</td><td>precision</td><td>recall</td><td>f1-score</td><td>support</td></tr><tr><td>0</td><td>0.9702</td><td>0.9130</td><td>0.9408</td><td>1000</td></tr><tr><td>1</td><td>0.9239</td><td>0.9830</td><td>0.9525</td><td>1000</td></tr><tr><td>2</td><td>0.9960</td><td>0.9970</td><td>0.9965</td><td>1000</td></tr><tr><td>3</td><td>0.9627</td><td>0.9800</td><td>0.9713</td><td>1000</td></tr><tr><td>4</td><td>0.9538</td><td>0.9920</td><td>0.9725</td><td>1000</td></tr><tr><td>5</td><td>0.9828</td><td>0.9700</td><td>0.9763</td><td>1000</td></tr><tr><td>6</td><td>0.9466</td><td>0.9740</td><td>0.9601</td><td>1000</td></tr><tr><td>7</td><td>0.9967</td><td>0.9010</td><td>0.9464</td><td>1000</td></tr><tr><td>8</td><td>0.9822</td><td>0.9930</td><td>0.9876</td><td>1000</td></tr><tr><td>9</td><td>0.9771</td><td>0.9820</td><td>0.9796</td><td>1000</td></tr><tr><td>accuracy</td><td></td><td></td><td>0.9685</td><td>10000</td></tr><tr><td>macro_avg</td><td>0.9692</td><td>0.9685</td><td>0.9684</td><td>10000</td></tr><tr><td>weighted_avg</td><td>0.9692</td><td>0.9685</td><td>0.9684</td><td>10000</td></tr></table>

Table 2: Classification report for the dig dataset   

<table><tr><td>Class</td><td>precision</td><td>recall</td><td>f1-score</td><td>support</td></tr><tr><td>0</td><td>0.8360</td><td>0.6074</td><td>0.7036</td><td>1024</td></tr><tr><td>1</td><td>0.9013</td><td>0.7578</td><td>0.8233</td><td>1024</td></tr><tr><td>2</td><td>0.6385</td><td>0.9434</td><td>0.7615</td><td>1024</td></tr><tr><td>3</td><td>0.8787</td><td>0.6016</td><td>0.7142</td><td>1024</td></tr><tr><td>4</td><td>0.9191</td><td>0.7432</td><td>0.8218</td><td>1024</td></tr><tr><td>5</td><td>0.7441</td><td>0.9541</td><td>0.8361</td><td>1024</td></tr><tr><td>6</td><td>0.5511</td><td>0.7949</td><td>0.6509</td><td>1024</td></tr><tr><td>7</td><td>0.8918</td><td>0.5713</td><td>0.6964</td><td>1024</td></tr><tr><td>8</td><td>0.7732</td><td>0.7725</td><td>0.7728</td><td>1024</td></tr><tr><td>9</td><td>0.7936</td><td>0.8711</td><td>0.8305</td><td>1024</td></tr><tr><td>accuracy</td><td></td><td></td><td>0.7617</td><td>10240</td></tr><tr><td>macro_avg</td><td>0.7927</td><td>0.7617</td><td>0.7611</td><td>10240</td></tr><tr><td>weighted_avg</td><td>0.7927</td><td>0.7617</td><td>0.7611</td><td>10240</td></tr></table>

adversarial perturbations) and discontinuities in the strokes in many of the erroneously classified images, which we posit might well explain the misclassifications.

# 5 Conclusion and Future work

In this paper, we described in detail the creation of a new handwritten digits dataset for the Kannada language, which we term as Kannada-MNIST dataset. We have duly open sourced all aspects of the dataset creation including the raw scan images, the specific brand of paper used8, the exact scanner model used, the signal processing script used to slice and extract the individual digits and the CNN models used to obtain the baseline accuracies. We were able to attain $\sim 9 7 \%$ top-1 accuracy when we trained and tested on what we term as the main dataset with $6 0 0 0 0 \ : 2 8 \times 2 8$ gray-scale training images and 10000 test images. This is meant to be in a drop-in replacement for the standard MNIST dataset. We also achieved a top-1 accuracy of $\sim 7 7 \%$ when we trained on the 60000 main dataset and tested on 10240 $2 8 \times 2 8$ gray-scale test images from what we term as the Dig-MNIST dataset. The images in the Dig-MNSIT dataset are noisier with smudges and grid borders sneaking in during the grid-image segmentation phase.

We propose the following open challenges to the machine learning community at large.

1. Achieve MNIST-level accuracy by training on the Kannada-MNIST dataset and testing on the Dig-MNIST dataset without resorting to image pre-processing.

Table 3: Classification report for the 1280 digits dataset used in [22]   

<table><tr><td>Class</td><td>Precision</td><td>Recall</td><td>f1-score</td><td>Support</td></tr><tr><td>0</td><td>0.99</td><td>0.61</td><td>0.75</td><td>128</td></tr><tr><td>1</td><td>0.88</td><td>0.95</td><td>0.91</td><td>128</td></tr><tr><td>2</td><td>0.75</td><td>1.00</td><td>0.86</td><td>128</td></tr><tr><td>3</td><td>0.99</td><td>0.79</td><td>0.88</td><td>128</td></tr><tr><td>4</td><td>0.98</td><td>0.87</td><td>0.92</td><td>128</td></tr><tr><td>5</td><td>0.83</td><td>0.98</td><td>0.90</td><td>128</td></tr><tr><td>6</td><td>0.60</td><td>0.89</td><td>0.72</td><td>128</td></tr><tr><td>7</td><td>0.95</td><td>0.63</td><td>0.76</td><td>128</td></tr><tr><td>8</td><td>0.78</td><td>0.71</td><td>0.74</td><td>128</td></tr><tr><td>9</td><td>0.88</td><td>0.93</td><td>0.90</td><td>128</td></tr><tr><td>Accuracy</td><td></td><td></td><td>0.84</td><td>1280</td></tr><tr><td>macro_avg</td><td>0.86</td><td>0.84</td><td>0.84</td><td>1280</td></tr><tr><td>weighted_avg</td><td>0.86</td><td>0.84</td><td>0.84</td><td>1280</td></tr></table>

2. To characterize the nature of catastrophic forgetting when a CNN pre-trained on MNIST is retrained with Kannada-MNIST. This is particularly interesting given the observation that the typographical glyphs for 3 and 7 in Kannada-MNIST hold uncanny resemblance with the glyph for 2 in MNIST.   
3. Get a model trained on purely synthetic data generated9 using the fonts (as in [22]) and augmenting using frameworks such as [20] and [23] to achieve high accuracy of the Kannada-MNIST and Dig-MNIST datasets.   
4. Replicate the procedure described in the paper across different languages/scripts, especially the Indic scripts.   
5. With regards to the dig-MNIST dataset, we saw that some of the volunteers had transgressed the borders of the grid and hence some of the images either have only a partial slice of the glyph/stroke or have an appearance where it can be argued that they could potentially belong to either of two different classes. With regards to these images, it would be worthwhile to see if we can design a classifier that will allocate proportionate softmax masses to the candidate classes.   
6. The main reason behind us sharing the raw scan images was to foster research into auto-segmentation algorithms that will parse the individual digit images from the grid, which might in turn lead to higher quality of images in the upgraded versions of the dataset.

# Acknowledgement

To begin with, we’d like to acknowledge the contribution of all the volunteers who contributed towards this dataset. Specifically, we’d like to thank Kaushik BK, who was instrumental in helping manage the cohort of 65 volunteers who contributed to the main dataset in Bangalore, India. We’d also like to acknowledge the contributors of the Dig-MNIST dataset in Redwood City, including John Whaley, Nick Richardson, Joseph Gardi and Preethi Sheshadri. Last but not the least, we’d like to acknowledge the helpful advice shared by the authors of the K-MNIST paper, Tarin Clanuwat, Alex Lamb(Mila) and David Ha (Google Brain).

# References

[1] Yann LeCun, Corinna Cortes, and CJ Burges. Mnist handwritten digit database. AT&T Labs [Online]. Available: http://yann. lecun. com/exdb/mnist, 2:18, 2010.   
[2] Evelyn Richter. Student slang at IIT Madras: a linguistic field study. Master’s thesis, Technische Universitat Chemnitz, Str. der Nationen 62, 09111 Chemnitz, Germany, 2006.   
[3] Kannada. https://en.wikipedia.org/wiki/Kannada, 2019. [Online; accessed 16-Mar-2019].   
[4] Eighth schedule to the constitution of india. https://en.wikipedia.org/wiki/Eighth_Schedule_ to_the_Constitution_of_India, 2019. [Online; accessed 16-Mar-2019].   
[5] Special correspondent;. Scholar throws light on the origin, evolution of kannada numerals. The Hindu, Nov 2017.

[6] BR Gopal. Gudnapur inscription of kadamba ravivarma. Srikanthika: Dr S. Srikantha Sastri Felicitation Volume, pages 61–72, 1973.   
[7] G. S. Gai. In01046 no.22: Plate xxii gudnapur inscription of ravivarman, 1996.   
[8] M. G. Manjunath and G. K. Devarajaswamy. Kannada lipi vikasa. Technical report, Jagadhguru Sri Madhvacharya Trust, Sri Raghavendra Swami Matta, Mantralaya, 2004.   
[9] Unicode charts. https://unicode.org/charts/PDF/U0C80.pdf, 2019. [Online; accessed 16-Mar-2019].   
[10] Nabin Sharma, U Pal, and Fumitaka Kimura. Recognition of handwritten kannada numerals. In 9th International Conference on Information Technology (ICIT’06), pages 133–136. IEEE, 2006.   
[11] GG Rajput and Mallikarjun Hangarge. Recognition of isolated handwritten kannada numerals based on image fusion method. In International Conference on Pattern Recognition and Machine Intelligence, pages 153–160. Springer, 2007.   
[12] GG Rajput, Rajeswari Horakeri, and Sidramappa Chandrakant. Printed and handwritten mixed kannada numerals recognition using svm. International Journal on Computer Science and Engineering, 2(05):1622–1626, 2010.   
[13] T. E. de Campos, B. R. Babu, and M. Varma. Character recognition in natural images. In Proceedings of the International Conference on Computer Vision Theory and Applications, Lisbon, Portugal, February 2009.   
[14] Anirudh Ganesh, Ashwin R Jadhav, and KA Cibi Pragadeesh. Deep learning approach for recognition of handwritten kannada numerals. In International Conference on Soft Computing and Pattern Recognition, pages 294–303. Springer, 2016.   
[15] Chhavi Yadav and Léon Bottou. Cold case: The lost mnist digits. Technical report, arxiv-1905.10498, may 2019.   
[16] Han Xiao, Kashif Rasul, and Roland Vollgraf. Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms. 2017.   
[17] Tarin Clanuwat, Mikel Bober-Irizar, Asanobu Kitamoto, Alex Lamb, Kazuaki Yamamoto, and David Ha. Deep learning for classical japanese literature. 2018.   
[18] Sudoku solver 2. https://gist.github.com/mineshpatel1/ 209038c64c19d5e78e0a878320797631#file-sudoku_cv-py, 2017. [Online; accessed 16- July-2019].   
[19] Tensorflow, mnist and your own handwritten digits. https://medium.com/@o.kroeger/ tensorflow-mnist-and-your-own-handwritten-digits-4d1cd32bbab4, 2016. [Online; accessed 16-July-2019].   
[20] Daniel C. Castro, Jeremy Tan, Bernhard Kainz, Ender Konukoglu, and Ben Glocker. Morpho-MNIST: Quantitative assessment and diagnostics for representation learning. 2018.   
[21] Leland McInnes, John Healy, and James Melville. Umap: Uniform manifold approximation and projection for dimension reduction. arXiv preprint arXiv:1802.03426, 2018.   
[22] Vinay Uday Prabhu, Sanghyun Han, Dian Ang Yap, Mihail Douhaniaris, Preethi Seshadri, and John Whaley. Fonts-2-handwriting: A seed-augment-train framework for universal digit classification. arXiv preprint arXiv:1905.08633, 2019.   
[23] Marcus D Bloice, Peter M Roth, and Andreas Holzinger. Biomedical image augmentation using augmentor. Bioinformatics, 2019.

![](images/c39c14785154756448615f677bd39859b6b99f44b32fb98fb1326744f0ff6604.jpg)  
Figure 7: The main dataset creation workflow

![](images/6461e6953bb67ce5000daa7555c57debc3454633b48be4fce22ad437f02c1175.jpg)

![](images/23bd41071a905871b4d2b3f211f719772669a4842c95de199927b095ba878acd.jpg)  
Figure 8: Class-wise mean images of the train set, the test set and the difference between the means of the train and test sets   
Figure 9: Preparing the dig-dataset in Redwood City

![](images/df6618e523361831bc7a3195ed10be88e4bfd793838e96f4972baea8edc75156.jpg)  
Figure 10: User-wise class-wise mean images for the Dig-10k dataset

![](images/71165c6b6d343b59d64f360c44f8b893daad46dc86f2efe023adc53e6b18e492.jpg)

![](images/d9353b1d5611b55311f3732e618094c8d7b8d0bf90e8ce4caddddcb687ac5c3b.jpg)  
Figure 11: Mean pixel-wise intensities comparisons between MNIST and the Kannada-MNIST datasets

![](images/b7f5dad83bc09c13d91a47451fbad399db76c1574ec81fa4c767edda2fd6797c.jpg)  
(a)MNIST   
(b) Kannada-MNIST

![](images/035b8cde601bf6a0397cdbea23e09667f5876f3ac2b6aa911574cc014cccb64e.jpg)  
Figure 12: Morphological comparisons between MNIST and the Fashion-MNIST

![](images/bbad6efdc382091a3753ea09939e591d210e772dc33d7f38722367324253406e.jpg)  
Figure 13: PCA analysis for the two datasets

![](images/2973de6478c6611c437ef8afe2c6aa3e76d9b1e2c2c943d3d1f17da0a6f14c9f.jpg)

![](images/c612d764e722872903c31e32306aed894b5d4945c50550e3e3b00a174354613a.jpg)  
Figure 14: Two-dimensional Uniform Manifold Approximation and Projection (UMAP) plots for the two datasets

![](images/61a55013aa7ee692019d6287fd854ecbbc2bccda846413192482f66acb9ec023.jpg)  
Figure 15: The CNN architecture used in the paper

![](images/efe86786794b84f90878e06b83380843ba5240eaaf312fac7b5137516fdb3e71.jpg)  
Figure 16: Confusion matrix with regards to the Kannada-MNIST datasets

![](images/97c773328a839f20a318e494d9b97d1adbe90dd7ad0e5537a4fce2b44a6759a7.jpg)

![](images/5f94ee8c2f067c55ba67377893694b1c192936fec4a46109165c5b6ad6a5191e.jpg)  
Figure 17: Train and test accuracies for the CNN trained and tested on the main dataset

![](images/00881aa764006c58a211d8ae2c00a4197dee93a5079d687bc5e3ce6b11528807.jpg)  
Confusion matrix,without normalization   
Figure 18: Confusion matrix for the dig-10k dataset

![](images/abe688cf1c9240513fbb8086fe0a3cef5fb39c3bbdde38cfb489037592f6186c.jpg)  
Confusion matrix,without normalization   
Figure 19: Un-normalized confusion matrix for the 1280 digits dataset using in [22]

![](images/67e1407dd9084ea9798b38a0eb0055d556b5f7c44118762f21f7c414c9821cff.jpg)  
Figure 20: Images belonging to class-0 in the 1280-digits dataset that were misclassified by the CNN trained on the main dataset

![](images/43eac576fd0eb3a52bab6ffadc1ab3ed7edc5fb84130a901935f02673408eca0.jpg)  
Figure 21: Images belonging to class-7 in the 1280-digits dataset that were misclassified by the CNN trained on the main dataset

![](images/649e0cdf7b702051cf8c3d772fdbb312fff3b2e388a619c47e7da581b2e315ae.jpg)  
Figure 22: Images belonging to class-8 in the 1280-digits dataset that were misclassified by the CNN trained on the main dataset

![](images/e0e4c7669c1752eb6a48702558bb1039500a18271c8cf0aa1ef8fb2825f264e8.jpg)

![](images/d5b8584aa062371a3941c9aff234ec4e6b288d085d30a6d2091a490731991a7a.jpg)  
Figure 23: Photos of hard copies of the handwritten sheets for the two datasets