# CVPR_2018_Final_Project
This repository hosts the material for the final project of the Computer Vision and Pattern Recognition course, offered at University of Trieste in the academic year 2018/2019. All the scripts have been written in the Python programming language. In particular, the following dependencies have been employed: `opencv-python`, `sklearn`, `numpy`, `scipy`.
## Problem Statement
The goal of this project is to implement an image classifier that is able to distinguish between different scenes. The approach adopted is the Bag-of-Words (BoW) model. The dataset to carry out the experiments (from [Lazebnik et al., (2006)]) has been provided and consists of 15 scene categories (office, kitchen, living room, bedroom, store, industrial, tall building, inside city, street, highway, coast, open country, mountain, forest, suburb). As we can see, the scenes can be conceptually "clustered" in three groupings: outdoors (e.g. coast, open country, mountain, forest), indoors e.g. (office, kitchen, store) and hybrid scenes (e.g. suburb). This distinction will become more evident as we present the results. Images were split between a 1500-sized training set and a 3000-sized test set.
## Approach and Implementation
The following tasks have been tackled and accomplished:
### 1. Visual Vocabulary
A visual vocabulay has been built has follows:
1. I computed SIFT descriptors from all of the training images, building a quite large bank of 762,331 descriptors (then, each image had on average ~500 descriptors extracted). To do so I used the OpenCV implementation of SIFT to detect keypoints and then obtain a descriptor at the same location. To iterate over the training images, I defined a generator function to yield an image at a time, thus reducing memory consumption quite a lot. This functionality has been exploited throughout the project anytime there was the necessity to access the image files.
2. Of these features, I sampled 100,000 to actually build the vocabulary. I quantized the 128-dimensional descriptors into k clusters. **Investigated values for the parameter k were: ... 50 ...**. To perform clustering I used the Scikit-Learn implementation of the k-means algorithm, which (after fitting) stores very handly the cluster centers in an attribute. In fact, the centroids themselves represent our visual words.
### 2. Compute Histograms
Given the vocabulary, I represented each image of the training set as a normalized histogram of k bins, mapping to the occurences of each of the visual words in the image. In particular, each descriptor detected at very beginning of point 1 was coupled to the closest word in the 128-dimensional feature space. In doing so, the `vq` function inside the `scipy.clister.vq` module turned out very helpful (it automatically performs the nearest-neighbor search among a set of k-means centroids). For normalization, I computed the standard deviation of the single features (i.e. the bins) across the whole training set, and divided by them to obtain features with unit standard deviation. The standard deviation have been stashed for reuse at test time.**This approach turned out to outperform other solutions (like computing relative-frequency histograms) quite a lot**.
### 3. Nearest-neighbor classifier
A 1-nearest-neighbor classifier was constructed and fitted to the training data, of type `sklearn.neighbors.KNeighborsClassifier`. Its performance was then evaluated on the test set. For each test image, the SIFT keypoints were detected and the descriptor computed on top of them. The descriptors are then quantized into a k-bins histogram as done in the previous point for the training images and normalized using the stashed standard deviations. The classifier was done queried for every test image and the predictions used to build a confusion matrix, from which the accuracy can be easily computed. The results are reported below.**missing results and discussion about distance metric**
### 4. Linear SVM classifier
A linear SVM classifier was constructed and fitted to the training data. I used the `sklearn.svm.LinearSVC` class since it uses the one-vs-rest philosophy for multiclass classification under-the-hood. Predictions were formulated as done at the previous point, and the results are reported below.
### 5. Chi-squared kernel (optional)
A SVM classifier having a generalized Gaussian kernel with chi-squared distance was implemented. In particular, this entailed the abandoment of the `LinearSVC` object and its substitution with an instance of the `sklearn.multiclass.OneVsRestClassifier` with a `sklearn.svm.SVC` class at the base. To allow the algorithm to perform the "kernel trick", the kernel was declared `precomputed` in the constructor and the Gram matrix computed among all the training examples using the Chi-squared distance (whose implementation is rather straight-forward). The results are reported below.**reason for failure**
### 6. Soft-assignment (optional)
A soft-assignment scheme was developed in accordance with what illustrated in [Van Gemert, 2008]. The main idea is that hard-assignment (each descriptor votes only for the closest centroid) suffers from two dilemmas, for a given descriptor: first, there migth be one or more other visual words beyond the closest that reside quite nearby, but do not get any vote; second, if the descriptor is quite far from all of the centroids, it might be the case that it does not belong to any cluster (see outliers). As a result, the authors of the reference paper propose to use a kernel density estimator instead of the histograms (which, by the way, can be seen as rough density approximators). To do so a kernel has to be defined. In accordance with the paper, I implemented a Gaussian kernel with Euclidean distance. When quantizing the descriptors for a training or test image, the kernel is evaluated between the descriptor and every visual word, and the result gets accumulated in the corresponding bin. As the authors put it, we are building a "kernel codebook" by placing one kernel density in correspondence of each visual word.
### 7. Spatial Pyramid Kernel (optional)
Following the ideas of [Lazebnik et al., 2006], I implemented the SPK. Each image was partitioned into 21 subregions spre three levels: level 0 with the whole image, level 1 with 4 quadrants and level 2 with 16 quadrants. An histogram is computed for each of the subregions as done previously, weighted with weights proportional to the level (consistent schemes have been proposed, but I followed the choice of the paper to weight levels 0 and 1 with a coefficient of 0.25 and level 2 with a coefficient of 0.5) and all of them concantenated into one "long" histogram vector for each image. A SVM is then trained using a histogram intersection kernel, as done in the paper. The results are reported below.
## Results

1. Here are the results for the **nearest-neighbor** classifier described at point 3 of the previous paragraph, including a confusion matrix computed for k=50 visual words.

2. Here are the results for the linear SVM classifier illustrated at point 4 of the previous paragraph, including a confusion matrix computed for k=50 visual words.

