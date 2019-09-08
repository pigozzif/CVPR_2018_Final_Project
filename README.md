# CVPR_2018_Final_Project
This repository hosts the material for the final project of the Computer Vision and Pattern Recognition course, offered at University of Trieste in the academic year 2018/2019. All the scripts have been written in the Python programming language. In particular, the following dependencies have been employed: `opencv-python`, `sklearn`, `numpy`, `scipy`.
## Problem Statement
The goal of this project is to implement an image classifier that is able to distinguish between different scenes. The approach adopted is the Bag-of-Words (BoW) model. The dataset to carry out the experiments (from [Lazebnik et al., (2006)]) has been provided and consists of 15 scene categories (office, kitchen, living room, bedroom, store, industrial, tall building, inside city, street, highway, coast, open country, mountain, forest, suburb). As we can see, the scenes can be conceptually "clustered" in three groupings: outdoors (e.g. coast, open country, mountain, forest), indoors e.g. (office, kitchen, store) and hybrid scenes (e.g. suburb). This distinction will become more evident as we present the results. Images were split between a 1500-sized training set and a 2985-sized test set.
## Approach and Implementation
The following tasks have been tackled and accomplished:
### 1. Visual Vocabulary
A visual vocabulay has been built has follows:
1. I computed SIFT descriptors from all of the training images, building a quite large bank of 762,331 descriptors (then, each image had on average ~500 descriptors extracted). To do so I used the OpenCV implementation of SIFT to detect keypoints and then obtain a descriptor at the same location. To iterate over the training images, I defined a generator function to yield an image at a time, thus reducing memory consumption quite a lot. This functionality has been exploited throughout the project anytime there was the necessity to access the image files.
2. Of these features, I sampled 100,000 to actually build the vocabulary. I quantized the 128-dimensional descriptors into k clusters. Investigated values for the parameter k were: 25, 50, 100, 200. To perform clustering I used the Scikit-Learn implementation of the k-means algorithm, which (after fitting) stores very handly the cluster centers in an attribute. In fact, the centroids themselves represent our visual words.
### 2. Compute Histograms
Given the vocabulary, I represented each image of the training set as a normalized histogram of k bins, mapping to the occurences of each of the visual words in the image. In particular, each descriptor detected at very beginning of point 1 was coupled to the closest word in the 128-dimensional feature space. In doing so, the `vq` function inside the `scipy.clister.vq` module turned out very helpful (it automatically performs the nearest-neighbor search among a set of k-means centroids). For normalization, I computed the standard deviation of the single features (i.e. the bins) across the whole training set, and divided by them to obtain features with unit standard deviation. The standard deviation have been stashed for reuse at test time.**This approach turned out to outperform other solutions (like computing relative-frequency histograms) quite a lot**.
### 3. Nearest-neighbor classifier
A 1-nearest-neighbor classifier was constructed and fitted to the training data, of type `sklearn.neighbors.KNeighborsClassifier`. Its performance was then evaluated on the test set. For each test image, the SIFT keypoints were detected and the descriptor computed on top of them. The descriptors are then quantized into a k-bins histogram as done in the previous point for the training images and normalized using the stashed standard deviations. The classifier was done queried for every test image and the predictions used to build a confusion matrix, from which the accuracy can be easily computed. The results are reported below.
### 4. Linear SVM classifier
A linear SVM classifier was constructed and fitted to the training data. As required, 15 different classifiers (each one belonging to the `sklearn.svm.SVC` class), each one discriminating one class against the remaining others, were trained. In doing so, we follow the one-vs-rest philosophy to multiclass classification. Predictions were formulated evaluating the decision function on the test vector (without applying thresholding), and then assigning the image to the class having the highest real-valued output, coinciding with the class having the furthest hyperplane. The results are reported below.
### 5. Chi-squared kernel (optional)
A SVM classifier having a generalized Gaussian kernel with chi-squared distance was implemented. In particular, this entailed the abandoment of the `LinearSVC` object and its substitution with an instance of the `sklearn.multiclass.OneVsRestClassifier` with a `sklearn.svm.SVC` class at the base. To allow the algorithm to perform the "kernel trick", the kernel was declared `precomputed` in the constructor and the Gram matrix computed among all the training examples using the Chi-squared distance (whose implementation is rather straight-forward). Subsequently, we follow the procedure suggested by [Zhang, 2007] and take the mean of all the distances across the training set (the upper triangular portion of the Gram matrix). The corresponding Gaussian generalized kernel is immediate to obtain. The results are reported below.
### 6. Soft-assignment (optional)
A soft-assignment scheme was developed in accordance with what illustrated in [Van Gemert, 2008]. The main idea is that hard-assignment (each descriptor votes only for the closest centroid) suffers from two dilemmas, for a given descriptor: first, there migth be one or more other visual words beyond the closest that reside quite nearby, but do not get any vote; second, if the descriptor is quite far from all of the centroids, it might be the case that it does not belong to any cluster (see outliers). As a result, the authors of the reference paper propose to use a kernel density estimator instead of the histograms (which, by the way, can be seen as rough density approximators). To do so a kernel has to be defined. In accordance with the paper, I implemented a Gaussian kernel with Euclidean distance. When quantizing the descriptors for a training or test image, the kernel is evaluated between the descriptor and every visual word, and the result gets accumulated in the corresponding bin. As the authors put it, we are building a "kernel codebook" by placing one kernel density in correspondence of each visual word. As did in the paper, we use a histogram intersection kernel for the multiclass SVM and set the scale parameter for the Gaussian kernel between 100 and 200 (again, following the best results achieved by the authors).
### 7. Spatial Pyramid Kernel (optional)
Following the ideas of [Lazebnik et al., 2006], I implemented the SPK. Each image was partitioned into 21 subregions spre three levels: level 0 with the whole image, level 1 with 4 quadrants and level 2 with 16 quadrants. An histogram is computed for each of the subregions as done previously, weighted with weights proportional to the level (consistent schemes have been proposed, but I followed the choice of the paper to weight levels 0 and 1 with a coefficient of 0.25 and level 2 with a coefficient of 0.5) and all of them concantenated into one "long" histogram vector for each image. A SVM is then trained using a histogram intersection kernel, as done in the paper. The results are reported below.
## Results
In all our experiments, the baseline is given by the performance of a classifier that always predicts the most numerous test class (being open country with 310 pictures), having a performance of 100 / (2985 / 310) = 10.39%. We will thereon refer to it as a "trivial" classifier. On the other side, a random classifier would score 100 / 15 = 6.67%, a less conservative baseline estimate. 
NOTE: some of the plots have the "predicted class" axis label cut. Nevertheless, I believe this is sort of implicit and the plots still make sense.
1. Here are the results for the **nearest neighbor** classifier described at point 3 of the previous paragraph, including a confusion matrix computed for k=50 visual words and normalized with respect to the true class.
![alt text](https://github.com/pigozzif/CVPR_2018_Final_Project/blob/master/images/KNN50.png)
|k  | Accuracy|
|---|--------:|
|25 |0.24     |
|50 |0.26     |
|100|0.24     |
|200|/        |
As we can see, a nearest-neighbor classifier performs quite better than a trivial one, meaning that some learning has actually taken place. On the other hand, we should not think this is a really good classifier, since the accuracy score leaves a lot of room for improvement. At the same time, we see that a couple of "outstanding" classes (forest and suburb) are easily recognized by the algorithm, while for a big portion of the remaining classes the behavior seems to be rather "stochastic". Notice that the distance metric used to find the nearest neighbor has been the Earth Mover's Distance, that is very suitable at comparing histograms as we know. Other metrics have been tried (Euclidean, Manhattan, ...) yielding worse results, which have been omitted for brevity.
Notice that results for k=200 were not produced, since the evaluation of the EMD distance among histograms turned out to require an overwhelming amount of hours. In fact, according to the OpenCV documentation, computing the EMD requires solving a linear programming problem whose complexity is exponential in the number of elements to match (being it 2xk) in the worst case. Still, we can speculate that performance would not have been so good.
2. Here are the results for the **linear SVM** classifier illustrated at point 4 of the previous paragraph, including a confusion matrix computed for k=50 visual words and normalized with respect to the true class.
![alt text](https://github.com/pigozzif/CVPR_2018_Final_Project/blob/master/images/linearSVC50.png)
|k  | Accuracy|
|---|--------:|
|25 |0.38     |
|50 |0.39     |
|100|0.43     |
|200|0.45     |
As we can see, the linear SVM does a pretty decent job at classifying different scenes, jumping by more or less 14 percentage points over the KNN classifier in terms of accuracy. Of course, we are not really surprised by this result, since Support Vector Machines are a much more sophisticated tool than a trivial nearest-neighbor search. Predictions start to accumulate on the main diagonal (denoting success), and we witness a very decent performance for outdoors scenery (like coast, forest, highway, mountain, suburb). Intuitively, indoors turn out tough to classify and in particular to distinguish, since misclassifications are strong between them.
3. Here are the results for the SVM with **chisquared** kernel illustrated at point 5.
|k  | Accuracy|
|---|--------:|
|25 |0.43     |
|50 |0.51     |
|100|0.54     |
|200|0.55     |
As we can see, this kernel seems to squeeze out a lot of performance from the SVM mechanism and consistently outperforms the simpler linear kernel. Indeed, this has been my top performer, with an astonishing 55% of accuracy at k=200. Notice the big jump in performance going from k=25 to k=50.
4. Here are the results for the **soft-assigment** rule discussed at point 6, including a confusion matrix computed for k=50 visual words and normalized with respect to the true class.
![alt text](https://github.com/pigozzif/CVPR_2018_Final_Project/blob/master/images/soft50.png)
|k  | Accuracy|
|---|--------:|
|25 |0.48     |
|50 |0.45     |
|100|0.47     |
|200|0.51     |
As we can see, this approach consistently outranks the linear SVM classifier in terms of performance (even if computing the kernel many times impacted the time of computation). In fact, this turned out to be the top performer for k=25 beyond any doubt, outpacing the chisquared kernel by 5 percentage points. As a result, we can conclude that the intuition behind allowing each descriptor to contribute to multiple bins in a distance-weighted fashion was a winning one. Interestingly enough, if we compare this confusion matrix with that for the linear SVM, we discover that this approach abandons the unwanted preference for predicting "bedroom". 
5. Here are the results for the **spatial pyramid kernel** analyzed at point 7.
|k   |Accuracy |
|----|--------:|
|25  |0.41     |
|50  |0.46     |
|100 |0.48     |
|200 |0.52     |
As we can see, even this solution seems to deliver pretty good performance. As expected, the intuition behind the SPK was not a wrong one, and augmenting the plain BoW approach with a notion of location helps us achieving better results.

