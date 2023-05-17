# Week 1 
* 


# Week 2
1. How is margin related to regularization?
* The model is regularized by increasing the margin between the decision boundary. Margin is the distance between two samples from each class, while regularization is to make model not too comple by controling weights.
-----

2. Why is regularization needed?
* Regularization can help to reduce overfitting in the model. It controls the model complexity so that weights in the model will not be too large. Having small weight in the model means samll change in the $x$ - input will not make big change in output $\hat{y}$
-----

3. How is structural risk minimization different from empirical risk minimization?
* SRM controls the model complexity by changing the $\lambda$ parameter and reduces the hypothesis space and overfitting and focuses on generalization performance.
* ERM describes how well the model fits the training data. It learns from training error and pays attention on misclassification error over training data and aims to minizes the error.
-----

4. What is role of the parameter "C" in an SVM?
* $C$ controls the trade-off between model complexity and training error minimization. If $C$ is large, the model will focus on training error minimization and ignore model complexity, this may result in overfitting and bad generalization performance.
-----

5. As discussed in the lectures, some SVM formulations use "Lambda" in their expression instead of "C". What is the relationship between "C" and "Lambda"?
* $C$ is used in as the constant coefficient in the loss function part while $\lambda$ is used in regularization part. They have the same function, to control the trade off between model complexity and training error minimization. 
-----

6. What is a support vector?

* The points that determine the margin are called support vectors
------

7. Write the weight update step from gradient descent for an SVM?

* Objective function is following: $ L(w)= \frac{1}{N}\sum max(0, 1-y_i \cdot (w^Tx_i+b)) + \frac{\lambda}{2}\cdot ||w||^2 $ 
$$ ∇(loss function) = 0 -if-y_i\cdot(w^Tx_i+b)>1, -yx - else $$
$$ = 1(y_if(x)<1)(-yx) $$
$$ ∇P=\lambda\cdot w-1(yf(x)<1)(-yx)$$
$$ = \lambda\cdot w+1(yf(x)<1)(yx) $$

$$ w_k \leftarrow w_{k-1} -\alpha 1(yf(x)<1)(yx) $$
-----

8. How can you determine if the objective function in the optimization problem underlying an SVM is convex or non-convex?

* Obtain the partial derivate of objective function in terms of $w$. If the derivative function is a quadratic function, compute when it is equal to $0$ and calculate its second partial derivative. If the second partial derivative is positive then it is convex, or it is not convex.
----

9. What are the advantages of linear discriminant based classifier such as perceptron or a support vector machine?

* If the data is linearly separable then the classifier can easily find the solution. 
* Linear classifier produces decision boundaries that are linear combinations of input features.
* They are robust and less prone to overfitting compare to complex models. They have a good generalization performance by balancing model complexity.
-----

10. How do transformations of data points change the concept of distance in the feature space?

* It can transformed by kernel function or feature transformation to higher dimensionality
-----

11. How do transformations of data points allow us to use a linear discriminant in the transformed space to solve an originally linearly non-separable classification problem?

* The transformation of data points change the concept of distance or dot product between two points. This allows us to use linear discriminant to solve linearly non-separable problem.

* There are scaling, translation, rotation and non-linear transformations to transform the concept of distance.
-----

12. What is the relationship between distance and dot products?

* Euclidean Distance can be showed by computing dot product of vectors in an n-dimensional space.
$$ ||x-y|| = \sqrt{||x||^2-2\cdot x \cdot y +||y||^2} $$
-----

13. How can we achieve implicit transformations of the data by changing the definition of dot products to kernel functions?

* Redefine the dot product in the original space to be the dot product in the transformed space, using a kernel function. 
* A kernel function denoted by $ K(x,y)$, is a function that takes two data points $x$ and $y$ in the input space and computes the dot product between their images in a higher-dimensional feature space.
-----

14. What is a kernel function?

* Kernel function is to map the original non-linear space into a higher-dimensional space where data points are linearly separable.
-----

15. What is the representer theorem?

* **Representer theorem** allows us to represent the SVM weight vector as a linear combination of input vectors with each example's contribution weighted by a factor $\alpha_i$.
-----

16. How does the representer theoreom allow us to use kernels in an SVM?

* The representer theorem the optimal solution $w$ can be represented as $w=\sum{a_ix_i}$. Then $$ f(x)=w^Tx+b=b+ \sum_{j=1}^{N}a_jx_j^Tx $$
$$ f(x)=b+w^Tx = b+\sum_{j=1}^{N} a_jk(x_j,x) $$ 
$$ w^Tw = (\sum_{i=1}^{N}a_ix_i)^T \sum_{j=1}^{N}a_jx_j = \sum_{i,j=1}^{N}a_ia_jk(x_i,x_j) $$
The optimization solution will be obtain the $\alpha $
$$ f(x)=b+\sum_{j=1}^{N}a_jk(x_j,x) $$
We only need to compute the kernel function between pairs of data points, without computing the transformed data points.
-----

17. Write the discriminant function of a support vector machine using the representer theoreom?

* $$ f(x)=w^Tx+b=b+\sum_{j=1}^{N}a_jx_j^Tx $$ 
Here is the discriminant function of SVM using the representer theoreom:
$$ min_{a,b}\sum_{i,j=1}^{N}a_ia_jk(x_i, x_j) + \frac{C}{N}\sum_{i=1}^{N}max(0, 1-y_i(\sum_{j=1}^{N}a_jk(x_i,x_j)+b)) $$ 
-----

18. Obtain an expression for the optimization problem underlying a kernelized support vector machine in terms of the data points and the representer theorem.

* $$ f(x) = b+\sum_{j=1}^{N}a_jk(x_j, x) $$
-----

19. Solve the optimization problem underlying a kernelized support vector machine with respect to "alpha" using gradient descent algorithm by writing the update expression for alpha.

* 
-----

20. What is the Gram matrix?

* Gram matrix is symmetric, positive semi-definite (for all given data)
-----

21. What are the conditions for a function to be a valid kernel function?

* The function is considered as a valid kernel function if it is symmetric, which means $k(x,y) = k(y,x) $. It needs to be positive semi-definite.

* Any function k can be a kernel if it is Gram matrix.
-----

22. What is the role of the degree (d) and coefficient (c) in a polynomial kernel k(a,b)=(a^t b + c)^d? What impact do these have on the classification boundary?

* Degree $d$ determines the complexity of the decision boundary in the transformed feature space. Higher degree can capture more complex, non-linear relationships between feature and target variable. However, it may arise overfitting if degree $d$ is too large especially limited amount of training data.

* Coefficient $C$ (bias term) affects the shape of the decision boundary by controlling the importance of lower-degree terms. Larger value of $C$ makes the decision boundary more flexible, helps the model to adapt to data. Lower value of $C$ makes the decision boundary more complex.
-----

23. What is the role of the parameter "gamma" in an RBF kernel? How does it affect the classification boundary?

* $\gamma$ parameter controls the shape and smoothness of the decision boundary. It determines how sensitive the model is to the distance between data points and affects the trade-off between model complexity and generalization performance.
-----

24. How can you specify your own kernel function in the SVM?

* 
-----

25. How does using a kernel matrix eliminate the need for explicit feature representation of examples for a classification problem?

* 
-----


# Week 3

1. What is the objective of validation?

* Validation is used to measure the generalization performance (Use part of training set to approximate the generalization performance)
---- 

2. What is stratified validation?

* Stratified cross-validation is to ensure the each fold of dataset has the same proportion of observations with a given label. While K-fold divides data into K folds
----

3. What are underlying assumptions for accuracy as a metric?

1. The dataset is balanced 
2. misclassfication of any class is equally bad
3. The threshold for classifier is optimal
----

4. What is precision, recall, false positive rate?

* $$ Precision = \frac{TP}{TP+FN} $$
* $$ Recal/Sensitivity=\frac{TP}{TP+FN} $$
* $$ FPR =\frac{FP}{FP+FN} $$
----

5. Why are accuracy, precision, recall etc. dependent upon the threshold of the classifier?

* The change of Threshold may result in the change of label over data. The label change will lead to the change of these metrics.
----

6. How do precision, recall and false positive rate change as the threshold of the classifier is increased?

* Precision **increases**
* Recall **decreases**
* FPR **decreases**
With threshold increasing, the false positive is reduced
-----

7. What is the ROC curve?

* ROC curve describes the trade-off between TPR and FPR of the model. A good ROC curve should closer to left-upper corner, which is the model has high TPR with small FPR.
-----

8. How does area under the ROC curve serve as a performance metric?

* The area under the ROC curve is a quality metric. The higher ROC curve, the better the model 's performance at classifying class
----

9. Why is the ROC curve called the ROC curve?

* The ROC curve demonstrates the performance of the model with changing the threshold. It gives the usefulness of the model
----

10. How does the performance estimate of your model change with increase in the size of your validation set?

* If validation set increase, then the approximation performance may be worse since there are fewer training examples in the training phase, so the model learns insufficiently.
-----

11. What are the limitations of ROC curves?

* ROC curve can't compare two different models to show which is better
* ROC shouldn't be used when number of negative samples > positive samples by a huge amount (class imbalance)
* Shouldn't be used when we care about the prediction of one class.
-------

12. What is the most important region of an ROC curve?

* The left-upper corner of ROC curve shows the accuracy of the test. Hence, the closer the ROC curve the better accuracy of the test.
------

13. How is a precision recall curve useful?

* PR curve depicts the trade-off between precision and recall. It is useful in the imbalanced data and when precision is a requirement. 
-----

14. What is the relationship between the ROC and Precision-Recall curves?

* One-to-one correspondence between two cruves
* If a curve dominates in ROC curve then it dominates in PR curve. Vice versa
----

15. What are the limitations of the precision recall curve?

* 
----

16. How do we train the final model for deployment?

* We can use cross-validation to measure the generalization performance. If data set is small then LOOCV is preferred to use. 
* Grid search is used tu tune the hyperparameter of the model and find the best hyperparameter
* Find good metrics to evaluate the generalization performance
----

17. How can you choose an "operating point" for a machine learning model?

* ROC curve is used to find operating point. This point lies on a 45 degree line cloest to the left-upper corner.
----

18. What is the impact of the choice of K in K-fold cross validation on performance statistics?

* If $K$ is large, then there are more training examples in each fold. This will reduce the prediction error and give a better approximation performance.
----

19. What K should we use?

* 
-----

20. What is F1?

* F1 score is a performance metric combines precision and recall.
$$ F_1=2\cdot\frac{precision\cdot recall}{precision+recall} =\frac{2tp}{2tp+fp+fn}$$
-----

21. What is Matthews Correlation Coefficient?

* MCC is to measure the difference between the predicted values and actual values
-----

22. Why are FPR and TPR monotonically non-increasing functions of threshold but precision is not?

* With the threshold increasing, $TP$ and $FP$ are reduced.  
* When threshold is $-\infty$, TPR and FPR are $100$% since all examples are positive, there is only FP. 
$$ TPR = \frac{TP}{TP+0}=1 $$
$$ FPR = \frac{FP}{FP+0} = 1 $$

* With threshold increasing to $\infty$, all examples will classified to negative label. So no positive examples left.
$$ TPR = \frac{TP=0}{TP=0+FN}=0 $$
$$ FPR = \frac{FP=0}{FP=0+TN}=0 $$

-----

23. What is grid search?

* Grid search is exhaustive search through corss-validation to find optimal hyperparameters of the model
----

24. What is bootstrapping? What are .632 and .632+ bootstrap?

* Bootstrapping is a statistical technique for estimating quantities about a population by averaging estimates from multiple small data samples.
----


# Week 4

1. What is meant by a manifold?

* **Manifold** is a generalization of Euclidean space. It has locally properties of Euclidean space and can be computed with Euclidean distance. This helps for dimensionality reduction
----

2. **What is the primary idea behind incremental PCA**?

* Incremental PCA build principle component basis incrementally, which is not robust. 
* Incremental PCA updates the principle components by using new data samples without directly computing the covariance matrix.
----

3. What is the SRM formulation of PCA?

* $$ min_w \frac{\lambda}{2}w^Tw +  V -w^TCw  $$

* Optimization:
$$ Cw=\lambda w $$
----

4. What is the fundamcental idea of robust regression?

* Robust regression is implemented by adding the regularization to loss function. It imposes a penalty on the size of coefficients. This makes the model more stable to outliers than least square regression. 
----

5. What is UMAP? How does it work? What is the role of the neighborhood and distance constraint parameter for UMAP?

* UMAP is machine learning method for dimensionality reduction on manifold learning. 

* **Neighborhood** parameter balances local and global structure in the data. It does this by constraining the size of the local neighborhood.

* **Distance** parameter controls how tightly UMAP is allowed to pack points together. 
----

6. What is meant by t-SNE? How does it work?

* It is called **t-distributed stochastic neighbor embedding (t-SNE)**. It is a statistical method for dimensionality reduction and is used to understand high-dimensional data and project it into low-dimensional space (2D or 3D).  

* t-SNE comprises two stages: construct a probability distribution over pairs of high-dimensional data and then it defines a similar probability distribution over the points in the low-dimensional map, and it minimizes KL divergence.
----

7. How can PCA be kernelized?

* The PCA can use the 'kernel trick' to kernelize: 3
$$ min_w \frac{\lambda}{2}w^Tw+V-w^TCw  $$
$$ w=\sum_{i=1}^{N}a_jx $$
$$ min_w\frac{\lambda}{2}\sum_{i,j=1}^{N}a_ia_jk(x_i,x_j)+V-\sum_{i,j=1}^{N}a_ia_jCk(x_i,x_j) $$
-----

8. What does "Component Analysis" mean in general? How is it different from "Discriminant Analysis"?

* It simplifies the complexity in high-dimensional data while retaining trends and patterns

* The difference between PCA and discriminant analysis is PCA calculates the best discriminating components without foreknowledge about groups, while discriminant analysis calculates the best discriminating components for groups that are defined by user.
-----

9. What is meant by regression? How is it different from classification?

* **Regression** is to estimate relationship among variables. 

* Regression outputs the continuous value while classification outputs the discrete value (label). For example, to predict the male and age of man is the difference of classification and regression.
------

10. Derive a closed-form formula for the optimal weights of ordinary least squares regression?

* The Optimization expression:
$$ min_{w,b}w^Tw +\frac{C}{N}(w^Tx-y_i)^2  $$

$$ min_wL(X,Y;w) $$
$$ L = (Xw-y)(Xw-y) $$
$$ = (w^TX^T-y^T)(Xw-y) $$
$$ = w^T X^T Xw - w^TX^T y - y^TXw + y^Ty $$
$$ = 2X^TXw - X^Ty - (y^TX)^T $$
$$ = 2X^TXw - 2X^Ty =0 $$
$$ \rightarrow w = (X^TX)^{-1}X^T y $$

$$ w = X^{+}y $$  

$$ X^{+} = (X^TX)^{-1}X^T $$
-----

11. Can OLS be used for classification? Give a justification of why this is or is not a good choice.

* Yes, the OLS regression can be used for classification as the continuous output value can be considered as the probality of the class. 

* It attempts to find the optimal decision boundary. 
-----

12. What is the difference in terms of Representation, Evaluation and Optimization between each of the following models: Ordinary Least Squares Regression, Ridge Regression, Lasso Regression, Support Vector Regression, Logistic Regression?

* **OLSR** REO:
1. Representation: $$ f(x) = w^Tx+b $$
2. Evaluation: $$ L(X,Y;w) = (Xw-y)^2 $$
3. Optimization: $$ min_w||(Xw- y)||^2 $$

* **Ridge Regression** REO:
1. Representation: $$ f(x) = w^Tx+b $$
2. Evaluation: $$ L(X,Y;w) = (Xw-y)^2 $$
3. Optimization: $$ min_w\alpha ||w||^2_2 + \frac{C}{N}||Xw-y||^2_2 $$

* **Lasso Regression** REO:
1. Representation: $$ f(x) = w^Tx+b $$
2. Evaluation: $$ L(X,Y;w) = (Xw-y)^2 $$
3. Optimization: $$ min_w||w||_1 + ||(Xw- y)||^2 $$

* **Support Vector Regression** REO:
1. Representation: $$ f(x) = w^Tx+b $$
2. Evaluation: $$ L(X,Y;w) = max(0, |Xw-y|-\epsilon) $$
3. Optimization: $$ min_w||w||^2_2 + \frac{C}{N}\sum_{i=1}^{N} max(0, |Xw-y|-\epsilon) $$

* **Logistic Regression** REO:
1. Representation: $$ f(x) = w^Tx+b $$
2. Evaluation: $$ L(X,Y;w) = log(exp(-y_if(x))+1) $$
3. Optimization: $$ min_{w,b}\frac{1}{2}|w^Tw + C\sum_{i=1}{N} log(exp(-y_i f(x))+1)  $$

-----

13. Is Logistic Regression a regression method or a classification method? Given an explanation for this in terms of the loss function used in logistic regression?

* Logistic regression is a classification method since its loss function returns the probability of the class between 0 and 1.
----

14. What are the limitations of square error loss?

* Square error loss is sensitive to outliers, which may affect the slope of prediction and makes it not stable.
-----

15. What is the motivation behind epsilon-insensitive loss function?

* When absolute error loss function is used, it is not differentiable when error is equal to $0$. To solve this problem, a new term $\epsilon $ is added to absolute loss function. It is differentiable between $-\epsilon$ and $\epsilon$.    
-----

16. What is meant by pseudo-inverse of a matrix?

* It can be represented by: 
$$ pseudo-inverse: (X^TX)^{-1} X^T $$
-----

17. How can you improve hurricane intensity prediction?

* Do data pre-processing to hurricane speed data.
* Try different regression model with regularization and use cross-validation and grid search to tune hyperparameters.
* Use different performance metrics to quantify the generalization performance.
-----

18. How is ordinary least squares regression related to solving a system of linear equations?

* Let's say there is an A with features and an euqation of B = w^TA + b. We can get B by calculating the ordinary least squares regression on to B. 
------


# SRM Questions: 

1. Write  the represenation and evaluation of different types of loss functions used in various problems discussed in this week.

----

2. What are the desired characteristics of loss function for a given problem?


----

3. What is the difference between L0, L1 and L2 regularization?


----

4. What is the impact of L0, L1 and L2 regularization?


----

5. What is the main idea behind: ranking, reinforcement learning, survival prediction, etc.


----

6. What are the performance metrics for each type of machine learning problem discussed in this week?


---

7. How can you use click data to obtain training data in the design of recommendation systems?


---

8. Why is collaborative filtering called "collaborative" filtering?


----

9. How can you use cross-validation and other performance assessment techniques for different types of ML problems discussed in this week?


-----

10. Can you describe the process of obtaining solution to a novel machine learning problem?

----


# Neural networks

1. What is meant by a neuron, soma, axon, dendrites, synaptic gap?


----

2. What is meant by "firing" of a neuron?


-------

3. What is the mathematical model of a neuron?


-------

4. What is meant by activation function?


-----------

5. What is the role of the activation function?

6. What is a neural network?

7. What is meant by a fully connected feed-forward neural network?

8. Write the representation of a fully connected neural network in mathematical form for an input x?

9. What is the evaluation of a FCNN?

10. How can we optimize a neural network?

11. What is meant by a layer of neurons?

12. What is the impact of adding neurons to a neural layer?

13. What is the impact of adding more layers?
