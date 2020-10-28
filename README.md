scikit-learn-Hy
===

**An introduction to [scikit-learn](https://www.scikit-learn.org/) (machine learning in Python) and [Hy](https://www.github.com/hylang/hy/) (a Lisp dialect embedded in Python).**

* [Hy 0.19.0](https://github.com/hylang/hy) and [Python 3.7](https://www.python.org/downloads/release/python-377/) reference platform.
* Examples are in Hy and tests are in Python (to showcase Hy module support).

## Author

[brackendev](https://www.github.com/brackendev)

## License

scikit-learn-Hy is released under the BSD 3-Clause license. See the `LICENSE` file for more info.

## Installation

It is assumed that [Python 3.7](https://www.python.org/downloads/release/python-377/) and [pyenv](https://github.com/pyenv/pyenv) are installed. Via a command shell in the project directory, execute:

```bash
$ pip install -r requirements.txt
```

To run the tests, execute:

```bash
$ pytest scikit_learn_tests.py
```

## Usage

Follow [An introduction to machine learning with scikit-learn and Hy](#an-introduction-to-machine-learning-with-scikit-learn-and-hy) below this **Usage** section.

Additionally, the example code in the [introduction](#an-introduction-to-machine-learning-with-scikit-learn-and-hy) is also available in this project's `scikit_learn` Hy module. For example:

```hy
$ hy
hy 0.19.0+5.gd6af7c4 using CPython(default) 3.7.7 on Darwin
=> (import scikit_learn)
Welcome to scikit-learn-Hy!

=> (scikit-learn.learning-and-predicting)
array([8])
```

Available example functions:
* [learning-and-predicting](#choosing-the-parameters-of-the-model)
* [model-persistence](#model-persistence)
* [model-persistence-from-file](#model-persistence)
* [type-casting-32](#type-casting)
* [type-casting-64](#type-casting)
* [type-casting-more1](#type-casting)
* [type-casting-more2](#type-casting)
* [refitting-and-updating-parameters1](#refitting-and-updating-parameters)
* [refitting-and-updating-parameters2](#refitting-and-updating-parameters)
* [multiclass-vs-multilabel-fitting1](#multiclass-vs-multilabel-fitting)
* [multiclass-vs-multilabel-fitting2](#multiclass-vs-multilabel-fitting)
* [multiclass-vs-multilabel-fitting3](#multiclass-vs-multilabel-fitting)

- - -

An Introduction to Machine Learning with scikit-learn and Hy
===

Machine Learning: The Problem Setting
-------------------------------------

In general, a learning problem considers a set of n
[samples](https://en.wikipedia.org/wiki/Sample_(statistics)) of
data and then tries to predict properties of unknown data. If each sample is
more than a single number and, for instance, a multi-dimensional entry
(aka [multivariate](https://en.wikipedia.org/wiki/Multivariate_random_variable)
data), it is said to have several attributes or **features**.

Learning problems fall into a few categories:

 * [Supervised learning](https://en.wikipedia.org/wiki/Supervised_learning)
   in which the data comes with additional attributes that we want to predict.
   ([Click here](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning)
   to go to the scikit-learn supervised learning page.) This problem
   can be either:

    * [Classification](https://en.wikipedia.org/wiki/Classification_in_machine_learning):
      samples belong to two or more classes and we
      want to learn from already labeled data how to predict the class
      of unlabeled data. An example of a classification problem would
      be handwritten digit recognition, in which the aim is
      to assign each input vector to one of a finite number of discrete
      categories.  Another way to think of classification is as a discrete
      (as opposed to continuous) form of supervised learning where one has a
      limited number of categories and for each of the n samples provided,
      one is to try to label them with the correct category or class.

    * [Regression](https://en.wikipedia.org/wiki/Regression_analysis):
      if the desired output consists of one or more
      continuous variables, then the task is called *regression*. An
      example of a regression problem would be the prediction of the
      length of a salmon as a function of its age and weight.

 * [Unsupervised learning](https://en.wikipedia.org/wiki/Unsupervised_learning),
   in which the training data consists of a set of input vectors x
   without any corresponding target values. The goal in such problems
   may be to discover groups of similar examples within the data, where
   it is called [clustering](https://en.wikipedia.org/wiki/Cluster_analysis),
   or to determine the distribution of data within the input space, known as
   [density estimation](https://en.wikipedia.org/wiki/Density_estimation), or
   to project the data from a high-dimensional space down to two or three
   dimensions for the purpose of *visualization*. ([Click here](https://scikit-learn.org/stable/unsupervised_learning.html#unsupervised-learning)
   to go to the Scikit-Learn unsupervised learning page.)

#### Training Set and Testing Set

Machine learning is about learning some properties of a data set
    and then testing those properties against another data set. A common
    practice in machine learning is to evaluate an algorithm by splitting a data
    set into two. We call one of those sets the **training set**, on which we
    learn some properties; we call the other set the **testing set**, on which
    we test the learned properties.


Loading an Example Dataset
--------------------------

`scikit-learn` comes with a few standard datasets, for instance the
[iris](https://en.wikipedia.org/wiki/Iris_flower_data_set) and [digits](https://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits)
datasets for classification and the [boston house prices dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/housing/) for regression.

In the following, we start a Hy REPL from our shell and then
load the ``iris`` and ``digits`` datasets.  Our notational convention is that
``$`` denotes the shell prompt while ``=>`` denotes the Hy
REPL prompt:

```hy
$ hy
=> (import [sklearn [datasets]])
=> (setv iris (datasets.load-iris))
=> (setv digits (datasets.load-digits))
```

A dataset is a dictionary-like object that holds all the data and some
metadata about the data. This data is stored in the ``.data`` member,
which is a ``n_samples, n_features`` array. In the case of supervised
problem, one or more response variables are stored in the ``.target`` member. More
details on the different datasets can be found in the :ref:`dedicated
section <datasets>`.

For instance, in the case of the digits dataset, ``digits.data`` gives
access to the features that can be used to classify the digits samples:

```hy
=> (print digits.data)
[[ 0.  0.  5. ...  0.  0.  0.]
 [ 0.  0.  0. ... 10.  0.  0.]
 [ 0.  0.  0. ... 16.  9.  0.]
 ...
 [ 0.  0.  1. ...  6.  0.  0.]
 [ 0.  0.  2. ... 12.  0.  0.]
 [ 0.  0. 10. ... 12.  1.  0.]]
```

and ``digits.target`` gives the ground truth for the digit dataset, that
is the number corresponding to each digit image that we are trying to
learn:

```hy
=> (print digits.target)
[0 1 2 ... 8 9 8]
```

#### Shape of the Data Arrays

The data is always a 2D array, shape ``(n_samples n_features)``, although
    the original data may have had a different shape. In the case of the
    digits, each original sample is an image of shape ``(8 8)`` and can be
    accessed using:

```hy
=> (first digits.images)
array([[ 0.,  0.,  5., 13.,  9.,  1.,  0.,  0.],
       [ 0.,  0., 13., 15., 10., 15.,  5.,  0.],
       [ 0.,  3., 15.,  2.,  0., 11.,  8.,  0.],
       [ 0.,  4., 12.,  0.,  0.,  8.,  8.,  0.],
       [ 0.,  5.,  8.,  0.,  0.,  9.,  8.,  0.],
       [ 0.,  4., 11.,  0.,  1., 12.,  7.,  0.],
       [ 0.,  2., 14.,  5., 10., 12.,  0.,  0.],
       [ 0.,  0.,  6., 13., 10.,  0.,  0.,  0.]])
```

The [simple example on this dataset](https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html#sphx-glr-auto-examples-classification-plot-digits-classification-py) illustrates how starting
    from the original problem one can shape the data for consumption in
    scikit-learn.

#### Loading from External Datasets

To load from an external dataset, please refer to [loading external datasets](https://scikit-learn.org/stable/datasets/index.html#external-datasets).

Learning and Predicting
------------------------

In the case of the digits dataset, the task is to predict, given an image,
which digit it represents. We are given samples of each of the 10
possible classes (the digits zero through nine) on which we *fit* an
[estimator](https://en.wikipedia.org/wiki/Estimator) to be able to *predict*
the classes to which unseen samples belong.

In scikit-learn, an estimator for classification is a Python object that
implements the functions ``fit(X, y)`` and ``predict(T)``.

An example of an estimator is the class ``sklearn.svm.SVC``, which
implements [support vector classification](https://en.wikipedia.org/wiki/Support_vector_machine>). The
estimator's constructor takes as arguments the model's parameters.

For now, we will consider the estimator as a black box:

```hy
=> (import [sklearn.svm [SVC]])
=> (setv clf (SVC :gamma 0.001 :C 100))
```

#### Choosing the Parameters of the Model

  In this example, we set the value of ``gamma`` manually.
  To find good values for these parameters, we can use tools
  such as [grid search](https://scikit-learn.org/stable/modules/grid_search.html#grid-search) and [cross validation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation).

The ``clf`` (for classifier) estimator instance is first
fitted to the model; that is, it must *learn* from the model. This is
done by passing our training set to the ``fit`` method. For the training
set, we'll use all the images from our dataset, except for the last
image, which we'll reserve for our predicting. We select the training set with
the ``all-but-last`` Hy function, which produces a new array that contains all but
the last item from ``digits.data``:

```hy
=> (defn all-but-last [lst]
... (list (drop-last 1 lst)))
=> (clf.fit (all-but-last digits.data) (all-but-last digits.target))
SVC(C=100, gamma=0.001)
```

Now you can *predict* new values. In this case, you'll predict using the last
image from ``digits.data``. By predicting, you'll determine the image from the 
training set that best matches the last image.

```hy
=> (clf.predict (cut digits.data -1))
array([8])
```

The corresponding image is:

![](https://scikit-learn.org/stable/_images/sphx_glr_plot_digits_last_image_001.png)

As you can see, it is a challenging task: after all, the images are of poor
resolution. Do you agree with the classifier?

A complete example of this classification problem is available as an
example that you can run and study:
[Recognizing hand-written digits](https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html#sphx-glr-auto-examples-classification-plot-digits-classification-py).


Model Persistence
-----------------

It is possible to save a model in scikit-learn by using Python's built-in
persistence model, [pickle](https://docs.python.org/2/library/pickle.html):

```hy
=> (import [sklearn.svm [SVC]])
=> (import [sklearn [datasets]])
=> (setv clf (SVC :gamma "scale"))
=> (setv x iris.data)
=> (setv y iris.target)
=> (clf.fit x y)
SVC()
    
=> (import pickle)
=> (setv s (pickle.dumps clf))
=> (setv clf2 (pickle.loads s))
=> (clf2.predict (cut x 0 1))
array([0])

=> (first y)
0
```

In the specific case of scikit-learn, it may be more interesting to use
joblib's replacement for pickle (``joblib.dump`` & ``joblib.load``),
which is more efficient on big data but it can only pickle to the disk
and not to a string:

```hy
=> (import [joblib [dump load]])
=> (dump clf "filename.joblib")
['filename.joblib']
```

Later, you can reload the pickled model (possibly in another Hy process)
with:

```hy
=> (setv clf (load "filename.joblib"))
```

##### Note:

``joblib.dump`` and ``joblib.load`` functions also accept file-like object
    instead of filenames. More information on data persistence with Joblib is
    available [here](https://joblib.readthedocs.io/en/latest/persistence.html).

Note that pickle has some security and maintainability issues. Please refer to
section [Model persistence](https://scikit-learn.org/stable/modules/model_persistence.html#model-persistence) for more detailed information about model
persistence with scikit-learn.


Conventions
-----------

scikit-learn estimators follow certain rules to make their behavior more
predictive.  These are described in more detail in the [Glossary of Common Terms and API Elements](https://scikit-learn.org/stable/glossary.html#glossary).

#### Type Casting

Unless otherwise specified, input will be cast to ``float64``:

```hy
=> (import [numpy :as np])
=> (import [sklearn [random_projection]])
=> (setv rng (np.random.RandomState 0))
=> (setv x (np.array (rng.rand 10 2000) :dtype "float32"))
=> (print x.dtype)
float32

=> (setv transformer (random-projection.GaussianRandomProjection))
=> (setv x-new (transformer.fit-transform x))
=> (print x-new.dtype)
float64
```

In this example, ``x`` is ``float32``, which is cast to ``float64`` by
``(transformer.fit-transform x)``.

Regression targets are cast to ``float64`` and classification targets are
maintained:

```hy
=> (import [sklearn [datasets]])
=> (import [sklearn.svm [SVC]])
=> (setv iris (datasets.load-iris))
=> (setv clf (SVC :gamma "scale"))
=> (clf.fit iris.data iris.target)
SVC()
    
=> (list (clf.predict (list (take 3 iris.data))))
[0, 0, 0]

=> (clf.fit iris.data (. iris target-names [iris.target]))
SVC()
    
=> (list (clf.predict (list (take 3 iris.data))))
['setosa', 'setosa', 'setosa']
```

Here, the first ``predict`` returns an integer array, since ``iris.target``
(an integer array) was used in ``fit``. The second ``predict`` returns a string
array, since ``iris.target_names`` was for fitting.

#### Refitting and Updating Parameters

Hyper-parameters of an estimator can be updated after it has been constructed
via the [set_params](https://scikit-learn.org/stable/glossary.html#term-set-params) method. Calling ``fit`` more than
once will overwrite what was learned by any previous ``fit``:

```hy
=> (import [numpy :as np])
=> (import [sklearn.datasets [load_iris]])
=> (import [sklearn.svm [SVC]])
=> (setv x (first (load-iris :return_X_y True)))
=> (setv y (last (load-iris :return_X_y True)))
=> (setv clf (SVC :gamma "auto"))
=> (clf.set-params :kernel "linear")
SVC(gamma='auto', kernel='linear')

=> (clf.fit x y)
SVC(gamma='auto', kernel='linear')

=> (clf.predict (list (take 5 x)))
array([0, 0, 0, 0, 0])

=> (clf.set-params :kernel "rbf" :gamma "scale")
SVC()

=> (clf.fit x y)
SVC()

=> (clf.predict (list (take 5 x)))
array([0, 0, 0, 0, 0])
```

Here, the default kernel ``rbf`` is first changed to ``linear`` via
[clf.set-params](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC.set_params) after the estimator has
been constructed, and changed back to ``rbf`` to refit the estimator and to
make a second prediction.

#### Multiclass vs. Multilabel Fitting

When using [multiclass classifiers](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.multiclass),
the learning and prediction task that is performed is dependent on the format of
the target data fit upon:

```hy
=> (import [sklearn.svm [SVC]])
=> (import [sklearn.multiclass [OneVsRestClassifier]])
=> (import [sklearn.preprocessing [LabelBinarizer]])
=> (setv x '((1 2) (2 4) (4 5) (3 2) (3 1)))
=> (setv y '(0 0 1 1 2))
=> (setv clf (SVC :gamma "scale" :random-state 0))
=> (setv classif (OneVsRestClassifier :estimator clf))
=> (classif.fit x y)
OneVsRestClassifier(estimator=SVC(random_state=0))

=> (classif.fit x y)
OneVsRestClassifier(estimator=SVC(random_state=0))

=> (classif.predict x)
array([0, 0, 1, 1, 2])
```

In the above case, the classifier is fit on a 1d array of multiclass labels and
the ``predict`` function therefore provides corresponding multiclass predictions.
It is also possible to fit upon a 2d array of binary label indicators:

```hy
=> (setv y (LabelBinarizer))
=> (setv y (y.fit_transform '(0 0 1 1 2)))
=> (classif.fit x y)
OneVsRestClassifier(estimator=SVC(random_state=0))

=> (classif.predict x)
array([[1, 0, 0],
       [1, 0, 0],
       [0, 1, 0],
       [0, 0, 0],
       [0, 0, 0]])
```

Here, the classifier is ``fit``  on a 2d binary label representation of ``y``,
using the [LabelBinarizer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelBinarizer.html#sklearn.preprocessing.LabelBinarizer).
In this case ``predict`` returns a 2d array representing the corresponding
multilabel predictions.

Note that the fourth and fifth instances returned all zeroes, indicating that
they matched none of the three labels ``fit`` upon. With multilabel outputs, it
is similarly possible for an instance to be assigned multiple labels:

```hy
=> (import [sklearn.preprocessing [MultiLabelBinarizer]])
=> (setv y (MultiLabelBinarizer))
=> (setv y (y.fit_transform '((0 1) (0 2) (1 3) (0 2 3) (2 4))))
=> (classif.fit x y)
OneVsRestClassifier(estimator=SVC(random_state=0))

=> (classif.predict x)
array([[1, 1, 0, 0, 0],
       [1, 0, 1, 0, 0],
       [0, 1, 0, 1, 0],
       [1, 0, 1, 0, 0],
       [1, 0, 1, 0, 0]])
```

In this case, the classifier is fit upon instances each assigned multiple labels.
The [MultiLabelBinarizer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html#sklearn.preprocessing.MultiLabelBinarizer) is
used to binarize the 2d array of multilabels to ``fit`` upon. As a result,
``predict`` returns a 2d array with multiple predicted labels for each instance.
