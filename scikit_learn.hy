(import [sklearn [datasets random_projection]]
  [sklearn.svm [SVC]]
  [sklearn.datasets [load_iris]]
  [sklearn.multiclass [OneVsRestClassifier]]
  [sklearn.preprocessing [LabelBinarizer MultiLabelBinarizer]]
  pickle
  [joblib [dump load]]
  [numpy :as np])

(defn all-but-last [lst]
  (list (drop-last 1 lst)))

(defn learning-and-predicting []
  (setv iris (datasets.load-iris))
  (setv digits (datasets.load-digits))
  (setv clf (SVC :gamma 0.001 :C 100))
  (clf.fit (all-but-last digits.data) (all-but-last digits.target))
  (clf.predict (cut digits.data -1)))

(defn model-persistence []
  (setv iris (datasets.load-iris))
  (setv clf (SVC :gamma "scale"))
  (setv x iris.data)
  (setv y iris.target)
  (clf.fit x y)
  (setv s (pickle.dumps clf))
  (setv clf2 (pickle.loads s))
  (clf2.predict (cut x 0 1)))

(defn model-persistence-to-file []
  (setv iris (datasets.load-iris))
  (setv clf (SVC :gamma "scale"))
  (setv x iris.data)
  (setv y iris.target)
  (clf.fit x y)
  (dump clf "filename.joblib"))

(defn model-persistence-from-file []
  (model-persistence-to-file)
  (setv iris (datasets.load-iris))
  (setv x iris.data)
  (setv clf (load "filename.joblib"))
  (clf.predict (cut x 0 1)))

(defn type-casting-32 []
  (setv rng (np.random.RandomState 0))
  (setv x (np.array (rng.rand 10 2000) :dtype "float32"))
  (return x))

(defn type-casting-64 []
  (setv transformer (random-projection.GaussianRandomProjection))
  (setv x-new (transformer.fit-transform (type-casting-32)))
  (return x-new))

(defn type-casting-more1 []
  (setv iris (datasets.load-iris))
  (setv clf (SVC :gamma "scale"))
  (clf.fit iris.data iris.target)
  (list (clf.predict (list (take 3 iris.data)))))

(defn type-casting-more2 []
  (setv iris (datasets.load-iris))
  (setv clf (SVC :gamma "scale"))
  (clf.fit iris.data (. iris target-names [iris.target]))
  (list (clf.predict (list (take 3 iris.data)))))

(defn refitting-and-updating-parameters1 []
  (setv x (first (load-iris :return_X_y True)))
  (setv y (last (load-iris :return_X_y True)))
  (setv clf (SVC :gamma "auto"))
  (clf.set-params :kernel "linear")
  (clf.fit x y)
  (clf.predict (list (take 5 x))))

(defn refitting-and-updating-parameters2 []
  (setv x (first (load-iris :return_X_y True)))
  (setv y (last (load-iris :return_X_y True)))
  (setv clf (SVC :gamma "auto"))
  (clf.set-params :kernel "rbf" :gamma "scale")
  (clf.fit x y)
  (clf.predict (list (take 5 x))))

(defn multiclass-vs-multilabel-fitting1 []
  (setv x '((1 2) (2 4) (4 5) (3 2) (3 1)))
  (setv y '(0 0 1 1 2))
  (setv clf (SVC :gamma "scale" :random-state 0))
  (setv classif (OneVsRestClassifier :estimator clf))
  (classif.fit x y)
  (classif.predict x))

(defn multiclass-vs-multilabel-fitting2 []
  (setv x '((1 2) (2 4) (4 5) (3 2) (3 1)))
  (setv y (LabelBinarizer))
  (setv y (y.fit_transform '(0 0 1 1 2)))
  (setv clf (SVC :gamma "scale" :random-state 0))
  (setv classif (OneVsRestClassifier :estimator clf))
  (classif.fit x y)
  (classif.predict x))

(defn multiclass-vs-multilabel-fitting3 []
  (setv x '((1 2) (2 4) (4 5) (3 2) (3 1)))
  (setv y (MultiLabelBinarizer))
  (setv y (y.fit_transform '((0 1) (0 2) (1 3) (0 2 3) (2 4))))
  (setv clf (SVC :gamma "scale" :random-state 0))
  (setv classif (OneVsRestClassifier :estimator clf))
  (classif.fit x y)
  (classif.predict x))

;; https://github.com/hylang/hy/issues/1786
(eval-and-compile
  (import shutil multiprocessing)
  (setv python_path (.which shutil "python3.7"))
  (.set_executable multiprocessing python_path))
