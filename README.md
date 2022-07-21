# Graduation-Thesis

Replicated a paper about Deep learning model for supervised learning with tabular data, specifically learn about the TabNet model. Original paper: https://ojs.aaai.org/index.php/AAAI/article/view/16826.

In addition, we do some more experiments in order to compare TabNet with XGBoost and the results are:
- In the most of cases, TabNet and XGBoost give the same results about accuracy, but training time of XGBoost is more faster than TabNet.
- Both models have strong feature selection ability and have instance-wise feature selection.
- Both models have global and local interpretability, but TabNet is more convenient in local interpretability because XGBoost needs additional support libraries.
- In multi-output case, because TabNet is a deep learning model so training time of TabNet is faster than XGBoost if large number of outputs/classes.
