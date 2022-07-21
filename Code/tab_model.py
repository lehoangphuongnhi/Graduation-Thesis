import torch #import thư viện torch
import numpy as np 
from scipy.special import softmax
from Code.utils import PredictDataset, filter_weights
from Code.abstract_model import TabModel
from Code.multiclass_utils import infer_output_dim, check_output_dim
from torch.utils.data import DataLoader


class TabNetClassifier(TabModel):# lớp class dành cho phân lớp
        # loss_fn : callable or None
        # a PyTorch loss function
    def __post_init__(self): #hàm khởi tạo
        super(TabNetClassifier, self).__post_init__()
        self._task = 'classification'#nhiệm vụ khởi tạo
        self._default_loss = torch.nn.functional.cross_entropy#hàm mất mát là hàm cross entropy
        self._default_metric = 'accuracy'#tính theo độ chính xác

    def weight_updater(self, weights):#hàm cập nhật trọng số
        """
        Updates weights dictionary according to target_mapper.
        Parameters
        ----------
        weights : bool or dict
            Given weights for balancing training.
        Returns
        -------
        bool or dict
            Same bool if weights are bool, updated dict otherwise.
        """
        if isinstance(weights, int):#kiểm tra wieght là số hay là dict
            return weights #phải là số trả về số
        elif isinstance(weights, dict):#kiểm tra weight là số hay là dict 
            return {self.target_mapper[key]: value for key, value in weights.items()} #trả về giá trị weight
        else:
            return weights#không phải hai cái kia thì trả hết

    def prepare_target(self, y):#chuẩn bị mục tiêu trước khi đào tạo
        # Parameters:y (a :tensor: torch.Tensor) – Target matrix.

        # Returns:    Converted target matrix.

        # Return type:    torch.Tensor
        return np.vectorize(self.target_mapper.get)(y)#là một hàm của np thực hiện self.target_mapper.get, với đầu vào là một vetor

    def compute_loss(self, y_pred, y_true):#tính toán đầu ra dựa theo đầu ra của network và target
        # Parameters
        # y_pred (list of tensors) – Output of network

        # y_true (LongTensor) – Targets label encoded

        # Returns loss – output of loss function(s)

        # Return type torch.Tensor
        return self.loss_fn(y_pred, y_true.long())


    def update_fit_params(# set giá trị các thuộc tính 
    # X_train (np.ndarray) – Train set

    # y_train (np.array) – Train targets

    # eval_set (list of tuple) – List of eval tuple set (X, y).

    # weights (bool or dictionnary) – 0 for no balancing 1 for automated balancing
        self,
        X_train,
        y_train,
        eval_set,
        weights,
    ):
        out_dim, train_labels = infer_output_dim(y_train)#suy số label có trong đầu ra từ y_train
        for X, y in eval_set:
            check_output_dim(train_labels, y)#kiểm tra kiểu dữ liệu trong y có cùng loại không, kiểm tra những label trong y đều thuộc train_labels hay không
        self.out_dim = out_dim
        self._default_metric = ('auc' if self.out_dim == 2 else 'accuracy')#nếu phân lớp nhị phân thì dùng AUC, còn không dùng accuracy
        self.classes_ = train_labels #gán giá trị class
        self.target_mapper = {#tạo chỉ số, và giá trị class
            class_label: index for index, class_label in enumerate(self.classes_)
        }
        self.preds_mapper = {#y cái trên khác mỗi index là string
            str(index): class_label for index, class_label in enumerate(self.classes_)
        }
        self.updated_weights = self.weight_updater(weights)

    def stack_batches(self, list_y_true, list_y_score):
        y_true = np.hstack(list_y_true) #sắp xếp mảng theo chiều ngang
        y_score = np.vstack(list_y_score) #sắp xếp mảng theo chiều dọc
        y_score = softmax(y_score, axis=1) #chuyển qua hàm softmax
        return y_true, y_score#trả ra giá trị y true và y score

    def predict_func(self, outputs):
        outputs = np.argmax(outputs, axis=1)#tìm giá trị max trong matrix theo cột
        return np.vectorize(self.preds_mapper.get)(outputs.astype(str))

    def predict_proba(self, X):#dự đoán trên lô
        """
        Make predictions for classification on a batch (valid)
        Parameters
        ----------
        X : a :tensor: `torch.Tensor`
            Input data
        Returns
        -------
        res : np.ndarray
        """
        self.network.eval()

        dataloader = DataLoader(#một chình tải dữ liệu với batch size tùy ý, như data mà có batch size
            PredictDataset(X),#định dạng cho mảng numpy
            batch_size=self.batch_size,
            shuffle=False,
        )

        results = []
        for batch_nb, data in enumerate(dataloader):# theo từng lô và data từng lô
            data = data.to(self.device).float()
            output, M_loss = self.network(data)     #trong self.network lưu các model huấn luyện
            predictions = torch.nn.Softmax(dim=1)(output).cpu().detach().numpy()  #vứt qua hàm softmax
            results.append(predictions)
        res = np.vstack(results)
        return res


class TabNetRegressor(TabModel):
    def __post_init__(self):
        super(TabNetRegressor, self).__post_init__()#khởi tạo tabnetRegressor
        self._task = 'regression'
        self._default_loss = torch.nn.functional.mse_loss
        self._default_metric = 'mse'

    def prepare_target(self, y):#chuẩn bị target
        return y

    def compute_loss(self, y_pred, y_true):#hàm mất mát
        return self.loss_fn(y_pred, y_true)

    def update_fit_params(# update trọng số
        self,
        X_train,
        y_train,
        eval_set,
        weights
    ):
        if len(y_train.shape) != 2:#giả sử y_train có shape  ví dụ (124,) là không hợp lệ, phải là (124, 2)
            msg = "Targets should be 2D : (n_samples, n_regression) " + \
                  f"but y_train.shape={y_train.shape} given.\n" + \
                  "Use reshape(-1, 1) for single regression."
            raise ValueError(msg)
        self.out_dim = y_train.shape[1]#số n_regression
        self.preds_mapper = None

        self.updated_weights = weights
        filter_weights(self.updated_weights) #kiểm tra trọng số đúng format regression 

    def predict_func(self, outputs):
        return outputs

    def stack_batches(self, list_y_true, list_y_score):#kiểm tra 
        y_true = np.vstack(list_y_true)#ghép thành một danh sách các dòng
        y_score = np.vstack(list_y_score)
        return y_true, y_score