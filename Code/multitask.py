import torch
import numpy as np
from scipy.special import softmax
from Code.utils import PredictDataset, filter_weights
from Code.abstract_model import TabModel
from Code.multiclass_utils import infer_multitask_output, check_output_dim
from torch.utils.data import DataLoader


class TabNetMultiTaskClassifier(TabModel):
    def __post_init__(self):
        super(TabNetMultiTaskClassifier, self).__post_init__()
        self._task = 'classification'
        self._default_loss = torch.nn.functional.cross_entropy
        self._default_metric = 'logloss'

    def prepare_target(self, y):
        y_mapped = y.copy()
        for task_idx in range(y.shape[1]):
            task_mapper = self.target_mapper[task_idx] #mã hóa các label sang số thứ tự (ví dụ: a:0, b:1, c:2,...)
            y_mapped[:, task_idx] = np.vectorize(task_mapper.get)(y[:, task_idx]) #áp dụng các label đã được mã hóa 
                                                                                #vào y
        return y_mapped

    def compute_loss(self, y_pred, y_true): #Tính loss
        """
        Computes the loss according to network output and targets
        Parameters
        ----------
        y_pred : list of tensors
            Output of network
        y_true : LongTensor
            Targets label encoded
        Returns
        -------
        loss : torch.Tensor
            output of loss function(s)
        """
        loss = 0
        y_true = y_true.long()
        if isinstance(self.loss_fn, list): #Nếu mỗi task sử dụng 1 hàm loss khác nhau
            # if you specify a different loss for each task
            for task_loss, task_output, task_id in zip(
                self.loss_fn, y_pred, range(len(self.loss_fn))
            ):
                loss += task_loss(task_output, y_true[:, task_id])
        else: #Nếu tất cả các task sử dụng chung 1 hàm loss
            # same loss function is applied to all tasks
            for task_id, task_output in enumerate(y_pred):
                loss += self.loss_fn(task_output, y_true[:, task_id])

        loss /= len(y_pred)
        return loss

    def stack_batches(self, list_y_true, list_y_score):
        y_true = np.vstack(list_y_true)
        y_score = []
        for i in range(len(self.output_dim)):
            score = np.vstack([x[i] for x in list_y_score]) #tính score của mỗi task
            score = softmax(score, axis=1) #áp dụng softmax sao cho mỗi task có tổng xác suất của các nhãn = 1
            y_score.append(score) 
        return y_true, y_score

    def update_fit_params(self, X_train, y_train, eval_set, weights):
        output_dim, train_labels = infer_multitask_output(y_train) #lấy số chiều của đầu ra (số nhãn của mỗi task)
                                                                   # danh sách các mảng của label của mỗi task
        for _, y in eval_set:
            for task_idx in range(y.shape[1]):
                check_output_dim(train_labels[task_idx], y[:, task_idx]) #Kiểm tra xem label đã đúng chưa
        self.output_dim = output_dim
        self.classes_ = train_labels
        self.target_mapper = [
            {class_label: index for index, class_label in enumerate(classes)}
            for classes in self.classes_
        ] #mã hóa các label sang số thứ tự (ví dụ: a:0, b:1, c:2,...)
        self.preds_mapper = [
            {str(index): str(class_label) for index, class_label in enumerate(classes)}
            for classes in self.classes_
        ] #mã hóa các label dự đoán sang dạng chuỗi
        self.updated_weights = weights #cập nhật các trọng số mô hình
        filter_weights(self.updated_weights)

    def predict(self, X): #dự đoán trên 1 batch, trả về lớp có thể xảy ra nhất
        """
        Make predictions on a batch (valid)
        Parameters
        ----------
        X : a :tensor: `torch.Tensor`
            Input data
        Returns
        -------
        results : np.array
            Predictions of the most probable class
        """
        self.network.eval()
        dataloader = DataLoader(
            PredictDataset(X),
            batch_size=self.batch_size,
            shuffle=False,
        )

        results = {}
        for data in dataloader:
            data = data.to(self.device).float()
            output, _ = self.network(data)
            predictions = [
                torch.argmax(torch.nn.Softmax(dim=1)(task_output), dim=1) #dùng argmax để có được lớp có thể xảy ra nhất
                .cpu()
                .detach()
                .numpy()
                .reshape(-1)
                for task_output in output
            ]

            for task_idx in range(len(self.output_dim)): #đưa các dự đoán vào dict với key:task_id, value: prediction
                results[task_idx] = results.get(task_idx, []) + [predictions[task_idx]]
        # stack all task individually
        results = [np.hstack(task_res) for task_res in results.values()]
        # map all task individually
        results = [
            np.vectorize(self.preds_mapper[task_idx].get)(task_res.astype(str))
            for task_idx, task_res in enumerate(results)
        ] #mã hóa các dự đoán sang string
        return results

    def predict_proba(self, X): #đưa ra xác suất xảy ra của từng lớp trong từng task
        """
        Make predictions for classification on a batch (valid)
        Parameters
        ----------
        X : a :tensor: `torch.Tensor`
            Input data
        Returns
        -------
        res : list of np.ndarray
        """
        self.network.eval()

        dataloader = DataLoader(
            PredictDataset(X),
            batch_size=self.batch_size,
            shuffle=False,
        )

        results = {}
        for data in dataloader:
            data = data.to(self.device).float()
            output, _ = self.network(data)
            predictions = [
                torch.nn.Softmax(dim=1)(task_output).cpu().detach().numpy()
                for task_output in output
            ]
            for task_idx in range(len(self.output_dim)):
                results[task_idx] = results.get(task_idx, []) + [predictions[task_idx]]
        res = [np.vstack(task_res) for task_res in results.values()]
        return res