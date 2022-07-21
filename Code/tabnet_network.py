import torch
from torch.nn import Linear, BatchNorm1d, ReLU
import numpy as np
from torch.autograd import Function



def initialize_non_glu(module, inp_dim, out_dim): #Khởi tạo Glorot
    gain_value = np.sqrt((inp_dim + out_dim) / np.sqrt(4 * inp_dim))
    torch.nn.init.xavier_normal_(module.weight, gain=gain_value)
    return


def initialize_glu(module, inp_dim, out_dim): #Khởi tạo Glorot
    gain_value = np.sqrt((inp_dim + out_dim) / np.sqrt(inp_dim))
    torch.nn.init.xavier_normal_(module.weight, gain=gain_value)
    return

def make_ix_like(input, dim = 0):
    d = input.size(dim)
    rho = torch.arange(1, d + 1, device = input.device, dtype = input.dtype)
    view = [1] * input.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)

class SparsemaxFunction(Function): #sparsemax activation
    @staticmethod
    def forward(ctx, input, dim = -1):
        ctx.dim = dim
        max_val, _ = input.max (dim = dim, keepdim = True)
        input -= max_val
        tau, supp_size = SparsemaxFunction.threshold_and_support(input, dim = dim)
        output = torch.clamp(input - tau, min = 0)
        ctx.save_for_backward(supp_size, output)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        supp_size, output = ctx.saved_tensors
        dim = ctx.dim
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0

        v_hat = grad_input.sum(dim = dim) / supp_size.to(output.dtype).squeeze()
        v_hat = v_hat.unsqueeze(dim)
        grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)
        return grad_input, None

    @staticmethod
    def threshold_and_support(input, dim = -1):
        input_srt, _ = torch.sort(input, descending = True, dim = dim)
        input_cumsum = input_srt.cumsum(dim) - 1
        rhos = make_ix_like(input, dim)
        support = rhos * input_srt > input_cumsum

        support_size = support.sum(dim = dim).unsqueeze(dim)
        tau = input_cumsum.gather(dim, support_size - 1)
        tau /= support_size.to(input.dtype)
        return tau, support_size

sparsemax = SparsemaxFunction.apply


class Sparsemax(torch.nn.Module):

    def __init__(self, dim=-1):
        self.dim = dim
        super(Sparsemax, self).__init__()

    def forward(self, input):
        return sparsemax(input, self.dim)

class GBN(torch.nn.Module):
    """
    Ghost Batch Normalization
    https://arxiv.org/abs/1705.08741
    """

    def __init__(self, inp_dim, vbs=128, momentum=0.01):
        super(GBN, self).__init__()

        self.inp_dim = inp_dim
        self.vbs = vbs
        self.bn = BatchNorm1d(self.inp_dim, momentum=momentum)

    def forward(self, x):
        if x.shape[0] <= self.vbs: #Kích thước lô ảo lớn hơn lô đầu vào nên không thỏa điều kiện để sử dụng GBN
            return self.bn(x)
        else:
            chunks = x.chunk(int(np.ceil(x.shape[0] / self.vbs)), 0) #Chia lô đầu vào thành các lô nhỏ (lô ảo)
            res = [self.bn(y) for y in chunks] #áp dụng batch normalization cho từng lô ảo

            return torch.cat(res, dim = 0)

class GLU_Layer(torch.nn.Module):
    def __init__(
        self, inp_dim, out_dim, fc=None, vbs=128, momentum=0.02
    ):
        super(GLU_Layer, self).__init__()

        self.out_dim = out_dim
        if fc:
            self.fc = fc
        else:
            self.fc = Linear(inp_dim, 2 * out_dim, bias=False)
        initialize_glu(self.fc, inp_dim, 2 * out_dim)

        self.bn = GBN(
            2 * out_dim, vbs=vbs, momentum=momentum
        )

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        out = torch.mul(x[:, : self.out_dim], torch.sigmoid(x[:, self.out_dim :]))
        return out

class GLU_Block(torch.nn.Module):
    """
    Independent GLU block, specific to each step
    """

    def __init__(
        self,
        inp_dim,
        out_dim,
        n_glu=2,
        first=False,
        shared_layers=None,
        vbs=128,
        momentum=0.02,
    ):
        super(GLU_Block, self).__init__()
        self.first = first
        self.shared_layers = shared_layers
        self.n_glu = n_glu
        self.glu_layers = torch.nn.ModuleList()

        params = {"vbs": vbs, "momentum": momentum}

        fc = shared_layers[0] if shared_layers else None
        self.glu_layers.append(GLU_Layer(inp_dim, out_dim, fc=fc, **params))
        for glu_id in range(1, self.n_glu):
            fc = shared_layers[glu_id] if shared_layers else None
            self.glu_layers.append(GLU_Layer(out_dim, out_dim, fc=fc, **params))

    def forward(self, x):
        scale = torch.sqrt(torch.FloatTensor([0.5]).to(x.device))
        if self.first:  # the first layer of the block has no scale multiplication
            x = self.glu_layers[0](x)
            layers_left = range(1, self.n_glu)
        else:
            layers_left = range(self.n_glu)

        for glu_id in layers_left:
            x = torch.add(x, self.glu_layers[glu_id](x))
            x = x * scale
        return x

class AttentiveTransformer(torch.nn.Module):
    def __init__(
        self,
        inp_dim,
        out_dim,
        vbs=128,
        momentum=0.02,
        #mask_type="sparsemax",
    ):
        """
        Initialize an attention transformer.

        Parameters
        ----------
        inp_dim : int
            Input size
        out_dim : int
            Output_size
        vbs : int
            Batch size for Ghost Batch Normalization
        momentum : float
            Float value between 0 and 1 which will be used for momentum in batch norm
        mask_type : str
            Either "sparsemax" or "entmax" : this is the masking function to use
        """
        super(AttentiveTransformer, self).__init__()
        self.fc = Linear(inp_dim, out_dim, bias=False)
        initialize_non_glu(self.fc, inp_dim, out_dim)
        self.bn = GBN(
            out_dim, vbs=vbs, momentum=momentum
        )

        self.selector = Sparsemax(dim=-1)

    def forward(self, priors, processed_feat):
        x = self.fc(processed_feat) #Tầng Fully connected 
        x = self.bn(x) #Theo sau bởi Ghost batch normalization
        x = torch.mul(x, priors) #Mask là giá trị sparsemax của a và priors (mức độ 1 đặc trưng cụ thể đã được
                                                                                        #sử dụng trước đây)
        x = self.selector(x)
        return x

class FeatTransformer(torch.nn.Module):
    def __init__(
        self,
        inp_dim,
        out_dim,
        shared_layers,
        n_glu_independent,
        vbs=128,
        momentum=0.02,
    ):
        super(FeatTransformer, self).__init__()
        """
        Initialize a feature transformer.

        Parameters
        ----------
        inp_dim : int
            Input size
        out_dim : int
            Output_size
        shared_layers : torch.nn.ModuleList
            The shared block that should be common to every step
        n_glu_independent : int
            Number of independent GLU layers
        vbs : int
            Batch size for Ghost Batch Normalization within GLU block(s)
        momentum : float
            Float value between 0 and 1 which will be used for momentum in batch norm
        """

        params = {
            "n_glu": n_glu_independent,
            "vbs": vbs,
            "momentum": momentum,
        }

        if shared_layers is None:
            # no shared layers
            self.shared = torch.nn.Identity()
            is_first = True
        else:
            self.shared = GLU_Block(
                inp_dim,
                out_dim,
                first=True,
                shared_layers=shared_layers,
                n_glu=len(shared_layers),
                vbs=vbs,
                momentum=momentum,
            )
            is_first = False

        if n_glu_independent == 0:
            # no independent layers
            self.specifics = torch.nn.Identity()
        else:
            spec_inp_dim = inp_dim if is_first else out_dim
            self.specifics = GLU_Block(
                spec_inp_dim, out_dim, first=is_first, **params
            )

    def forward(self, x):
        x = self.shared(x)
        x = self.specifics(x)
        return x

class TabNetEncoder(torch.nn.Module):
    def __init__(
        self,
        inp_dim,
        out_dim,
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        n_ind=2,
        n_shared=2,
        epsilon=1e-15,
        vbs=128,
        momentum=0.02,
    ):
        """
        Defines main part of the TabNet network without the embedding layers.

        Parameters
        ----------
        inp_dim : int
            Number of features
        out_dim : int or list of int for multi task classification
            Dimension of network output
            examples : one for regression, 2 for binary classification etc...
        n_d : int
            Dimension of the prediction  layer (usually between 4 and 64)
        n_a : int
            Dimension of the attention  layer (usually between 4 and 64)
        n_steps : int
            Number of successive steps in the network (usually between 3 and 10)
        gamma : float
            Float above 1, scaling factor for attention updates (usually between 1.0 to 2.0)
        n_ind : int
            Number of independent GLU layer in each GLU block (default 2)
        n_shared : int
            Number of independent GLU layer in each GLU block (default 2)
        epsilon : float
            Avoid log(0), this should be kept very low
        vbs : int
            Batch size for Ghost Batch Normalization
        momentum : float
            Float value between 0 and 1 which will be used for momentum in all batch norm
        mask_type : str
            Either "sparsemax" or "entmax" : this is the masking function to use
        """
        super(TabNetEncoder, self).__init__()
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.is_multi_task = isinstance(out_dim, list)
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_ind = n_ind
        self.n_shared = n_shared
        self.vbs = vbs
        self.initial_bn = BatchNorm1d(self.inp_dim, momentum=0.01)

        if self.n_shared > 0: #Khởi tạo shared_layers
            shared_feat_transform = torch.nn.ModuleList()
            for i in range(self.n_shared):
                if i == 0:
                    shared_feat_transform.append(
                        Linear(self.inp_dim, 2 * (n_d + n_a), bias=False)
                    )
                else:
                    shared_feat_transform.append(
                        Linear(n_d + n_a, 2 * (n_d + n_a), bias=False)
                    )

        else:
            shared_feat_transform = None

        self.initial_splitter = FeatTransformer(
            self.inp_dim,
            n_d + n_a,
            shared_feat_transform,
            n_glu_independent=self.n_ind,
            vbs=self.vbs,
            momentum=momentum,
        )

        self.feat_transformers = torch.nn.ModuleList()
        self.att_transformers = torch.nn.ModuleList()

        for step in range(n_steps):
            transformer = FeatTransformer(
                self.inp_dim,
                n_d + n_a,
                shared_feat_transform,
                n_glu_independent=self.n_ind,
                vbs=self.vbs,
                momentum=momentum,
            )
            attention = AttentiveTransformer(
                n_a,
                self.inp_dim,
                vbs=self.vbs,
                momentum=momentum,
            )
            self.feat_transformers.append(transformer)
            self.att_transformers.append(attention)

    def forward(self, x, prior=None):
        x = self.initial_bn(x) #Theo cấu trúc thì dữ liệu đầu vào sẽ qua batch normalization để chuẩn hóa

        if prior is None:
            prior = torch.ones(x.shape).to(x.device)

        M_loss = 0
        att = self.initial_splitter(x)[:, self.n_d :] #Sau đó sẽ qua Feature transformer

        steps_output = []
        for step in range(self.n_steps):
            M = self.att_transformers[step](prior, att)
            M_loss += torch.mean(
                torch.sum(torch.mul(M, torch.log(M + self.epsilon)), dim=1)
            )
            # update prior
            prior = torch.mul(self.gamma - M, prior)
            # output
            masked_x = torch.mul(M, x)
            out = self.feat_transformers[step](masked_x)
            d = ReLU()(out[:, : self.n_d]) #split feature từ feature transformer
            steps_output.append(d)
            # update attention
            att = out[:, self.n_d :]

        M_loss /= self.n_steps
        return steps_output, M_loss

    def forward_masks(self, x): #đưa ra giải thích
        x = self.initial_bn(x)

        prior = torch.ones(x.shape).to(x.device)
        M_explain = torch.zeros(x.shape).to(x.device)
        att = self.initial_splitter(x)[:, self.n_d :]
        masks = {}

        for step in range(self.n_steps):
            M = self.att_transformers[step](prior, att)
            masks[step] = M
            # update prior
            prior = torch.mul(self.gamma - M, prior)
            # output
            masked_x = torch.mul(M, x)
            out = self.feat_transformers[step](masked_x)
            d = ReLU()(out[:, : self.n_d])
            # explain
            step_importance = torch.sum(d, dim=1)
            M_explain += torch.mul(M, step_importance.unsqueeze(dim=1))
            # update attention
            att = out[:, self.n_d :]

        return M_explain, masks

class EmbeddingGenerator(torch.nn.Module):
    """
    Classical embeddings generator
    """

    def __init__(self, inp_dim, cat_dims, cat_idxs, cat_emb_dim): #hàm khởi tạo
        """This is an embedding module for an entire set of features

        Parameters
        ----------
        inp_dim : int #số đặc trưng được đưa vào 
            Number of features coming as input (number of columns)
        cat_dims : list of int #số categorical có trong bảng#  Number of categories in each categorical column, số lượng categorical có trong một đặc trưng
            Number of modalities for each categorial features #số lượng phương thức cho mỗi categorical
            If the list is empty, no embeddings will be done
        cat_idxs : list of int
            Positional index for each categorical features in inputs #chỉ mục của tính năng, vị trí của tính năng
        cat_emb_dim : int or list of int #số lượng một feature được chia tành bao nhiều sau emdeding
            Embedding dimension for each categorical features
            If int, the same embedding dimension will be used for all categorical features
        """
        super(EmbeddingGenerator, self).__init__()
        if cat_dims == [] and cat_idxs == []: #nếu mà rỗng thì thôi ha, không embedding là không có, là không có đặc trưng categorical
            self.skip_embedding = True
            self.post_embed_dim = inp_dim
            return
        elif (cat_dims == []) ^ (cat_idxs == []): #giá trị cat_dims có thì cat_indexs phải có mang trong mình giá trị
            if cat_dims == []:
                msg = "If cat_idxs is non-empty, cat_dims must be defined as a list of same length."
            else:
                msg = "If cat_dims is non-empty, cat_idxs must be defined as a list of same length."
            raise ValueError(msg)
        elif len(cat_dims) != len(cat_idxs):
            msg = "The lists cat_dims and cat_idxs must have the same length."
            raise ValueError(msg)

        self.skip_embedding = False
        if isinstance(cat_emb_dim, int): # kiểm trả cat_emb_dim là số hay là list danh sách, nếu là số 
            self.cat_emb_dims = [cat_emb_dim] * len(cat_idxs) #thì tạo dánh sách cat_em_dims
        else:
            self.cat_emb_dims = cat_emb_dim

        #kiểm tra số lượng cat_em_dím và cat_dims có bằng nhau không, đây là số lượng categorical và số lượng tham số chuyển từ categorical
        if len(self.cat_emb_dims) != len(cat_dims):
            msg = f"""cat_emb_dim and cat_dims must be lists of same length, got {len(self.cat_emb_dims)}
                      and {len(cat_dims)}"""
            raise ValueError(msg)
        self.post_embed_dim = int(#số đặc trưng đầu vào
            inp_dim + np.sum(self.cat_emb_dims) - len(self.cat_emb_dims) #lấy số đặc trưng ban đầu trừ đi số categorical + số đặc trưng được tạo từ 
        )

        self.embeddings = torch.nn.ModuleList() #tạo ra một dánh tensor rỗng

        # Sort dims by cat_idx
        sorted_idxs = np.argsort(cat_idxs) #sắp xếp giá trị từ bé đến lớn trong cat_inds, sorted_indx là indx của cat_indx
        cat_dims = [cat_dims[i] for i in sorted_idxs] # giá trị cat_dims sẽ được lưu theo từ đặc trưng đầu tiên đến đặc trưng cuối cùng, là theo thứ tự, chứ ban đầu nó lộn xộn
        self.cat_emb_dims = [self.cat_emb_dims[i] for i in sorted_idxs] # giá trị cat_em_dim cũng sẽ được sắp xếp

        for cat_dim, emb_dim in zip(cat_dims, self.cat_emb_dims): # chơi theo cặp biết ha
            self.embeddings.append(torch.nn.Embedding(cat_dim, emb_dim)) # thêm tensor embedding có sẵn trong pytorch

        # record continuous indices
        self.continuous_idx = torch.ones(inp_dim, dtype=torch.bool) # input_dim là số đặc trưng ban đầu không có embedding, tất cả được gán 1
        self.continuous_idx[cat_idxs] = 0 # các tính năng categorical ban đầu mang giá trị 0

    def forward(self, x):
        """
        Apply embeddings to inputs
        Inputs should be (batch_size, inp_dim)
        Outputs will be of size (batch_size, self.post_embed_dim)
        """
        if self.skip_embedding: #nếu skip_embedding mà true thì ko thực hiện đồng nghĩa là categorical nào cả
            # no embeddings required
            return x

        cols = [] #tạo một cột rỗng
        cat_feat_counter = 0 # index của self.emdeddings
        for feat_init_idx, is_continuous in enumerate(self.continuous_idx): #gép cặp từ 1..n cho feat_init_idx, is_continous là cái giá trị 1, 0 được gán ở trên
            # Enumerate through continuous idx boolean mask to apply embeddings
            if is_continuous: #nếu bằng 1 là các giá trị trị feature số, ban đầu
                cols.append(x[:, feat_init_idx].float().view(-1, 1)) # vẫn giữ nguyên giá trị
            else:
                cols.append(
                    self.embeddings[cat_feat_counter](x[:, feat_init_idx].long()) #thực hiện emdedding so với các đặng trưng categorical
                )
                cat_feat_counter += 1 #thêm một để chọn giá trị tiếp theo trong self.embedding
        # concat
        post_embeddings = torch.cat(cols, dim=1) #ghép kết quả của thực hiện hai cái trên
        return post_embeddings

class TabNetNoEmbeddings(torch.nn.Module):
    def __init__(
        self,
        inp_dim,
        out_dim,
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        n_ind=2,
        n_shared=2,
        epsilon=1e-15,
        vbs=128,
        momentum=0.02,
        #mask_type="sparsemax",
    ):
        """
        Defines main part of the TabNet network without the embedding layers.

        Parameters
        ----------
        inp_dim : int
            Number of features
        out_dim : int or list of int for multi task classification
            Dimension of network output
            examples : one for regression, 2 for binary classification etc...
        n_d : int
            Dimension of the prediction  layer (usually between 4 and 64)
        n_a : int
            Dimension of the attention  layer (usually between 4 and 64)
        n_steps : int
            Number of successive steps in the network (usually between 3 and 10)
        gamma : float
            Float above 1, scaling factor for attention updates (usually between 1.0 to 2.0)
        n_ind : int
            Number of independent GLU layer in each GLU block (default 2)
        n_shared : int
            Number of independent GLU layer in each GLU block (default 2)
        epsilon : float
            Avoid log(0), this should be kept very low
        vbs : int
            Batch size for Ghost Batch Normalization
        momentum : float
            Float value between 0 and 1 which will be used for momentum in all batch norm
        mask_type : str
            Either "sparsemax" or "entmax" : this is the masking function to use
        """
        super(TabNetNoEmbeddings, self).__init__()
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.is_multi_task = isinstance(out_dim, list)
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_ind = n_ind
        self.n_shared = n_shared
        self.vbs = vbs
        self.initial_bn = BatchNorm1d(self.inp_dim, momentum=0.01)

        self.encoder = TabNetEncoder(
            inp_dim=inp_dim,
            out_dim=out_dim,
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            n_ind=n_ind,
            n_shared=n_shared,
            epsilon=epsilon,
            vbs=vbs,
            momentum=momentum,
        )

        if self.is_multi_task:
            self.multi_task_mappings = torch.nn.ModuleList()
            for task_dim in out_dim:
                task_mapping = Linear(n_d, task_dim, bias=False)
                initialize_non_glu(task_mapping, n_d, task_dim)
                self.multi_task_mappings.append(task_mapping)
        else:
            self.final_mapping = Linear(n_d, out_dim, bias=False)
            initialize_non_glu(self.final_mapping, n_d, out_dim)

    def forward(self, x):
        res = 0
        steps_output, M_loss = self.encoder(x)
        res = torch.sum(torch.stack(steps_output, dim=0), dim=0)

        if self.is_multi_task:
            # Result will be in list format
            out = []
            for task_mapping in self.multi_task_mappings:
                out.append(task_mapping(res))
        else:
            out = self.final_mapping(res) #Qua một tầng FC để có được output cuối cùng 
        return out, M_loss

    def forward_masks(self, x):
        return self.encoder.forward_masks(x)


class TabNet(torch.nn.Module):
    def __init__(
        self,
        inp_dim,
        out_dim,
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        cat_idxs=[],
        cat_dims=[],
        cat_emb_dim=1,
        n_ind=2,
        n_shared=2,
        epsilon=1e-15,
        vbs=128,
        momentum=0.02,
    ):
        """
        Defines TabNet network

        Parameters
        ----------
        inp_dim : int
            Initial number of features
        out_dim : int
            Dimension of network output
            examples : one for regression, 2 for binary classification etc...
        n_d : int
            Dimension of the prediction  layer (usually between 4 and 64)
        n_a : int
            Dimension of the attention  layer (usually between 4 and 64)
        n_steps : int
            Number of successive steps in the network (usually between 3 and 10)
        gamma : float
            Float above 1, scaling factor for attention updates (usually between 1.0 to 2.0)
        cat_idxs : list of int
            Index of each categorical column in the dataset
        cat_dims : list of int
            Number of categories in each categorical column
        cat_emb_dim : int or list of int
            Size of the embedding of categorical features
            if int, all categorical features will have same embedding size
            if list of int, every corresponding feature will have specific size
        n_ind : int
            Number of independent GLU layer in each GLU block (default 2)
        n_shared : int
            Number of independent GLU layer in each GLU block (default 2)
        epsilon : float
            Avoid log(0), this should be kept very low
        vbs : int
            Batch size for Ghost Batch Normalization
        momentum : float
            Float value between 0 and 1 which will be used for momentum in all batch norm
        mask_type : str
            Either "sparsemax" or "entmax" : this is the masking function to use
        """
        super(TabNet, self).__init__()
        self.cat_idxs = cat_idxs or []
        self.cat_dims = cat_dims or []
        self.cat_emb_dim = cat_emb_dim

        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_ind = n_ind
        self.n_shared = n_shared

        if self.n_steps <= 0: #kiểm tra điều kiện nếu n_step<0 thì coi ha
            raise ValueError("n_steps should be a positive integer.")
        if self.n_ind == 0 and self.n_shared == 0:
            raise ValueError("n_shared and n_ind can't be both zero.")

        self.vbs = vbs
        self.embedder = EmbeddingGenerator(inp_dim, cat_dims, cat_idxs, cat_emb_dim)
        self.post_embed_dim = self.embedder.post_embed_dim #số đặc trưng sau khi đã emdedding
        self.tabnet = TabNetNoEmbeddings( #thực hiện noemdedding, tuy tên như thế chứ được làm hết emdding rồi
            self.post_embed_dim,
            out_dim,
            n_d,
            n_a,
            n_steps,
            gamma,
            n_ind,
            n_shared,
            epsilon,
            vbs,
            momentum,
        )

    def forward(self, x):
        x = self.embedder(x) #thực hiện emdedding
        return self.tabnet(x) #thực hiện tabnet

    def forward_masks(self, x):
        x = self.embedder(x) #thực hiện emdedding
        return self.tabnet.forward_masks(x)



