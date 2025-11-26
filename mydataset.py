from torch_geometric.data import Data, InMemoryDataset, DataLoader,download_url, extract_zip
import torch
import numpy as np
from torch_geometric.utils import dense_to_sparse
from dig.sslgraph.evaluation import GraphUnsupervised
from dig.sslgraph.dataset.TUDataset import TUDatasetExt
from dig.sslgraph.dataset.feat_expansion import CatDegOnehot, get_max_deg
import os.path as osp
import os, shutil, torch
from torch_geometric.utils import coalesce, cumsum, one_hot, remove_self_loops,degree
from typing import Dict, List, Optional, Tuple
from torch import Tensor
from torch_geometric.data import InMemoryDataset, extract_zip,Batch, Data
from torch_geometric.io import read_tu_data
import os
import shutil
from itertools import repeat, product
import random
from copy import deepcopy
import pdb
import torch.nn.functional as F



def get_dataset(name, task='unsupervised', feat_str="deg", root=None):
    dataset = TUDatasetExt(root+"/unsuper_dataset/", name=name, task=task)
    if feat_str.find("deg") >= 0:
        max_degree = get_max_deg(dataset)
        dataset = TUDatasetExt(root+"./unsuper_dataset/", name=name, task=task,
                                transform=CatDegOnehot(max_degree), use_node_attr=True)
    return dataset

class myTUDataset(InMemoryDataset):
    r"""An extended TUDataset from `Pytorch Geometric 
    <https://pytorch-geometric.readthedocs.io/en/latest/index.html>`_, including
    a variety of graph kernel benchmark datasets, *e.g.* "IMDB-BINARY", 
    "REDDIT-BINARY" or "PROTEINS".

    .. note::
        Some datasets may not come with any node labels.
        You can then either make use of the argument :obj:`use_node_attr`
        to load additional continuous node attributes (if present) or provide
        synthetic node features using transforms such as
        like :class:`torch_geometric.transforms.Constant` or
        :class:`torch_geometric.transforms.OneHotDegree`.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The `name
            <https://chrsmrrs.github.io/datasets/docs/datasets/>`_ of the
            dataset.
        task (string): The evaluation task. Either 'semisupervised' or
            'unsupervised'.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        use_node_attr (bool, optional): If :obj:`True`, the dataset will
            contain additional continuous node attributes (if present).
            (default: :obj:`False`)
        use_edge_attr (bool, optional): If :obj:`True`, the dataset will
            contain additional continuous edge attributes (if present).
            (default: :obj:`False`)
        cleaned (bool, optional): If :obj:`True`, the dataset will
            contain only non-isomorphic graphs. (default: :obj:`False`)
        processed_filename (string, optional): The name of the processed data file.
            (default: obj: `data.pt`)
    """
    url = 'https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets'
    cleaned_url = ('https://raw.githubusercontent.com/nd7141/graph_datasets/master/datasets')

    def __init__(self,
                 root='.',
                 name="meg4lyf",
                 task="unsupervised",
                 graph_data=None,
                 feature=None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 use_node_attr=True,
                 use_edge_attr=True,
                 cleaned=False,
                 processed_filename='data.pt'
                 ):
        self.processed_filename = processed_filename
        self.name = name
        self.cleaned = cleaned
        self.task = task
        self.graph_data = graph_data
        self.feature = feature
        super(myTUDataset, self).__init__(root, transform, pre_transform, pre_filter)

        self.process()
        if self.task == "semisupervised":
            if self.data.x is not None and not use_node_attr:
                num_node_attributes = self.num_node_attributes
                self.data.x = self.data.x[:, num_node_attributes:]
            if self.data.edge_attr is not None and not use_edge_attr:
                num_edge_attributes = self.num_edge_attributes
                self.data.edge_attr = self.data.edge_attr[:, num_edge_attributes:]

        elif self.task == "unsupervised":
            if self.data.x is not None and not use_node_attr:
                num_node_attributes = self.num_node_attributes
                self.data.x = self.data.x[:, num_node_attributes:]
            if self.data.edge_attr is not None and not use_edge_attr:
                num_edge_attributes = self.num_edge_attributes
                self.data.edge_attr = self.data.edge_attr[:, num_edge_attributes:]
            if self.data.x is None:
                edge_index = self.data.edge_index[0, :].numpy()
                _, num_edge = self.data.edge_index.size()
                nlist = [edge_index[n] + 1 for n in range(num_edge - 1) if edge_index[n] > edge_index[n + 1]]
                nlist.append(edge_index[-1] + 1)

                num_node = np.array(nlist).sum()
                self.data.x = torch.ones((num_node, 1))

                edge_slice = [0]
                k = 0
                for n in nlist:
                    k = k + n
                    edge_slice.append(k)
                self.slices['x'] = torch.tensor(edge_slice)
        else:
            ValueError("Wrong task name")
        # print(f"self.data.x.size(): {self.data.x.size()}")
    @property
    def raw_dir(self):
        name = 'raw{}'.format('_cleaned' if self.cleaned else '')
        return osp.join(self.root, self.name, name)

    @property
    def processed_dir(self):
        name = 'processed{}'.format('_cleaned' if self.cleaned else '')
        return osp.join(self.root, self.name, name)

    @property
    def num_node_labels(self):
        if self.data.x is None:
            return 0
        for i in range(self.data.x.size(1)):
            x = self.data.x[:, i:]
            if ((x == 0) | (x == 1)).all() and (x.sum(dim=1) == 1).all():
                return self.data.x.size(1) - i
        return 0

    @property
    def num_node_attributes(self):
        if self.data.x is None:
            return 0
        # print(f"self.data.x.size: {self.data.x.size()}")
        return self.data.x.size(1) - self.num_node_labels

    @property
    def num_edge_labels(self):
        if self.data.edge_attr is None:
            return 0
        for i in range(self.data.edge_attr.size(1)):
            if self.data.edge_attr[:, i:].sum() == self.data.edge_attr.size(0):
                return self.data.edge_attr.size(1) - i
        return 0

    @property
    def num_edge_attributes(self):
        print(f"self.data.edge_attr.size: {self.data.edge_attr.size()}")
        if self.data.edge_attr is None:
            return 0
        return self.data.edge_attr.size(1) - self.num_edge_labels

    @property
    def raw_file_names(self):
        names = ['A', 'graph_indicator']
        return ['{}_{}.txt'.format(self.name, name) for name in names]

    @property
    def processed_file_names(self):
        return self.processed_filename

    def read_data(self):
        all_edge_indices = []
        all_edge_attrs = []
        all_x = []
        batch = []  # 记录每个节点属于哪个图
        node_offset = 0  # 用于调整边索引
        # 处理每个图并合并数据
        for i in range(self.graph_data.shape[0]):
            # 从邻接矩阵转换为稀疏表示
            edge_index, edge_attr = dense_to_sparse(torch.tensor(self.graph_data[i], dtype=torch.float))
            x = torch.tensor(self.feature[i], dtype=torch.float)
            x=x.unsqueeze(1)
            edge_attr=edge_attr.unsqueeze(1)
            num_nodes = x.size(0)
            
            # 调整边索引以适应全局索引
            edge_index = edge_index + node_offset
            
            # 将数据添加到列表中
            all_edge_indices.append(edge_index)
            all_edge_attrs.append(edge_attr)
            all_x.append(x)
            
            # 更新batch标识符
            batch.append(torch.full((num_nodes,), i, dtype=torch.long))
            
            # 更新节点偏移量
            node_offset += num_nodes
        
        # 合并所有图的数据
        edge_index = torch.cat(all_edge_indices, dim=1)
        edge_attr = torch.cat(all_edge_attrs, dim=0) if all_edge_attrs[0] is not None else None
        x = torch.cat(all_x, dim=0)
        batch = torch.cat(batch, dim=0)
        
        # 处理边：移除自环和合并重复边
        num_nodes = x.size(0)
        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes)
        
        # 创建数据对象
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        
        # 使用batch向量分割数据
        data, slices = split(data, batch)
        # 可选：记录特征维度信息
        # sizes = {
        #     'num_node_features': x.size(1),
        #     'num_edge_features': 0 if edge_attr is None else edge_attr.size(1),
        # }
        
        return data, slices
        def read_my_data(self):
            data_list = []
            for i in range(self.graph_data.shape[0]):
                edge_index,edge_attr = dense_to_sparse(torch.tensor(self.graph_data[i], dtype=torch.float))
                x = torch.tensor(self.feature[i], dtype=torch.float)
                data = Data(x=x, edge_index=edge_index,edge_attr=edge_attr)
                data_list.append(data)
            data, slices = self.collate(data_list)
            return data, slices


    def process(self):
        self.data, self.slices = self.read_data()

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)
        # torch.save((self.data, self.slices), self.processed_paths[0])
        # print(f"self.data.x.size(): {self.data.x.size()}")
        return self.data, self.slices

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))

    def get_num_feature(self):
        data = self.data.__class__()

        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data.__cat_dim__(key, item)] = slice(slices[0], slices[0 + 1])
            else:
                s = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]
        _, num_feature = data.x.size()

        return num_feature

    def get(self, idx):
        # print(f"self.data.x.size(): {self.data.x.size()}")
        data = self.data.__class__()

        for key in self.data.keys():
            if key == 'num_nodes':
                continue
            item, slices = self.data[key], self.slices[key]
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            else:
                s = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]

        if self.task == "unsupervised":
            node_num = data.edge_index.max()
            sl = torch.tensor([[n, n] for n in range(node_num)]).t()
            data.edge_index = torch.cat((data.edge_index, sl), dim=1)
            
        return data

def split(data: Data, batch: Tensor) -> Tuple[Data, Dict[str, Tensor]]:
    node_slice = cumsum(torch.bincount(batch))

    assert data.edge_index is not None
    row, _ = data.edge_index
    edge_slice = cumsum(torch.bincount(batch[row]))

    # Edge indices should start at zero for every graph.
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)

    slices = {'edge_index': edge_slice}
    if data.x is not None:
        slices['x'] = node_slice
    else:
        # Imitate `collate` functionality:
        data._num_nodes = torch.bincount(batch).tolist()
        data.num_nodes = batch.numel()
    if data.edge_attr is not None:
        slices['edge_attr'] = edge_slice
    if data.y is not None:
        assert isinstance(data.y, Tensor)
        if data.y.size(0) == batch.size(0):
            slices['y'] = node_slice
        else:
            slices['y'] = torch.arange(0, int(batch[-1]) + 2, dtype=torch.long)

    return data, slices

class unsupervised_training(object):
    def __init__(self, dataset, classifier='SVC', log_interval=1, epoch_select='test_max', 
                 metric='acc', n_folds=10, device=None, **kwargs):
        
        self.dataset = dataset
        self.epoch_select = epoch_select
        self.metric = metric
        self.classifier = classifier
        self.log_interval = log_interval
        self.n_folds = n_folds
        self.out_dim = dataset.num_classes
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, int):
            self.device = torch.device('cuda:%d'%device)
        else:
            self.device = device
    def setup_train_config(self, batch_size = 256, p_optim = 'Adam', p_lr = 0.01, 
                           p_weight_decay = 0, p_epoch = 20, svc_search = True):
        r"""Method to setup training config.
        
        Args:
            batch_size (int, optional): Batch size for pretraining and inference. 
                (default: :obj:`256`)
            p_optim (string, or torch.optim.Optimizer class): Optimizer for pretraining.
                (default: :obj:`"Adam"`)
            p_lr (float, optional): Pretraining learning rate. (default: :obj:`0.01`)
            p_weight_decay (float, optional): Pretraining weight decay rate. 
                (default: :obj:`0`)
            p_epoch (int, optional): Pretraining epochs number. (default: :obj:`20`)
            svc_search (string, optional): If :obj:`True`, search for hyper-parameter 
                :obj:`C` in SVC. (default: :obj:`True`)
        """
        
        self.batch_size = batch_size

        self.p_optim = p_optim
        self.p_lr = p_lr
        self.p_weight_decay = p_weight_decay
        self.p_epoch = p_epoch
        
        self.search = svc_search

    def get_optim(self, optim):
        if callable(optim):
            return optim
        optims = {'Adam': torch.optim.Adam}
        return optims[optim]

    def get_embed(self, model, loader):
    
        model.eval()
        ret, y = [], []
        with torch.no_grad():
            for data in loader:
                if data.y is not None:
                    y.append(data.y.numpy())
                data.to(self.device)
                embed = model(data)
                # 
                # print("embed shape: ",embed.shape)
                if embed.shape[0] > 1:
                    embed = embed.unsqueeze(0)
                
                # print(f"embed: {embed.shape}")
                ret.append(embed.cpu().numpy())

        ret = np.concatenate(ret, 0)
        if len(y) > 0:
            y = np.concatenate(y, 0)
        return ret, y

    def training(self,learning_model, encoder):
        pretrain_loader = DataLoader(self.dataset,self.batch_size, shuffle=True)
        if isinstance(encoder, list):
                params = [{'params': enc.parameters()} for enc in encoder]
        else:
            params = encoder.parameters()
        
        p_optimizer = self.get_optim(self.p_optim)(params, lr=self.p_lr, 
                                                    weight_decay=self.p_weight_decay)
        for i, enc in enumerate(learning_model.train(encoder, pretrain_loader, 
                                                     p_optimizer, self.p_epoch, True)):            
            encoder = enc
        return encoder 
    


class TUDatasetLaGraph(InMemoryDataset):
    r"""A variety of graph kernel benchmark datasets, *.e.g.* "IMDB-BINARY",
    "REDDIT-BINARY" or "PROTEINS", collected from the `TU Dortmund University
    <https://chrsmrrs.github.io/datasets>`_.
    In addition, this dataset wrapper provides `cleaned dataset versions
    <https://github.com/nd7141/graph_datasets>`_ as motivated by the
    `"Understanding Isomorphism Bias in Graph Data Sets"
    <https://arxiv.org/abs/1910.12091>`_ paper, containing only non-isomorphic
    graphs.

    .. note::
        Some datasets may not come with any node labels.
        You can then either make use of the argument :obj:`use_node_attr`
        to load additional continuous node attributes (if present) or provide
        synthetic node features using transforms such as
        like :class:`torch_geometric.transforms.Constant` or
        :class:`torch_geometric.transforms.OneHotDegree`.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The `name
            <https://chrsmrrs.github.io/datasets/docs/datasets/>`_ of the
            dataset.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        use_node_attr (bool, optional): If :obj:`True`, the dataset will
            contain additional continuous node attributes (if present).
            (default: :obj:`False`)
        use_edge_attr (bool, optional): If :obj:`True`, the dataset will
            contain additional continuous edge attributes (if present).
            (default: :obj:`False`)
        cleaned: (bool, optional): If :obj:`True`, the dataset will
            contain only non-isomorphic graphs. (default: :obj:`False`)
    """

    url = ('http://ls11-www.cs.tu-dortmund.de/people/morris/'
           'graphkerneldatasets')
    cleaned_url = ('https://raw.githubusercontent.com/nd7141/'
                   'graph_datasets/master/datasets')

    def __init__(self, root, name, graph_data, feature, transform=None, pre_transform=None,
                 pre_filter=None, use_node_attr=False, use_edge_attr=False,
                 cleaned=False, aug=False, args=None):
        self.name = name
        self.graph_data = graph_data
        self.feature = feature
        self.cleaned = cleaned
        self.th = args.th
        self.mask_ratio = args.mratio
        self.mask_std = args.mstd
        self.mask_mode = args.mmode
        
        super(TUDatasetLaGraph, self).__init__(root, transform, pre_transform,
                                        pre_filter)
        self.data, self.slices = self.read_data()

        # print("begin do it")

        if self.data.x is not None and not use_node_attr:
            num_node_attributes = self.num_node_attributes
            self.data.x = self.data.x[:, num_node_attributes:]
        if self.data.edge_attr is not None and not use_edge_attr:
            num_edge_attributes = self.num_edge_attributes
            self.data.edge_attr = self.data.edge_attr[:, num_edge_attributes:]
        if not (self.name == 'MUTAG' or self.name == 'PTC_MR' or self.name == 'DD' or self.name == 'PROTEINS' or self.name == 'NCI1' or self.name == 'NCI109'):
            # print("begin done it")
            edge_index = self.data.edge_index[0, :].numpy()
            _, num_edge = self.data.edge_index.size()
            nlist = [edge_index[n] + 1 for n in range(num_edge - 1) if edge_index[n] > edge_index[n + 1]]
            nlist.append(edge_index[-1] + 1)

            num_node = np.array(nlist).sum()
            self.data.x = torch.ones((num_node, 1))
            # print(f"self.data.x: {self.data.x}")
            # print(f"edge_index: {edge_index}")
            # print(f"nlist: {nlist}")
            # print(f"num_node: {num_node}")
            edge_slice = [0]
            k = 0
            for n in nlist:
                k = k + n
                edge_slice.append(k)
            self.slices['x'] = torch.tensor(edge_slice)

            '''
            print(self.data.x.size())
            print(self.slices['x'])
            print(self.slices['x'].size())
            assert False
            '''

        self.aug = aug
    def read_data(self):
        all_edge_indices = []
        all_edge_attrs = []
        all_x = []
        batch = []  # 记录每个节点属于哪个图
        node_offset = 0  # 用于调整边索引
        # 处理每个图并合并数据
        for i in range(self.graph_data.shape[0]):
            # 从邻接矩阵转换为稀疏表示
            edge_index, edge_attr = dense_to_sparse(torch.tensor(self.graph_data[i], dtype=torch.float))
            x = torch.tensor(self.feature[i], dtype=torch.float)
            x=x.unsqueeze(1)
            edge_attr=edge_attr.unsqueeze(1)
            num_nodes = x.size(0)
            
            # 调整边索引以适应全局索引
            edge_index = edge_index + node_offset
            
            # 将数据添加到列表中
            all_edge_indices.append(edge_index)
            all_edge_attrs.append(edge_attr)
            all_x.append(x)
            
            # 更新batch标识符
            batch.append(torch.full((num_nodes,), i, dtype=torch.long))
            
            # 更新节点偏移量
            node_offset += num_nodes
        
        # 合并所有图的数据
        edge_index = torch.cat(all_edge_indices, dim=1)
        edge_attr = torch.cat(all_edge_attrs, dim=0) if all_edge_attrs[0] is not None else None
        x = torch.cat(all_x, dim=0)
        batch = torch.cat(batch, dim=0)
        
        # 处理边：移除自环和合并重复边
        num_nodes = x.size(0)
        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes)
        
        # 创建数据对象
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        
        # 使用batch向量分割数据
        data, slices = split(data, batch)
        # 可选：记录特征维度信息
        # sizes = {
        #     'num_node_features': x.size(1),
        #     'num_edge_features': 0 if edge_attr is None else edge_attr.size(1),
        # }
        
        return data, slices
    @property
    def raw_dir(self):
        name = 'raw{}'.format('_cleaned' if self.cleaned else '')
        return osp.join(self.root, self.name, name)

    @property
    def processed_dir(self):
        name = 'processed{}'.format('_cleaned' if self.cleaned else '')
        return osp.join(self.root, self.name, name)

    @property
    def num_node_labels(self):
        if self.data.x is None:
            return 0
        for i in range(self.data.x.size(1)):
            x = self.data.x[:, i:]
            if ((x == 0) | (x == 1)).all() and (x.sum(dim=1) == 1).all():
                return self.data.x.size(1) - i
        return 0

    @property
    def num_node_attributes(self):
        if self.data.x is None:
            return 0
        return self.data.x.size(1) - self.num_node_labels

    @property
    def num_edge_labels(self):
        if self.data.edge_attr is None:
            return 0
        for i in range(self.data.edge_attr.size(1)):
            if self.data.edge_attr[:, i:].sum() == self.data.edge_attr.size(0):
                return self.data.edge_attr.size(1) - i
        return 0

    @property
    def num_edge_attributes(self):
        if self.data.edge_attr is None:
            return 0
        return self.data.edge_attr.size(1) - self.num_edge_labels

    @property
    def raw_file_names(self):
        names = ['A', 'graph_indicator']
        return ['{}_{}.txt'.format(self.name, name) for name in names]

    @property
    def processed_file_names(self):
        return 'data.pt'

    # def download(self):
    #     url = self.cleaned_url if self.cleaned else self.url
    #     folder = osp.join(self.root, self.name)
    #     path = download_url('{}/{}.zip'.format(url, self.name), folder)
    #     extract_zip(path, folder)
    #     os.unlink(path)
    #     shutil.rmtree(self.raw_dir)
    #     os.rename(osp.join(folder, self.name), self.raw_dir)

    def process(self):
        self.data, self.slices = self.read_data()

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))

    def get_num_feature(self,idx):
        data = self.data.__class__()

        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[0]

        for key in self.data.keys():
            item, slices = self.data[key], self.slices[key]
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data.__cat_dim__(key,
                                        item)] = slice(slices[0],
                                                       slices[0 + 1])
            else:
                s = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]
        _, num_feature = data.x.size()

        if self.th < 0:
            return num_feature
        else:
            return self.th + 1
#             if self.args.concat:
#                 return self.th + 1 + num_feature
#             else:
#                 return self.th + 1


    def get_num_classes(self):
        if self.name == 'COLLAB':
            return 3
        elif self.name == 'REDDIT-MULTI-5K':
            return 5
        else:
            return 2


    def get(self, idx):
        data = self.data.__class__()

        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[idx]

        for key in self.data.keys():
            item, slices = self.data[key], self.slices[key]
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data.__cat_dim__(key,
                                        item)] = slice(slices[idx],
                                                       slices[idx + 1])
            else:
                s = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]

        """
        edge_index = data.edge_index
        node_num = data.x.size()[0]
        edge_num = data.edge_index.size()[1]
        data.edge_index = torch.tensor([[edge_index[0, n], edge_index[1, n]] for n in range(edge_num) if edge_index[0, n] < node_num and edge_index[1, n] < node_num] + [[n, n] for n in range(node_num)], dtype=torch.int64).t()
        """

        # Add self loop
        node_num = data.edge_index.max()
        sl = torch.tensor([[n,n] for n in range(node_num)]).t()
        data.edge_index = torch.cat((data.edge_index, sl), dim=1)

        if self.th >= 0:
            idx = data.edge_index[0]
            deg = degree(idx, node_num+1)
            deg = torch.clamp(deg, 0, self.th).long()
            deg = F.one_hot(deg, num_classes=self.th+1).to(torch.float)
#             if self.args.concat:
#                 data.x = torch.cat((data.x, deg), dim=1)
#             else:
#                 data.x = deg
            data.x = deg

        if self.aug:
            data_aug = mask_nodes(deepcopy(data), self.mask_ratio, self.mask_std, self.mask_mode)
            return data, data_aug

        else:
            return data


def mask_nodes(data, mask_ratio=0.1, mask_std=0.5, mode='whole'):
    node_num, feat_dim = data.x.size()

    if mode == 'whole':
        mask_num = int(node_num * mask_ratio)

        idx_mask = np.random.choice(node_num, mask_num, replace=False)
        data.x[idx_mask] = torch.tensor(np.random.normal(loc=0, scale=mask_std, size=(mask_num, feat_dim)), dtype=torch.float32)

        mask = torch.zeros(node_num)
        mask[idx_mask] = 1

    elif mode == 'partial':
        mask = torch.zeros(node_num, feat_dim)

        for i in range(node_num):
            for j in range(feat_dim):
                if random.random() < mask_ratio:
                    data.x[i][j] = torch.tensor(np.random.normal(loc=0, scale=mask_std), dtype=torch.float32)
                    mask[i][j] = 1

    elif mode == 'onehot':
        mask_num = int(node_num * mask_ratio)
        idx_mask = np.random.choice(node_num, mask_num, replace=False)
        data.x[idx_mask] = torch.tensor(
            np.eye(feat_dim)[np.random.randint(0, feat_dim, size=(mask_num))], dtype=torch.float32)

        mask = torch.zeros(node_num)
        mask[idx_mask] = 1

    data.mask = mask
    return data