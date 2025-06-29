import torch
import torch.nn.functional as F
import spconv.pytorch as spconv
from torch_scatter import scatter_mean
import MinkowskiEngine as ME
from mmdet3d.registry import MODELS
from mmdet3d.structures import PointData
from mmdet3d.models import Base3DDetector
from .mask_matrix_nms import mask_matrix_nms
from oneformer3d import S3DISOneFormer3D
import torch.nn as nn
import functools
from .mamba import Mamba, MambaConfig
import torch
import torch.nn as nn
from mmengine.model import BaseModule

import torch
import torch.nn as nn

class MLP(nn.Sequential):
    def __init__(self, in_channels, out_channels, hidden_channels=None, norm_fn=None, num_layers=2, dropout=0.0):
        if hidden_channels is None:
            hidden_channels = in_channels  # 默认等于输入通道

        modules = []

        # 第一个隐藏层
        modules.append(nn.Linear(in_channels, hidden_channels))
        if norm_fn:
            modules.append(norm_fn(hidden_channels))
        modules.append(nn.ReLU())
        if dropout > 0:
            modules.append(nn.Dropout(p=dropout))

        # 中间隐藏层（如果有多个隐藏层）
        for _ in range(num_layers - 2):
            modules.append(nn.Linear(hidden_channels, hidden_channels))
            if norm_fn:
                modules.append(norm_fn(hidden_channels))
            modules.append(nn.ReLU())
            if dropout > 0:
                modules.append(nn.Dropout(p=dropout))

        # 输出层（不加激活和dropout）
        modules.append(nn.Linear(hidden_channels, out_channels))
        
        super().__init__(*modules)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        # 初始化最后一层
        if isinstance(self[-1], nn.Linear):
            nn.init.normal_(self[-1].weight, 0, 0.01)
            nn.init.constant_(self[-1].bias, 0)

class MLP1(nn.Sequential):
    def __init__(self, in_channels, out_channels, hidden_channels=None, norm_fn=None, num_layers=2):
        if hidden_channels is None:
            hidden_channels = in_channels  # 默认等于输入通道

        modules = []

        # 第一个隐藏层
        modules.append(nn.Linear(in_channels, hidden_channels))
        if norm_fn:
            modules.append(norm_fn(hidden_channels))
        modules.append(nn.ReLU())

        # 中间隐藏层（如果有多个隐藏层）
        for _ in range(num_layers - 2):
            modules.append(nn.Linear(hidden_channels, hidden_channels))
            if norm_fn:
                modules.append(norm_fn(hidden_channels))
            modules.append(nn.ReLU())

        # 输出层
        modules.append(nn.Linear(hidden_channels, out_channels))
        
        super().__init__(*modules)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        # 初始化最后一层
        if isinstance(self[-1], nn.Linear):
            nn.init.normal_(self[-1].weight, 0, 0.01)
            nn.init.constant_(self[-1].bias, 0)

@MODELS.register_module()
class SegInstSpformer3DSeg(Base3DDetector):
    r"""SegInstSpformer3D for training on different datasets jointly.

    Args:
        in_channels (int): Number of input channels.
        num_channels (int): Number of output channels.
        voxel_size (float): Voxel size.
        num_classes_1dataset (int): Number of classes in the first dataset.
        prefix_1dataset (string): Prefix for the first dataset.
        min_spatial_shape (int): Minimal shape for spconv tensor.
        backbone (ConfigDict): Config dict of the backbone.
        decoder (ConfigDict): Config dict of the decoder.
        criterion (ConfigDict): Config dict of the criterion.
        train_cfg (dict, optional): Config dict of training hyper-parameters.
            Defaults to None.
        test_cfg (dict, optional): Config dict of test hyper-parameters.
            Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or ConfigDict, optional): the config to control the
            initialization. Defaults to None.
    """

    def __init__(self,
                 in_channels,
                 num_channels,
                 voxel_size,
                 num_classes_1dataset,
                 prefix_1dataset,
                 min_spatial_shape,
                 backbone=None,
                 decoder=None,
                 criterion=None,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 init_cfg=None):
        super(SegInstSpformer3DSeg, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.num_classes_1dataset = num_classes_1dataset 
        
        self.prefix_1dataset = prefix_1dataset 
        
        self.unet = MODELS.build(backbone)
        self.decoder = MODELS.build(decoder)
        self.criterion = MODELS.build(criterion)
        self.voxel_size = voxel_size
        self.min_spatial_shape = min_spatial_shape
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._init_layers(in_channels, num_channels)
        config = MambaConfig(d_model=32, n_layers=1)
        self.mamba_layers = Mamba(config)
        norm_fn = functools.partial(nn.LayerNorm, eps=1e-5)
        self.ffn = MLP1(256, 32, hidden_channels=32, norm_fn=norm_fn, num_layers=1)
        self.out_sem = MLP(32, num_classes_1dataset + 1, hidden_channels=32, norm_fn=norm_fn, num_layers=3)

    
    def _init_layers(self, in_channels, num_channels):
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels,
                num_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key='subm1'))
        self.output_layer = spconv.SparseSequential(
            torch.nn.BatchNorm1d(num_channels, eps=1e-4, momentum=0.1),
            torch.nn.ReLU(inplace=True))

    def extract_feat(self, x):
        """Extract features from sparse tensor.

        Args:
            x (SparseTensor): Input sparse tensor of shape
                (n_points, in_channels).

        Returns:
            List[Tensor]: of len batch_size,
                each of shape (n_points_i, n_channels).
        """
        x = self.input_conv(x)
        x, _ = self.unet(x)
        x = self.output_layer(x)
        out = []
        for i in x.indices[:, 0].unique():
            out.append(x.features[x.indices[:, 0] == i])
        return out

    def collate(self, points, elastic_points=None):
        """Collate batch of points to sparse tensor.

        Args:
            points (List[Tensor]): Batch of points.
            quantization_mode (SparseTensorQuantizationMode): Minkowski
                quantization mode. We use random sample for training
                and unweighted average for inference.

        Returns:
            TensorField: Containing features and coordinates of a
                sparse tensor.
        """
        # 
        if elastic_points is None:
            coordinates, features = ME.utils.batch_sparse_collate(
                [((p[:, :3] - p[:, :3].min(0)[0]) / self.voxel_size,
                  # s rgb or rgb
                  torch.hstack((p[:, 3:], p[:, :3] - p[:, :3].mean(0))))
                 for p in points])
        else:
            coordinates, features = ME.utils.batch_sparse_collate(
                [((el_p - el_p.min(0)[0]),
                  # s rgb or rgb
                  torch.hstack((p[:, 3:], p[:, :3] - p[:, :3].mean(0))))
                #   torch.hstack((p[:, 3:], p[:, :3])))
                  # torch.hstack((p[:, 3:],)))
                 for el_p, p in zip(elastic_points, points)])
        # 
        spatial_shape = torch.clip(
            coordinates.max(0)[0][1:] + 1, self.min_spatial_shape)
        field = ME.TensorField(features=features, coordinates=coordinates)
        tensor = field.sparse()
        coordinates = tensor.coordinates
        features = tensor.features
        inverse_mapping = field.inverse_mapping(tensor.coordinate_map_key)
        return coordinates, features, inverse_mapping, spatial_shape

    def _forward(*args, **kwargs):
        """Implement abstract method of Base3DDetector."""
        pass

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Calculate losses from a batch of inputs dict and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instances_3d` and `gt_sem_seg_3d`.
        Returns:
            dict: A dictionary of loss components.
        """
        # 
        coordinates, features, inverse_mapping, spatial_shape = self.collate(
            batch_inputs_dict['points'],
            batch_inputs_dict.get('elastic_coords', None))
        # 
        x = spconv.SparseConvTensor(
            features, coordinates, spatial_shape, len(batch_data_samples))

        x = self.extract_feat(x)
        scene_names = []
        for i in range(len(batch_data_samples)):
           scene_names.append(batch_data_samples[i].lidar_path)
        # 
        x, mask_feats = self.decoder(x, scene_names)
        mask_feats = [mask_feats]
        sem_preds = []
        # print(mask_feats[0].shape)
        # import pdb;pdb.set_trace()
        for ex in mask_feats:
            sem_preds.append(self.out_sem(self.mamba_layers(self.ffn(ex).unsqueeze(0)).squeeze(0)).permute(1, 0))
        x['sem_preds'] = sem_preds
        # import pdb;pdb.set_trace()
        sp_gt_instances = []
        for i in range(len(batch_data_samples)):
            
            voxel_superpoints = inverse_mapping[coordinates[:, 0][ \
                                                        inverse_mapping] == i]
            voxel_superpoints = torch.unique(voxel_superpoints,
                                             return_inverse=True)[1]
            inst_mask = batch_data_samples[i].gt_pts_seg.pts_instance_mask
            sem_mask = batch_data_samples[i].gt_pts_seg.pts_semantic_mask
            assert voxel_superpoints.shape == inst_mask.shape
            # 
            batch_data_samples[i].gt_instances_3d.sp_sem_masks = \
                                S3DISOneFormer3D.get_gt_semantic_masks(sem_mask,
                                                            voxel_superpoints,
                                                            self.num_classes_1dataset)
            batch_data_samples[i].gt_instances_3d.sp_inst_masks = \
                                S3DISOneFormer3D.get_gt_inst_masks(inst_mask,
                                                       voxel_superpoints)
            sp_gt_instances.append(batch_data_samples[i].gt_instances_3d)

        loss = self.criterion(x, sp_gt_instances)
        return loss
    
    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Predict results from a batch of inputs and data samples with post-
        processing.
        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instance_3d` and `gt_sem_seg_3d`.
        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input samples. Each Det3DDataSample contains 'pred_pts_seg'.
            And the `pred_pts_seg` contains following keys.
                - instance_scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - instance_labels (Tensor): Labels of instances, has a shape
                    (num_instances, )
                - pts_instance_mask (Tensor): Instance mask, has a shape
                    (num_points, num_instances) of type bool.
        """
        # 
        coordinates, features, inverse_mapping, spatial_shape = self.collate(
            batch_inputs_dict['points'])
        x = spconv.SparseConvTensor(
            features, coordinates, spatial_shape, len(batch_data_samples))

        x = self.extract_feat(x)
        

        scene_names = []
        for i in range(len(batch_data_samples)):
            scene_names.append(batch_data_samples[i].lidar_path)
        x, mask_feats = self.decoder(x, scene_names)
        mask_feats = [mask_feats]
        sem_preds = []
        for ex in mask_feats:
            sem_preds.append(self.out_sem(self.mamba_layers(self.ffn(ex).unsqueeze(0)).squeeze(0)).permute(1, 0))
        x['sem_preds'] = sem_preds

        results_list = self.predict_by_feat(x, inverse_mapping, scene_names)

        for i, data_sample in enumerate(batch_data_samples):
            data_sample.pred_pts_seg = results_list[i]
        return batch_data_samples
    
    def pred_inst(self, pred_masks, pred_scores, pred_labels,
                  superpoints, score_threshold):
        """Predict instance masks for a single scene.

        Args:
            pred_masks (Tensor): of shape (n_queries, n_points).
            pred_scores (Tensor): of shape (n_queris, 1).
            pred_labels (Tensor): of shape (n_queries, n_instance_classes + 1).
            superpoints (Tensor): of shape (n_raw_points,).
            score_threshold (float): minimal score for predicted object.

        Returns:
            Tuple:
                Tensor: mask_preds of shape (n_preds, n_raw_points),
                Tensor: labels of shape (n_preds,),
                Tensor: scors of shape (n_preds,).
        """
        scores = F.softmax(pred_labels, dim=-1)[:, :-1]
        scores *= pred_scores

        labels = torch.arange(
            self.num_classes_1dataset,
            device=scores.device).unsqueeze(0).repeat(
                self.decoder.num_queries_1dataset,
                1).flatten(0, 1)
        
        scores, topk_idx = scores.flatten(0, 1).topk(
            self.test_cfg.topk_insts, sorted=False)
        labels = labels[topk_idx]

        topk_idx = torch.div(topk_idx, self.num_classes_1dataset, rounding_mode='floor')
        mask_pred = pred_masks
        mask_pred = mask_pred[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()
        if self.test_cfg.get('obj_normalization', None):
            mask_pred_thr = mask_pred_sigmoid > \
                self.test_cfg.obj_normalization_thr
            mask_scores = (mask_pred_sigmoid * mask_pred_thr).sum(1) / \
                (mask_pred_thr.sum(1) + 1e-6)
            scores = scores * mask_scores

        if self.test_cfg.get('nms', None):
            kernel = self.test_cfg.matrix_nms_kernel
            scores, labels, mask_pred_sigmoid, _ = mask_matrix_nms(
                mask_pred_sigmoid, labels, scores, kernel=kernel)

        mask_pred = mask_pred_sigmoid > self.test_cfg.sp_score_thr
        mask_pred = mask_pred[:, superpoints]
        # score_thr
        score_mask = scores > score_threshold
        scores = scores[score_mask]
        labels = labels[score_mask]
        mask_pred = mask_pred[score_mask]

        # npoint_thr
        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum > self.test_cfg.npoint_thr
        scores = scores[npoint_mask]
        labels = labels[npoint_mask]
        mask_pred = mask_pred[npoint_mask]

        return mask_pred, labels, scores
   
    def pred_sem(self, pred_masks, superpoints):
        """Predict semantic masks for a single scene.

        Args:
            pred_masks (Tensor): of shape (n_points, n_semantic_classes).
            superpoints (Tensor): of shape (n_raw_points,).        

        Returns:
            Tensor: semantic preds of shape
                (n_raw_points, 1).
        """
        # import pdb;pdb.set_trace()
        mask_pred = pred_masks.sigmoid()
        mask_pred = mask_pred[:, superpoints]
        seg_map = mask_pred.argmax(0)
        return seg_map

    def predict_by_feat(self, out, superpoints, scene_names):
        """Predict instance masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).
            scene_names (List[string]): of len 1, which contain scene name.

        Returns:
            List[PointData]: of len 1 with `pts_instance_mask`, 
                `instance_labels`, `instance_scores`.
        """
        pred_labels = out['cls_preds'][0]
        pred_masks = out['masks'][0]
        pred_scores = out['scores'][0]
        sem_preds = out['sem_preds'][0]

        inst_res = self.pred_inst(pred_masks,
                                  pred_scores,
                                  pred_labels,
                                  superpoints, self.test_cfg.inst_score_thr)
        sem_res = self.pred_sem(sem_preds,
                                superpoints)
        pts_semantic_mask = [sem_res.cpu().numpy()]
        pts_instance_mask = [inst_res[0].cpu().bool().numpy()]

        return [
            PointData(
                pts_semantic_mask=pts_semantic_mask,
                pts_instance_mask=pts_instance_mask,
                instance_labels=inst_res[1].cpu().numpy(),
                instance_scores=inst_res[2].cpu().numpy())]




@MODELS.register_module()
class SegInstSpformer3D(Base3DDetector):
    r"""SegInstSpformer3D for training on different datasets jointly.

    Args:
        in_channels (int): Number of input channels.
        num_channels (int): Number of output channels.
        voxel_size (float): Voxel size.
        num_classes_1dataset (int): Number of classes in the first dataset.
        prefix_1dataset (string): Prefix for the first dataset.
        min_spatial_shape (int): Minimal shape for spconv tensor.
        backbone (ConfigDict): Config dict of the backbone.
        decoder (ConfigDict): Config dict of the decoder.
        criterion (ConfigDict): Config dict of the criterion.
        train_cfg (dict, optional): Config dict of training hyper-parameters.
            Defaults to None.
        test_cfg (dict, optional): Config dict of test hyper-parameters.
            Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or ConfigDict, optional): the config to control the
            initialization. Defaults to None.
    """

    def __init__(self,
                 in_channels,
                 num_channels,
                 voxel_size,
                 num_classes_1dataset,
                 prefix_1dataset,
                 min_spatial_shape,
                 backbone=None,
                 decoder=None,
                 criterion=None,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 init_cfg=None):
        super(SegInstSpformer3D, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.num_classes_1dataset = num_classes_1dataset 
        
        self.prefix_1dataset = prefix_1dataset 
        
        self.unet = MODELS.build(backbone)
        self.decoder = MODELS.build(decoder)
        self.criterion = MODELS.build(criterion)
        self.voxel_size = voxel_size
        self.min_spatial_shape = min_spatial_shape
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._init_layers(in_channels, num_channels)
    
    def _init_layers(self, in_channels, num_channels):
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels,
                num_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key='subm1'))
        self.output_layer = spconv.SparseSequential(
            torch.nn.BatchNorm1d(num_channels, eps=1e-4, momentum=0.1),
            torch.nn.ReLU(inplace=True))

    def extract_feat(self, x):
        """Extract features from sparse tensor.

        Args:
            x (SparseTensor): Input sparse tensor of shape
                (n_points, in_channels).

        Returns:
            List[Tensor]: of len batch_size,
                each of shape (n_points_i, n_channels).
        """
        x = self.input_conv(x)
        x, _ = self.unet(x)
        x = self.output_layer(x)
        out = []
        for i in x.indices[:, 0].unique():
            out.append(x.features[x.indices[:, 0] == i])
        return out

    def collate(self, points, elastic_points=None):
        """Collate batch of points to sparse tensor.

        Args:
            points (List[Tensor]): Batch of points.
            quantization_mode (SparseTensorQuantizationMode): Minkowski
                quantization mode. We use random sample for training
                and unweighted average for inference.

        Returns:
            TensorField: Containing features and coordinates of a
                sparse tensor.
        """
        # 
        if elastic_points is None:
            coordinates, features = ME.utils.batch_sparse_collate(
                [((p[:, :3] - p[:, :3].min(0)[0]) / self.voxel_size,
                  # s rgb or rgb
                  torch.hstack((p[:, 3:], p[:, :3] - p[:, :3].mean(0))))
                 for p in points])
        else:
            coordinates, features = ME.utils.batch_sparse_collate(
                [((el_p - el_p.min(0)[0]),
                  # s rgb or rgb
                  torch.hstack((p[:, 3:], p[:, :3] - p[:, :3].mean(0))))
                #   torch.hstack((p[:, 3:], p[:, :3])))
                  # torch.hstack((p[:, 3:],)))
                 for el_p, p in zip(elastic_points, points)])
        # 
        spatial_shape = torch.clip(
            coordinates.max(0)[0][1:] + 1, self.min_spatial_shape)
        field = ME.TensorField(features=features, coordinates=coordinates)
        tensor = field.sparse()
        coordinates = tensor.coordinates
        features = tensor.features
        inverse_mapping = field.inverse_mapping(tensor.coordinate_map_key)
        return coordinates, features, inverse_mapping, spatial_shape

    def _forward(*args, **kwargs):
        """Implement abstract method of Base3DDetector."""
        pass

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Calculate losses from a batch of inputs dict and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instances_3d` and `gt_sem_seg_3d`.
        Returns:
            dict: A dictionary of loss components.
        """
        # 
        coordinates, features, inverse_mapping, spatial_shape = self.collate(
            batch_inputs_dict['points'],
            batch_inputs_dict.get('elastic_coords', None))
        # 
        x = spconv.SparseConvTensor(
            features, coordinates, spatial_shape, len(batch_data_samples))

        x = self.extract_feat(x)

        scene_names = []
        for i in range(len(batch_data_samples)):
           scene_names.append(batch_data_samples[i].lidar_path)
        # 
        x = self.decoder(x, scene_names)

        sp_gt_instances = []
        for i in range(len(batch_data_samples)):
            
            voxel_superpoints = inverse_mapping[coordinates[:, 0][ \
                                                        inverse_mapping] == i]
            voxel_superpoints = torch.unique(voxel_superpoints,
                                             return_inverse=True)[1]
            inst_mask = batch_data_samples[i].gt_pts_seg.pts_instance_mask
            sem_mask = batch_data_samples[i].gt_pts_seg.pts_semantic_mask
            assert voxel_superpoints.shape == inst_mask.shape
            # 
            batch_data_samples[i].gt_instances_3d.sp_sem_masks = \
                                S3DISOneFormer3D.get_gt_semantic_masks(sem_mask,
                                                            voxel_superpoints,
                                                            self.num_classes_1dataset)
            batch_data_samples[i].gt_instances_3d.sp_inst_masks = \
                                S3DISOneFormer3D.get_gt_inst_masks(inst_mask,
                                                       voxel_superpoints)
            sp_gt_instances.append(batch_data_samples[i].gt_instances_3d)

        loss = self.criterion(x, sp_gt_instances)
        return loss
    
    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Predict results from a batch of inputs and data samples with post-
        processing.
        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instance_3d` and `gt_sem_seg_3d`.
        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input samples. Each Det3DDataSample contains 'pred_pts_seg'.
            And the `pred_pts_seg` contains following keys.
                - instance_scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - instance_labels (Tensor): Labels of instances, has a shape
                    (num_instances, )
                - pts_instance_mask (Tensor): Instance mask, has a shape
                    (num_points, num_instances) of type bool.
        """
        # 
        coordinates, features, inverse_mapping, spatial_shape = self.collate(
            batch_inputs_dict['points'])
        x = spconv.SparseConvTensor(
            features, coordinates, spatial_shape, len(batch_data_samples))

        x = self.extract_feat(x)

        scene_names = []
        for i in range(len(batch_data_samples)):
            scene_names.append(batch_data_samples[i].lidar_path)
        x = self.decoder(x, scene_names)

        results_list = self.predict_by_feat(x, inverse_mapping, scene_names)

        for i, data_sample in enumerate(batch_data_samples):
            data_sample.pred_pts_seg = results_list[i]
        return batch_data_samples
    
    def pred_inst(self, pred_masks, pred_scores, pred_labels,
                  superpoints, score_threshold):
        """Predict instance masks for a single scene.

        Args:
            pred_masks (Tensor): of shape (n_queries, n_points).
            pred_scores (Tensor): of shape (n_queris, 1).
            pred_labels (Tensor): of shape (n_queries, n_instance_classes + 1).
            superpoints (Tensor): of shape (n_raw_points,).
            score_threshold (float): minimal score for predicted object.

        Returns:
            Tuple:
                Tensor: mask_preds of shape (n_preds, n_raw_points),
                Tensor: labels of shape (n_preds,),
                Tensor: scors of shape (n_preds,).
        """
        scores = F.softmax(pred_labels, dim=-1)[:, :-1]
        scores *= pred_scores

        labels = torch.arange(
            self.num_classes_1dataset,
            device=scores.device).unsqueeze(0).repeat(
                self.decoder.num_queries_1dataset,
                1).flatten(0, 1)
        
        scores, topk_idx = scores.flatten(0, 1).topk(
            self.test_cfg.topk_insts, sorted=False)
        labels = labels[topk_idx]

        topk_idx = torch.div(topk_idx, self.num_classes_1dataset, rounding_mode='floor')
        mask_pred = pred_masks
        mask_pred = mask_pred[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()
        if self.test_cfg.get('obj_normalization', None):
            mask_pred_thr = mask_pred_sigmoid > \
                self.test_cfg.obj_normalization_thr
            mask_scores = (mask_pred_sigmoid * mask_pred_thr).sum(1) / \
                (mask_pred_thr.sum(1) + 1e-6)
            scores = scores * mask_scores

        if self.test_cfg.get('nms', None):
            kernel = self.test_cfg.matrix_nms_kernel
            scores, labels, mask_pred_sigmoid, _ = mask_matrix_nms(
                mask_pred_sigmoid, labels, scores, kernel=kernel)

        mask_pred = mask_pred_sigmoid > self.test_cfg.sp_score_thr
        mask_pred = mask_pred[:, superpoints]
        # score_thr
        score_mask = scores > score_threshold
        scores = scores[score_mask]
        labels = labels[score_mask]
        mask_pred = mask_pred[score_mask]

        # npoint_thr
        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum > self.test_cfg.npoint_thr
        scores = scores[npoint_mask]
        labels = labels[npoint_mask]
        mask_pred = mask_pred[npoint_mask]

        return mask_pred, labels, scores
   
    def pred_sem(self, pred_masks, superpoints):
        """Predict semantic masks for a single scene.

        Args:
            pred_masks (Tensor): of shape (n_points, n_semantic_classes).
            superpoints (Tensor): of shape (n_raw_points,).        

        Returns:
            Tensor: semantic preds of shape
                (n_raw_points, 1).
        """
        # import pdb;pdb.set_trace()
        mask_pred = pred_masks.sigmoid()
        mask_pred = mask_pred[:, superpoints]
        seg_map = mask_pred.argmax(0)
        return seg_map

    def predict_by_feat(self, out, superpoints, scene_names):
        """Predict instance masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).
            scene_names (List[string]): of len 1, which contain scene name.

        Returns:
            List[PointData]: of len 1 with `pts_instance_mask`, 
                `instance_labels`, `instance_scores`.
        """
        pred_labels = out['cls_preds'][0]
        pred_masks = out['masks'][0]
        pred_scores = out['scores'][0]
        sem_preds = out['sem_preds'][0]

        inst_res = self.pred_inst(pred_masks,
                                  pred_scores,
                                  pred_labels,
                                  superpoints, self.test_cfg.inst_score_thr)
        sem_res = self.pred_sem(sem_preds,
                                superpoints)
        pts_semantic_mask = [sem_res.cpu().numpy()]
        pts_instance_mask = [inst_res[0].cpu().bool().numpy()]

        return [
            PointData(
                pts_semantic_mask=pts_semantic_mask,
                pts_instance_mask=pts_instance_mask,
                instance_labels=inst_res[1].cpu().numpy(),
                instance_scores=inst_res[2].cpu().numpy())]




@MODELS.register_module()
class InstanceOnlySpformer3D(Base3DDetector):
    r"""InstanceOnlySpformer3D for training on different datasets jointly.

    Args:
        in_channels (int): Number of input channels.
        num_channels (int): Number of output channels.
        voxel_size (float): Voxel size.
        num_classes_1dataset (int): Number of classes in the first dataset.
        prefix_1dataset (string): Prefix for the first dataset.
        min_spatial_shape (int): Minimal shape for spconv tensor.
        backbone (ConfigDict): Config dict of the backbone.
        decoder (ConfigDict): Config dict of the decoder.
        criterion (ConfigDict): Config dict of the criterion.
        train_cfg (dict, optional): Config dict of training hyper-parameters.
            Defaults to None.
        test_cfg (dict, optional): Config dict of test hyper-parameters.
            Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or ConfigDict, optional): the config to control the
            initialization. Defaults to None.
    """

    def __init__(self,
                 in_channels,
                 num_channels,
                 voxel_size,
                 num_classes_1dataset,
                 prefix_1dataset,
                 min_spatial_shape,
                 backbone=None,
                 decoder=None,
                 criterion=None,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 init_cfg=None):
        super(InstanceOnlySpformer3D, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.num_classes_1dataset = num_classes_1dataset 
        
        self.prefix_1dataset = prefix_1dataset 
        
        self.unet = MODELS.build(backbone)
        self.decoder = MODELS.build(decoder)
        self.criterion = MODELS.build(criterion)
        self.voxel_size = voxel_size
        self.min_spatial_shape = min_spatial_shape
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._init_layers(in_channels, num_channels)
    
    def _init_layers(self, in_channels, num_channels):
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels,
                num_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key='subm1'))
        self.output_layer = spconv.SparseSequential(
            torch.nn.BatchNorm1d(num_channels, eps=1e-4, momentum=0.1),
            torch.nn.ReLU(inplace=True))

    def extract_feat(self, x):
        """Extract features from sparse tensor.

        Args:
            x (SparseTensor): Input sparse tensor of shape
                (n_points, in_channels).

        Returns:
            List[Tensor]: of len batch_size,
                each of shape (n_points_i, n_channels).
        """
        x = self.input_conv(x)
        x, _ = self.unet(x)
        x = self.output_layer(x)
        out = []
        for i in x.indices[:, 0].unique():
            out.append(x.features[x.indices[:, 0] == i])
        return out

    def collate(self, points, elastic_points=None):
        """Collate batch of points to sparse tensor.

        Args:
            points (List[Tensor]): Batch of points.
            quantization_mode (SparseTensorQuantizationMode): Minkowski
                quantization mode. We use random sample for training
                and unweighted average for inference.

        Returns:
            TensorField: Containing features and coordinates of a
                sparse tensor.
        """
        # 
        if elastic_points is None:
            coordinates, features = ME.utils.batch_sparse_collate(
                [((p[:, :3] - p[:, :3].min(0)[0]) / self.voxel_size,
                  # s rgb or rgb
                  torch.hstack((p[:, 3:], p[:, :3] - p[:, :3].mean(0))))
                 for p in points])
        else:
            coordinates, features = ME.utils.batch_sparse_collate(
                [((el_p - el_p.min(0)[0]),
                  # s rgb or rgb
                  torch.hstack((p[:, 3:], p[:, :3] - p[:, :3].mean(0))))
                #   torch.hstack((p[:, 3:], p[:, :3])))
                  # torch.hstack((p[:, 3:],)))
                 for el_p, p in zip(elastic_points, points)])
        # 
        spatial_shape = torch.clip(
            coordinates.max(0)[0][1:] + 1, self.min_spatial_shape)
        field = ME.TensorField(features=features, coordinates=coordinates)
        tensor = field.sparse()
        coordinates = tensor.coordinates
        features = tensor.features
        inverse_mapping = field.inverse_mapping(tensor.coordinate_map_key)
        return coordinates, features, inverse_mapping, spatial_shape

    def _forward(*args, **kwargs):
        """Implement abstract method of Base3DDetector."""
        pass

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Calculate losses from a batch of inputs dict and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instances_3d` and `gt_sem_seg_3d`.
        Returns:
            dict: A dictionary of loss components.
        """
        
        coordinates, features, inverse_mapping, spatial_shape = self.collate(
            batch_inputs_dict['points'],
            batch_inputs_dict.get('elastic_coords', None))
        # 
        x = spconv.SparseConvTensor(
            features, coordinates, spatial_shape, len(batch_data_samples))

        x = self.extract_feat(x)
        # import pdb;pdb.set_trace()
        scene_names = []
        for i in range(len(batch_data_samples)):
           scene_names.append(batch_data_samples[i].lidar_path)
        # 
        x = self.decoder(x, scene_names)

        sp_gt_instances = []
        for i in range(len(batch_data_samples)):
            voxel_superpoints = inverse_mapping[coordinates[:, 0][inverse_mapping] == i]
            voxel_superpoints = torch.unique(
                voxel_superpoints, return_inverse=True)[1]
            inst_mask = batch_data_samples[i].gt_pts_seg.pts_instance_mask
            assert voxel_superpoints.shape == inst_mask.shape
            batch_data_samples[i].gt_instances_3d.sp_masks = \
                S3DISOneFormer3D.get_gt_inst_masks(inst_mask, voxel_superpoints)
            sp_gt_instances.append(batch_data_samples[i].gt_instances_3d)

        loss = self.criterion(x, sp_gt_instances)
        return loss
    
    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Predict results from a batch of inputs and data samples with post-
        processing.
        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instance_3d` and `gt_sem_seg_3d`.
        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input samples. Each Det3DDataSample contains 'pred_pts_seg'.
            And the `pred_pts_seg` contains following keys.
                - instance_scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - instance_labels (Tensor): Labels of instances, has a shape
                    (num_instances, )
                - pts_instance_mask (Tensor): Instance mask, has a shape
                    (num_points, num_instances) of type bool.
        """
        
        coordinates, features, inverse_mapping, spatial_shape = self.collate(
            batch_inputs_dict['points'])
        x = spconv.SparseConvTensor(
            features, coordinates, spatial_shape, len(batch_data_samples))

        x = self.extract_feat(x)

        scene_names = []
        for i in range(len(batch_data_samples)):
            scene_names.append(batch_data_samples[i].lidar_path)
        x = self.decoder(x, scene_names)

        results_list = self.predict_by_feat(x, inverse_mapping, scene_names)

        for i, data_sample in enumerate(batch_data_samples):
            data_sample.pred_pts_seg = results_list[i]
        return batch_data_samples

    def predict_by_feat(self, out, superpoints, scene_names):
        """Predict instance masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).
            scene_names (List[string]): of len 1, which contain scene name.

        Returns:
            List[PointData]: of len 1 with `pts_instance_mask`, 
                `instance_labels`, `instance_scores`.
        """
        pred_labels = out['cls_preds']
        pred_masks = out['masks']
        pred_scores = out['scores']
        scene_name = scene_names[0]

        scores = F.softmax(pred_labels[0], dim=-1)[:, :-1]
        scores *= pred_scores[0]

        # if self.prefix_1dataset in scene_name:
        labels = torch.arange(
            self.num_classes_1dataset,
            device=scores.device).unsqueeze(0).repeat(
                self.decoder.num_queries_1dataset,  
                1).flatten(0, 1)        
        # else:
        #     raise RuntimeError(f'Invalid scene name "{scene_name}".')
        
        scores, topk_idx = scores.flatten(0, 1).topk(
            self.test_cfg.topk_insts, sorted=False)
        labels = labels[topk_idx]

        # if self.prefix_1dataset in scene_name:
        topk_idx = torch.div(topk_idx, self.num_classes_1dataset, 
                                rounding_mode='floor')     
        # else:
        #     raise RuntimeError(f'Invalid scene name "{scene_name}".')
        
        mask_pred = pred_masks[0]
        mask_pred = mask_pred[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()
        if self.test_cfg.get('obj_normalization', None):
            mask_pred_thr = mask_pred_sigmoid > \
                self.test_cfg.obj_normalization_thr
            mask_scores = (mask_pred_sigmoid * mask_pred_thr).sum(1) / \
                (mask_pred_thr.sum(1) + 1e-6)
            scores = scores * mask_scores

        if self.test_cfg.get('nms', None):
            kernel = self.test_cfg.matrix_nms_kernel
            scores, labels, mask_pred_sigmoid, _ = mask_matrix_nms(
                mask_pred_sigmoid, labels, scores, kernel=kernel)

        mask_pred = mask_pred_sigmoid > self.test_cfg.sp_score_thr
        mask_pred = mask_pred[:, superpoints]
        # score_thr
        score_mask = scores > self.test_cfg.score_thr
        scores = scores[score_mask]
        labels = labels[score_mask]
        mask_pred = mask_pred[score_mask]

        # npoint_thr
        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum > self.test_cfg.npoint_thr
        scores = scores[npoint_mask]
        labels = labels[npoint_mask]
        mask_pred = mask_pred[npoint_mask]

        return [
            PointData(
                pts_instance_mask=mask_pred,
                instance_labels=labels,
                instance_scores=scores)
        ]
