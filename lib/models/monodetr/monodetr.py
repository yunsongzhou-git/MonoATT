"""
MonoDETR: Depth-aware Transformer for Monocular 3D Object Detection
"""
import torch
import torch.nn.functional as F
from torch import nn
import math
import copy

from utils import box_ops
from utils.misc import (NestedTensor, nested_tensor_from_tensor_list,
                            accuracy, get_world_size, interpolate,
                            is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .depthaware_transformer import build_depthaware_transformer
from .depth_predictor import DepthPredictor
from .depth_predictor.ddn_loss import DDNLoss
from lib.losses.focal_loss import sigmoid_focal_loss

# from .tcformer_module.tcformer_origin import TCFormer
from .tcformer_module.tcformer import TCFormer

from .tcformer_module.tcformer_layers import Block, TCBlock, OverlapPatchEmbed, CTM
from .tcformer_module.tcformer_utils import (load_checkpoint, get_root_logger, token2map)
from .tcformer_module.transformer_utils import trunc_normal_

from .heatmap.hm_utils import sigmoid_hm
from .heatmap.heatmap_coder import (
    get_transfrom_matrix,
    affine_transform,
    gaussian_radius,
    draw_umich_gaussian,
)
from .heatmap.focal_loss import FocalLoss

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MonoDETR(nn.Module):
    """ This is the MonoDETR module that performs monocualr 3D object detection """
    def __init__(self, backbone, depthaware_transformer, depth_predictor, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False, init_box=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            depthaware_transformer: depth-aware transformer architecture. See depth_aware_transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For KITTI, we recommend 50 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage MonoDETR
        """
        super().__init__()
 
        self.num_queries = num_queries
        self.depthaware_transformer = depthaware_transformer
        self.depth_predictor = depth_predictor
        hidden_dim = depthaware_transformer.d_model
        self.num_feature_levels = num_feature_levels

        # prediction heads
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value

        self.bbox_embed = MLP(hidden_dim, hidden_dim, 6, 3)
        self.dim_embed_3d = MLP(hidden_dim, hidden_dim, 3, 2)
        self.angle_embed = MLP(hidden_dim, hidden_dim, 24, 2)
        self.depth_embed = MLP(hidden_dim, hidden_dim, 2, 2)  # depth and deviation

        if init_box == True:
            nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)

        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        self.tcformer = TCFormer(img_size=160, in_chans=hidden_dim, embed_dims=[512, 1024, 2048, 4096],depths=[1, 1, 1, 1], num_stages=3)

        in_channels = backbone.num_channels[0]//2
        head_conv = in_channels//2
        self.hm_head = nn.Sequential(nn.Conv2d(in_channels,head_conv,kernel_size=3,padding=1,bias=True),nn.GroupNorm(32, head_conv),nn.ReLU(inplace=True),nn.Conv2d(head_conv,1,kernel_size=1,padding=1 // 2,bias=True)
            )

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (depthaware_transformer.decoder.num_layers + 1) if two_stage else depthaware_transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.depthaware_transformer.decoder.bbox_embed = self.bbox_embed
            self.dim_embed_3d = _get_clones(self.dim_embed_3d, num_pred)
            self.depthaware_transformer.decoder.dim_embed = self.dim_embed_3d  
            self.angle_embed = _get_clones(self.angle_embed, num_pred)
            self.depth_embed = _get_clones(self.depth_embed, num_pred)
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.dim_embed_3d = nn.ModuleList([self.dim_embed_3d for _ in range(num_pred)])
            self.angle_embed = nn.ModuleList([self.angle_embed for _ in range(num_pred)])
            self.depth_embed = nn.ModuleList([self.depth_embed for _ in range(num_pred)])
            self.depthaware_transformer.decoder.bbox_embed = None

        if two_stage:
            # hack implementation for two-stage
            self.depthaware_transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)


    def forward(self, images, calibs, img_sizes):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """
        features, pos = self.backbone(images)
        B, _, H, W = images.shape
        H, W = H//8, W//8

        src, _ = features[0].decompose()
        heatmap = sigmoid_hm(self.hm_head(src))

        import random
        img_id = random.randint(0,8000)

        features_tc = self.tcformer(features[1], pos[1], images, heatmap, img_id)
        features, pos = features[1:], pos[1:]

        
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            if l==2:
                src_2 = features_tc
                srcs_2 = self.input_proj[l+1](src_2)

                m = torch.zeros(src_2.shape[0], src_2.shape[2], src_2.shape[3]).to(torch.bool).to(src_2.device)
                mask_2 = F.interpolate(m[None].float(), size=src_2.shape[-2:]).to(torch.bool)[0]
                pos_l_2 = self.backbone[1](NestedTensor(src_2, mask_2)).to(src_2.dtype)
                srcs.append(srcs_2)
                masks.append(mask_2)
                pos.append(pos_l_2)

            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None


        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[1](features[-1].tensors)
                    #
                else:
                    src = self.input_proj[l](srcs[-1])
                m = torch.zeros(src.shape[0], src.shape[2], src.shape[3]).to(torch.bool).to(src.device)
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)
                

        query_embeds = None

        if not self.two_stage:
            query_embeds = self.query_embed.weight

        pred_depth_map_logits, depth_pos_embed, weighted_depth = self.depth_predictor(srcs, masks[1], pos[1])

        hs, init_reference, inter_references, inter_references_dim, enc_outputs_class, enc_outputs_coord_unact = self.depthaware_transformer(
            srcs, masks, pos, query_embeds, depth_pos_embed)

        outputs_coords = []
        outputs_classes = []
        outputs_3d_dims = []
        outputs_depths = []
        outputs_angles = []
      

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)

            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 6:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference

            # 3d center + 2d box
            outputs_coord = tmp.sigmoid()
            outputs_coords.append(outputs_coord)

            # classes
            outputs_class = self.class_embed[lvl](hs[lvl])
            outputs_classes.append(outputs_class)

            # 3D sizes
            size3d = inter_references_dim[lvl]
            outputs_3d_dims.append(size3d)

            # depth_geo
            box2d_height_norm = outputs_coord[:, :, 4] + outputs_coord[:, :, 5]
            box2d_height = torch.clamp(box2d_height_norm * img_sizes[:, 1: 2], min=1.0)
            depth_geo = size3d[:, :, 0] / box2d_height * calibs[:, 0, 0].unsqueeze(1)

            # depth_reg
            depth_reg = self.depth_embed[lvl](hs[lvl])

            # depth_map
            outputs_center3d = ((outputs_coord[..., :2] - 0.5) * 2).unsqueeze(2).detach()
            depth_map = F.grid_sample(
                weighted_depth.unsqueeze(1),
                outputs_center3d,
                mode='bilinear',
                align_corners=True).squeeze(1)

            # depth average + sigma
            depth_ave = torch.cat([((1. / (depth_reg[:, :, 0: 1].sigmoid() + 1e-6) - 1.) + depth_geo.unsqueeze(-1) + depth_map) / 3,
                                    depth_reg[:, :, 1: 2]], -1)
            outputs_depths.append(depth_ave)

            # angles
            outputs_angle = self.angle_embed[lvl](hs[lvl])
            outputs_angles.append(outputs_angle)

        outputs_coord = torch.stack(outputs_coords)
        outputs_class = torch.stack(outputs_classes)
        outputs_3d_dim = torch.stack(outputs_3d_dims)
        outputs_depth = torch.stack(outputs_depths)
        outputs_angle = torch.stack(outputs_angles)

        # print(outputs_3d_dim, outputs_3d_dim.shape)

        center_loc = outputs_angle[-1]

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        out['pred_3d_dim'] = outputs_3d_dim[-1]
        out['pred_depth'] = outputs_depth[-1]
        out['pred_angle'] = outputs_angle[-1]
        out['pred_depth_map_logits'] = pred_depth_map_logits
        out['pred_center_loc'] = center_loc
        out['heatmap'] = heatmap.to(outputs_class.device)
        # out['pred_center_loc'] = pred_depth_map_logits
        # center_loc = center_loc.to(outputs_class.device)
        heatmap = heatmap.to(outputs_class.device)
        

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(
                outputs_class, outputs_coord, outputs_3d_dim, outputs_angle, outputs_depth, center_loc, heatmap)

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}


        # input()
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_3d_dim, outputs_angle, outputs_depth, center_loc, heatmap):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b, 
                 'pred_3d_dim': c, 'pred_angle': d, 'pred_depth': e, 'pred_center_loc': f, 'heatmap': g}
                for a, b, c, d, e ,f, g in zip(outputs_class[:-1], outputs_coord[:-1],
                                         outputs_3d_dim[:-1], outputs_angle[:-1], outputs_depth[:-1], center_loc, heatmap)]


class SetCriterion(nn.Module):
    """ This class computes the loss for MonoDETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, focal_alpha, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.ddn_loss = DDNLoss()  # for depth map
        LOSS_ALPHA,LOSS_BETA = 2, 4
        self.focal_loss = FocalLoss(LOSS_ALPHA,LOSS_BETA)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)

        target_classes[idx] = target_classes_o.squeeze().long()

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_3dcenter(self, outputs, targets, indices, num_boxes):
        
        idx = self._get_src_permutation_idx(indices)
        src_3dcenter = outputs['pred_boxes'][:, :, 0: 2][idx]
        target_3dcenter = torch.cat([t['boxes_3d'][:, 0: 2][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_3dcenter = F.l1_loss(src_3dcenter, target_3dcenter, reduction='none')
        losses = {}
        losses['loss_center'] = loss_3dcenter.sum() / num_boxes
        return losses


    def prepare_heatmap(self, targets):
        import numpy as np

        B = len(targets)
        output_height, output_width = 96, 320
        heat_map = np.zeros([B, 1, output_height, output_width], dtype=np.float32)


        for i in range(B):
            target_box2d = targets[i]['boxes'].cpu().numpy()
            target_3dcenter = targets[i]['boxes_3d'][:, 0: 2].cpu().numpy()
            target_box2d[:,0::2], target_box2d[:,1::2] = target_box2d[:,0::2]*output_width, target_box2d[:,1::2]*output_height
            target_3dcenter[:,0], target_3dcenter[:,1] = target_3dcenter[:,0]*output_width, target_3dcenter[:,1]*output_height
            

            for j in range(len(target_box2d)):

                point_int = target_3dcenter[j].astype(int)
                box2d = target_box2d[j,:]

                # print(point_int)

                h, w = box2d[3] - box2d[1], box2d[2] - box2d[0]
                radius = gaussian_radius(h, w)
                radius = max(0, int(radius))
                heat_map[i,...] = draw_umich_gaussian(heat_map[i,...], point_int, radius)
        return torch.from_numpy(heat_map)



    def loss_heatmap(self, outputs, targets, indices, num_boxes):

        targets_heatmap = self.prepare_heatmap(targets)
        pred_heatmap = outputs['heatmap']

        device = pred_heatmap.device
        targets_heatmap = targets_heatmap.to(device)

        hm_loss = self.focal_loss(pred_heatmap, targets_heatmap)


        losses = {}
        losses['loss_heatmap'] = hm_loss
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_2dboxes = outputs['pred_boxes'][:, :, 2: 6][idx]
        target_2dboxes = torch.cat([t['boxes_3d'][:, 2: 6][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # l1
        loss_bbox = F.l1_loss(src_2dboxes, target_2dboxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        # giou
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes_3d'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        try:
            loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
                box_ops.box_cxcylrtb_to_xyxy(src_boxes),
                box_ops.box_cxcylrtb_to_xyxy(target_boxes)))
        except Expection as e:
            print(src_boxes, target_boxes)
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_depths(self, outputs, targets, indices, num_boxes):  

        idx = self._get_src_permutation_idx(indices)
        src_depths = outputs['pred_depth'][idx]
        target_depths = torch.cat([t['depth'][i] for t, (_, i) in zip(targets, indices)], dim=0).squeeze()

        depth_input, depth_log_variance = src_depths[:, 0], src_depths[:, 1] 
        depth_loss = 1.4142 * torch.exp(-depth_log_variance) * torch.abs(depth_input - target_depths) + depth_log_variance  
        losses = {}
        losses['loss_depth'] = depth_loss.sum() / num_boxes 
        return losses  
    
    def loss_dims(self, outputs, targets, indices, num_boxes):  

        idx = self._get_src_permutation_idx(indices)
        src_dims = outputs['pred_3d_dim'][idx]
        target_dims = torch.cat([t['size_3d'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        dimension = target_dims.clone().detach()
        dim_loss = torch.abs(src_dims - target_dims)
        dim_loss /= dimension
        with torch.no_grad():
            compensation_weight = F.l1_loss(src_dims, target_dims) / dim_loss.mean()
        dim_loss *= compensation_weight
        losses = {}
        losses['loss_dim'] = dim_loss.sum() / num_boxes
        return losses

    def loss_angles(self, outputs, targets, indices, num_boxes):  

        idx = self._get_src_permutation_idx(indices)
        heading_input = outputs['pred_angle'][idx]
        target_heading_cls = torch.cat([t['heading_bin'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_heading_res = torch.cat([t['heading_res'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        heading_input = heading_input.view(-1, 24)
        heading_target_cls = target_heading_cls.view(-1).long()
        heading_target_res = target_heading_res.view(-1)

        # classification loss
        heading_input_cls = heading_input[:, 0:12]
        cls_loss = F.cross_entropy(heading_input_cls, heading_target_cls, reduction='none')

        # regression loss
        heading_input_res = heading_input[:, 12:24]
        cls_onehot = torch.zeros(heading_target_cls.shape[0], 12).cuda().scatter_(dim=1, index=heading_target_cls.view(-1, 1), value=1)
        heading_input_res = torch.sum(heading_input_res * cls_onehot, 1)
        reg_loss = F.l1_loss(heading_input_res, heading_target_res, reduction='none')
        
        angle_loss = cls_loss + reg_loss
        losses = {}
        losses['loss_angle'] = angle_loss.sum() / num_boxes 
        return losses

    def loss_depth_map(self, outputs, targets, indices, num_boxes):
        depth_map_logits = outputs['pred_depth_map_logits']

        num_gt_per_img = [len(t['boxes']) for t in targets]
        gt_boxes2d = torch.cat([t['boxes'] for t in targets], dim=0) * torch.tensor([80, 24, 80, 24], device='cuda')
        gt_boxes2d = box_ops.box_cxcywh_to_xyxy(gt_boxes2d)
        gt_center_depth = torch.cat([t['depth'] for t in targets], dim=0).squeeze(dim=1)

        losses = dict()
        losses["loss_depth_map"] = self.ddn_loss(
            depth_map_logits, gt_boxes2d, num_gt_per_img, gt_center_depth)
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'depths': self.loss_depths,
            'dims': self.loss_dims,
            'angles': self.loss_angles,
            'center': self.loss_3dcenter,
            'depth_map': self.loss_depth_map,
            'heatmap': self.loss_heatmap

        }

        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'depth_map':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(cfg):
    # backbone
    backbone = build_backbone(cfg)

    # detr
    depthaware_transformer = build_depthaware_transformer(cfg)

    # depth prediction module
    depth_predictor = DepthPredictor(cfg)

    model = MonoDETR(
        backbone,
        depthaware_transformer,
        depth_predictor,
        num_classes=cfg['num_classes'],
        num_queries=cfg['num_queries'],
        aux_loss=cfg['aux_loss'],
        num_feature_levels=cfg['num_feature_levels'],
        with_box_refine=cfg['with_box_refine'],
        two_stage=cfg['two_stage'],
        init_box=cfg['init_box'])

    # matcher
    matcher = build_matcher(cfg)

    # loss
    weight_dict = {'loss_ce': cfg['cls_loss_coef'], 'loss_bbox': cfg['bbox_loss_coef']}
    weight_dict['loss_giou'] = cfg['giou_loss_coef']
    weight_dict['loss_dim'] = cfg['dim_loss_coef']
    weight_dict['loss_angle'] = cfg['angle_loss_coef']
    weight_dict['loss_depth'] = cfg['depth_loss_coef']
    weight_dict['loss_center'] = cfg['3dcenter_loss_coef']
    weight_dict['loss_depth_map'] = cfg['depth_map_loss_coef']
    weight_dict['loss_heatmap'] = cfg['depth_map_loss_coef']

    # TODO this is a hack
    if cfg['aux_loss']:
        aux_weight_dict = {}
        for i in range(cfg['dec_layers'] - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality', 'depths', 'dims', 'angles', 'center', 'depth_map', 'heatmap']
    
    criterion = SetCriterion(
        cfg['num_classes'],
        matcher=matcher,
        weight_dict=weight_dict,
        focal_alpha=cfg['focal_alpha'],
        losses=losses)
    device = torch.device(cfg['device'])
    criterion.to(device)
    
    return model, criterion
