import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from torch.cuda.amp import autocast as autocast
from .cwt import CWTransformer, CONFIGS


class ClusVPR(nn.Module):
    """CWTVLAD module implementation"""

    def __init__(self, num_clusters=64, dim=512, alpha=100.0, expansion=2, group=8, normalize_input=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(ClusVPR, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.expansion = expansion
        self.group = group
        self.normalize_input = normalize_input

        self.linear1 = nn.Linear(dim, expansion * dim, bias=False)
        self.GeM_pool = GeMPooling(self.H * self.W, pool_size=(1, expansion * dim // group), init_norm=3.0)

        self.conv = nn.Conv2d(expansion * dim // group, num_clusters, kernel_size=(1, 1), bias=False)
        self.centroids = nn.Parameter(torch.rand(num_clusters, expansion * dim // group), requires_grad=True)

        self.clsts = None
        self.traindescs = None

        self.H = 30
        self.W = 40
        self.vit_layer = nn.Sequential(
            CWTNet(self.dim),
            CWTNet(self.dim),
            CWTNet(self.dim),
            CWTNet(self.dim),
        )

    def _init_params(self):
        clstsAssign = self.clsts / np.linalg.norm(self.clsts, axis=1, keepdims=True)
        dots = np.dot(clstsAssign, self.traindescs.T)
        dots.sort(0)
        dots = dots[::-1, :] # sort, descending

        self.alpha = (-np.log(0.01) / np.mean(dots[0,:] - dots[1,:])).item()
        self.centroids.data.copy_(torch.from_numpy(self.clsts))
        self.conv.weight.data.copy_(torch.from_numpy(self.alpha*clstsAssign).unsqueeze(2).unsqueeze(3))

    def forward(self, x):
        N, _, H, W = x.shape

        if (not self.training):
            _, _, H, W = x.shape
            if H != self.H or W != self.W:
                Trans = T.Resize((self.H, self.W))
                x = Trans(x)
            x = self.vit_layer(x)
            if H != self.H or W != self.W:
                Trans = T.Resize((H, W))
                x = Trans(x)

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        x = self.linear1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        dim = self.expansion * self.dim // self.group
        x_group = x.view(N, self.group, dim, H, W)

        attention = self.GeM_pool(x.view(N, self.group, dim, -1)).squeeze()
        attention = F.sigmoid(attention)

        # soft-assignment
        soft_assign = self.conv(x_group.reshape(-1, dim, H, W)).view(N, self.group, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=2)

        x_group_flatten = x_group.view(N, self.group, dim, -1)

        # calculate residuals to each clusters in one loop
        residual = x_group_flatten.expand(self.num_clusters, -1, -1, -1, -1).permute(1, 2, 0, 3, 4) - \
            self.centroids.expand(x_group_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0).unsqueeze(0)
        residual *= soft_assign.unsqueeze(3) * attention.unsqueeze(2).unsqueeze(2)
        vlad = residual.sum(dim=-1).sum(dim=1)

        return vlad


class CWTNet(nn.Module):
    def __init__(self, dim):
        super(CWTNet, self).__init__()
        self.dim = dim
        self.branch_dim = dim // 2

        self.local_dwconvl = nn.Conv2d(self.branch_dim, self.branch_dim, kernel_size=3, stride=1, padding=1, groups=self.branch_dim, bias=False)
        self.local_dwconvg = nn.Conv2d(self.branch_dim, self.branch_dim, kernel_size=3, stride=1, padding=1, groups=self.branch_dim, bias=False)
        self.global_dwconv = nn.Conv2d(self.branch_dim, self.branch_dim, kernel_size=3, stride=1, padding=1, groups=self.branch_dim, bias=False)
        config = CONFIGS['vgg']
        self.H = 30
        self.W = 40
        self.global_cwt = CWTransformer(config, (self.H, self.W))
        self.local_bn = nn.BatchNorm2d(self.branch_dim)
        self.global_bn = nn.BatchNorm2d(self.branch_dim)
        self.relu = nn.ReLU()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x_local = x[:,:self.branch_dim,:,:]
        x_global = x[:,self.branch_dim:,:,:]
        x_local_out = self.local_dwconvl(x_local) + self.global_dwconv(x_global)
        x_global_out = self.local_dwconvg(x_local) + self.global_cwt(x_global)
        x_local_out = self.relu(self.local_bn(x_local_out))
        x_global_out = self.relu(self.local_bn(x_global_out))
        x = torch.cat((x_local_out, x_global_out), dim=1)
        return x


class GeMPooling(nn.Module):
    def __init__(self, feature_size, pool_size=(1, 128), init_norm=3.0, eps=1e-6, normalize=False, **kwargs):
        super(GeMPooling, self).__init__(**kwargs)
        self.feature_size = feature_size  # Final layer channel size, the pow calc at -1 axis
        self.pool_size = pool_size
        self.init_norm = init_norm
        self.p = torch.nn.Parameter(torch.ones(self.feature_size) * self.init_norm, requires_grad=True)
        self.p.data.fill_(init_norm)
        self.normalize = normalize
        self.avg_pooling = nn.AvgPool2d((self.pool_size[0], self.pool_size[1]))
        self.eps = eps

    def forward(self, features):
        # filter invalid value: set minimum to 1e-6
        # features-> (B, H, W, C)
        features = features.clamp(min=self.eps).pow(self.p)
        features = features.permute((0, 3, 1, 2))
        features = self.avg_pooling(features)
        features = features.permute((0, 2, 3, 1))
        features = torch.pow(features, (1.0 / self.p))
        # unit vector
        if self.normalize:
            features = F.normalize(features, dim=-1, p=2)
        return features


class EmbedNet(nn.Module):
    def __init__(self, base_model, net_vlad):
        super(EmbedNet, self).__init__()
        self.base_model = base_model
        self.net_vlad = net_vlad

    def _init_params(self):
        self.base_model._init_params()
        self.net_vlad._init_params()

    def forward(self, x):
        pool_x, x = self.base_model(x)
        vlad_x = self.net_vlad(x)

        # [IMPORTANT] normalize
        vlad_x = F.normalize(vlad_x, p=2, dim=2)  # intra-normalization
        vlad_x = vlad_x.view(x.size(0), -1)  # flatten
        vlad_x = F.normalize(vlad_x, p=2, dim=1)  # L2 normalize

        return pool_x, vlad_x


class EmbedRegionTrans(nn.Module):
    def __init__(self, base_model, net_vlad, tuple_size=1):
        super(EmbedRegionTrans, self).__init__()
        self.base_model = base_model
        self.net_vlad = net_vlad
        self.tuple_size = tuple_size

    def _init_params(self):
        self.base_model._init_params()
        self.net_vlad._init_params()

    def _compute_patch_multi_scale(self, feature_A, feature_B):
        # feature_A: B*C*H*W
        # feature_B: (B*(1+neg_num))*C*H*W

        def reshape(x):
            # re-arrange local features for aggregating multi-scale patches
            N, C, H, W = x.size()
            x = x.view(N, C, 3, int(H/3), 4, int(W/4))
            x = x.permute(0,1,2,4,3,5).contiguous()
            x = x.view(N, C, -1, int(H/3), int(W/4))
            return x

        feature_A = reshape(feature_A)
        feature_B = reshape(feature_B)

        # computer s-patches features
        def aggregate_s_patches(x):
            N, C, B, H, W = x.size()
            x = x.permute(0,2,1,3,4).contiguous()
            x = x.view(-1,C,H,W)
            vlad_x = self.net_vlad(x) # (N*B)*64*512
            _, cluster_num, feat_dim = vlad_x.size()
            vlad_x = vlad_x.view(N,B,cluster_num,feat_dim)
            return vlad_x
    
        vlad_A_s_patches = aggregate_s_patches(feature_A)
        vlad_B_s_patches= aggregate_s_patches(feature_B)

        # computer m-patches features
        def s_to_m_patches(vlad_x):
            return torch.stack((vlad_x[:,0]+vlad_x[:,1]+vlad_x[:,4]+vlad_x[:,5], 
                                vlad_x[:,1]+vlad_x[:,2]+vlad_x[:,5]+vlad_x[:,6],
                                vlad_x[:,2]+vlad_x[:,3]+vlad_x[:,6]+vlad_x[:,7],
                                vlad_x[:,4]+vlad_x[:,5]+vlad_x[:,8]+vlad_x[:,9],
                                vlad_x[:,5]+vlad_x[:,6]+vlad_x[:,9]+vlad_x[:,10],
                                vlad_x[:,6]+vlad_x[:,7]+vlad_x[:,10]+vlad_x[:,11]), dim=1).contiguous()

        # computer l-patches features
        def s_to_l_patches(vlad_x):
            return torch.stack((vlad_x[:,0]+vlad_x[:,1]+vlad_x[:,4]+vlad_x[:,5]+vlad_x[:,8]+vlad_x[:,9], 
                                vlad_x[:,2]+vlad_x[:,3]+vlad_x[:,6]+vlad_x[:,7]+vlad_x[:,10]+vlad_x[:,11]), dim=1).contiguous()

        vlad_A_m_patches = s_to_m_patches(vlad_A_s_patches)
        vlad_B_m_patches = s_to_m_patches(vlad_B_s_patches)

        vlad_A_l_patches = s_to_l_patches(vlad_A_s_patches)
        vlad_B_l_patches = s_to_l_patches(vlad_B_s_patches)

        # computer global-image features
        def s_to_global(vlad_x):
            return vlad_x.sum(1).unsqueeze(1).contiguous()

        vlad_A_global = s_to_global(vlad_A_s_patches)
        vlad_B_global = s_to_global(vlad_B_s_patches)
    
        def norm(vlad_x):
            N, B, C, _ = vlad_x.size()
            vlad_x = F.normalize(vlad_x, p=2, dim=3)  # intra-normalization
            vlad_x = vlad_x.view(N, B, -1)  # flatten
            vlad_x = F.normalize(vlad_x, p=2, dim=2)  # L2 normalize
            return vlad_x

        vlad_A = torch.cat((vlad_A_global, vlad_A_l_patches, vlad_A_m_patches, vlad_A_s_patches), dim=1)
        vlad_B = torch.cat((vlad_B_global, vlad_B_l_patches, vlad_B_m_patches, vlad_B_s_patches), dim=1)

        vlad_A = norm(vlad_A)
        vlad_B = norm(vlad_B)

        _, B, L = vlad_B.size()
        vlad_A = vlad_A.view(self.tuple_size,-1,B,L)
        vlad_B = vlad_B.view(self.tuple_size,-1,B,L)

        score = torch.bmm(vlad_A.expand_as(vlad_B).view(-1,B,L), vlad_B.view(-1,B,L).transpose(1,2))
        score = score.view(self.tuple_size,-1,B,B)

        return score, vlad_A, vlad_B

    def _forward_train(self, x):
        B, C, H, W = x.size()
        x = x.view(self.tuple_size, -1, C, H, W)

        anchors = x[:, 0].unsqueeze(1).contiguous().view(-1,C,H,W) # B*C*H*W
        pairs = x[:, 1:].view(-1,C,H,W) # (B*(1+neg_num))*C*H*W

        return self._compute_patch_multi_scale(anchors, pairs)

    def forward(self, x):
        pool_x, x = self.base_model(x)

        if (not self.training):
            vlad_x = self.net_vlad(x)
            # normalize
            vlad_x = F.normalize(vlad_x, p=2, dim=2)  # intra-normalization
            vlad_x = vlad_x.view(x.size(0), -1)  # flatten
            vlad_x = F.normalize(vlad_x, p=2, dim=1)  # L2 normalize
            return pool_x, vlad_x

        x = self.net_vlad.vit_layer(x)

        return self._forward_train(x)