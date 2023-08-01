import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class DepthwiseMultiScaleFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels, scales):
        super(DepthwiseMultiScaleFeatureExtractor, self).__init__()
        self.scales = scales
        self.feature_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=s, dilation=s, groups=in_channels),
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for s in scales
        ])
    
    def forward(self, x):
        return [extractor(x) for extractor in self.feature_extractors]


class ResidualAdaptiveFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualAdaptiveFusion, self).__init__()
        self.project = None
        if in_channels != out_channels:
            self.project = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x, features, attentions):
        fused_features = sum([attention * feature for feature, attention in zip(features, attentions)])
        if self.project:
            x = self.project(x)
        return x + fused_features


class SophisticatedAttention(nn.Module):
    def __init__(self, in_channels):
        super(SophisticatedAttention, self).__init__()
        # Self Attention
        self.W_Q = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.W_K = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.W_V = nn.Conv2d(in_channels, in_channels, 1)
        
        # Channel-wise Attention
        self.fc1 = nn.Linear(in_channels, in_channels // 16)
        self.fc2 = nn.Linear(in_channels // 16, in_channels)

    def forward(self, x):
        # Self Attention
        b, c, h, w = x.size()
        Q = self.W_Q(x).view(b, -1, h * w).permute(0, 2, 1)
        K = self.W_K(x).view(b, -1, h * w)
        V = self.W_V(x).view(b, -1, h * w)
        attention = torch.bmm(Q, K) / math.sqrt(c)
        attention = nn.Softmax(dim=-1)(attention)
        self_attention_map = torch.bmm(V, attention.permute(0, 2, 1))
        self_attention_map = self_attention_map.view(b, c, h, w)
        
        # Channel-wise Attention
        avg_pool = torch.mean(x, dim=[2, 3])
        channel_attention = self.fc2(F.relu(self.fc1(avg_pool)))
        channel_attention = torch.sigmoid(channel_attention).unsqueeze(2).unsqueeze(3)
        
        return self_attention_map * channel_attention

class RefinedAMSAM(nn.Module):
    def __init__(self, in_channels, out_channels, scales=[1, 2, 4]):
        super(RefinedAMSAM, self).__init__()
        self.feature_extractor = DepthwiseMultiScaleFeatureExtractor(in_channels, out_channels, scales)
        self.attention = nn.ModuleList([SophisticatedAttention(out_channels) for _ in scales])
        self.fusion = ResidualAdaptiveFusion(in_channels, out_channels)

    def forward(self, x):
        features = self.feature_extractor(x)
        attended_features = [att(feature) for att, feature in zip(self.attention, features)]
        return self.fusion(x, attended_features, attended_features)

class NoAttentionAMSAM(nn.Module):
    def __init__(self, in_channels, out_channels, scales=[1, 2, 4]):
        super(NoAttentionAMSAM, self).__init__()
        self.feature_extractor = DepthwiseMultiScaleFeatureExtractor(in_channels, out_channels, scales)
        self.fusion = ResidualAdaptiveFusion(in_channels, out_channels)

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.fusion(x, features, features)

# Adjust the network to use NoAttentionAMSAM
class NoAttentionAMSAMNet(nn.Module):
    def __init__(self, in_channels, num_classes=128):
        super(NoAttentionAMSAMNet, self).__init__()
        self.no_attention_amsam = NoAttentionAMSAM(in_channels, 32)  # Output channels set to 32 for simplicity
        self.fc1 = nn.Linear(819200, 256)  # Adjusted for 32x32 image size
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.no_attention_amsam(x)
        x = x.view(x.size(0), -1)
        x = nn.ReLU()(self.fc1(x))
        return self.fc2(x)
    
    
######################################################################################################
# single scale
######################################################################################################
# Single Scale Ablation

class SingleScaleAMSAM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SingleScaleAMSAM, self).__init__()
        self.feature_extractor = DepthwiseMultiScaleFeatureExtractor(in_channels, out_channels, scales=[2])
        self.attention = nn.ModuleList([SophisticatedAttention(out_channels) for _ in [2]])
        self.fusion = ResidualAdaptiveFusion(in_channels, out_channels)

    def forward(self, x):
        features = self.feature_extractor(x)
        attended_features = [att(feature) for att, feature in zip(self.attention, features)]
        return self.fusion(x, attended_features, attended_features)

# Adjust the network to use SingleScaleAMSAM
class SingleScaleAMSAMNet(nn.Module):
    def __init__(self, in_channels, num_classes=2):
        super(SingleScaleAMSAMNet, self).__init__()
        self.single_scale_amsam = SingleScaleAMSAM(in_channels, 32)  # Output channels set to 32 for simplicity
        self.fc1 = nn.Linear(131072, 256)  # Adjusted for 32x32 image size
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.single_scale_amsam(x)
        x = x.view(x.size(0), -1)
        x = nn.ReLU()(self.fc1(x))
        return self.fc2(x)
    
    
    

#####################################################################################################
# Two Scales Ablation
#####################################################################################################
class TwoScalesAMSAM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TwoScalesAMSAM, self).__init__()
        self.feature_extractor = DepthwiseMultiScaleFeatureExtractor(in_channels, out_channels, scales=[1, 2])
        self.attention = nn.ModuleList([SophisticatedAttention(out_channels) for _ in [1, 2]])
        self.fusion = ResidualAdaptiveFusion(in_channels, out_channels)

    def forward(self, x):
        features = self.feature_extractor(x)
        attended_features = [att(feature) for att, feature in zip(self.attention, features)]
        return self.fusion(x, attended_features, attended_features)

# Adjust the network to use TwoScalesAMSAM
class TwoScalesAMSAMNet(nn.Module):
    def __init__(self, in_channels, num_classes=2):
        super(TwoScalesAMSAMNet, self).__init__()
        self.two_scales_amsam = TwoScalesAMSAM(in_channels, 32)  # Output channels set to 32 for simplicity
        self.fc1 = nn.Linear(131072, 256)  # Adjusted for 32x32 image size
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.two_scales_amsam(x)
        x = x.view(x.size(0), -1)
        x = nn.ReLU()(self.fc1(x))
        return self.fc2(x)
    
    
#####################################################################################################
# full scale
#####################################################################################################
class SimpleAMSAMNet(nn.Module):
    def __init__(self, in_channels, num_classes=2):
        super(SimpleAMSAMNet, self).__init__()
        self.refined_amsam = RefinedAMSAM(in_channels, 32)  # Output channels set to 32 for simplicity
        self.fc1 = nn.Linear(32 * 64 * 64, 256)  # Assuming 64x64 image size
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.refined_amsam(x)
        x = x.view(x.size(0), -1)
        x = nn.ReLU()(self.fc1(x))
        return self.fc2(x)
    
    
#####################################################################################################
# visualization
#####################################################################################################
class RefinedAMSAMWithAttentionExtraction(RefinedAMSAM):
    def forward(self, x):
        features = self.feature_extractor(x)
        attended_features = [att(feature) for att, feature in zip(self.attention, features)]
        
        # Extracting attention maps
        attention_maps = [feature / attended_feature for feature, attended_feature in zip(features, attended_features)]
        
        return self.fusion(x, attended_features, attended_features), attention_maps
