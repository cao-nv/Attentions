import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb


class SE(nn.Module):
    name = 'SE'
    
    def __init__(self, num_channels, reduce_rate=16):
        super(SE, self).__init__()
        self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(num_channels, num_channels/reduce_rate, bias=False)
        self.fc2 = nn.Linear(num_channels/reduce_rate, num_channels, bias=False)

    def forward(self, x):
        se = self.global_pooling(x)
        se = se.view(se.size(0), -1)
        se = self.fc1(se)
        se = F.relu(se)
        se = self.fc2(se)
        se = torch.sigmoid(se)
        se = se.view(se.size(0), -1, 1, 1)
        se = se * x
        return se
    

class BAMspatial(nn.Module):
    name = 'BAMspatial'
    
    def __init__(self, num_channels, reduce_rate=16, dilation_rate=4, stand_alone=True):
        super(BAMspatial, self).__init__()
        self.name = 'BAMspatial'
        self.stand_alone = stand_alone
        self.conv1 = nn.Conv2d(num_channels, num_channels/reduce_rate, 1)
        self.conv2 = nn.Conv2d(num_channels/reduce_rate, num_channels/reduce_rate, 3, dilation=dilation_rate, padding=dilation_rate)
        self.conv3 = nn.Conv2d(num_channels/reduce_rate, num_channels/reduce_rate, 3, dilation=dilation_rate, padding=dilation_rate)
        self.conv4 = nn.Conv2d(num_channels/reduce_rate, 1, 1)
        self.bn = nn.BatchNorm2d(1)
        
    def forward(self, x):
        spatial = self.conv1(x)
        spatial = self.conv2(spatial)
        spatial = self.conv3(spatial)
        spatial = self.conv4(spatial)
        spatial = self.bn(spatial)
        if self.stand_alone:
            spatial = torch.sigmoid(spatial)
            spatial = spatial * x
        return spatial
    

class BAMchannel(nn.Module):
    name = 'BAMchannel'
    
    def __init__(self, num_channels, reduce_rate=16, stand_alone=True):
        super(BAMchannel, self).__init__()
        self.stand_alone = stand_alone
        self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(num_channels, num_channels/reduce_rate, bias=True)
        self.fc2 = nn.Linear(num_channels/reduce_rate, num_channels, bias=True)
        self.bn = nn.BatchNorm2d(num_channels)
        
    def forward(self, x):
        se = self.global_pooling(x)
        se = se.view(se.size(0), -1)
        se = self.fc1(se)
        se = self.fc2(se)
        se = se.view(se.size(0), -1, 1, 1)
        se = self.bn(se)
        if self.stand_alone:
            se = torch.sigmoid(se)
            se = x * se
        return se

        
class BAM(nn.Module):
    name = 'BAM'
    
    def __init__(self, num_channels, reduce_rate=16, dilation_rate=4):
        super(BAM, self).__init__()
        self.name = 'BAM'
        self.channel = BAMchannel(num_channels, reduce_rate, stand_alone=False)
        self.spatial = BAMspatial(num_channels, reduce_rate, dilation_rate, stand_alone=False)

    def forward(self, x):
        spatial_out = self.spatial(x)
        channel_out = self.channel(x)
        bam = spatial_out + channel_out
        bam = torch.sigmoid(bam)
        bam = bam * x
        return bam
    
    
class CBAMchannel(nn.Module):
    name = 'CBAMchannel'
    
    def __init__(self, num_channels, reduce_rate=16):
        super(CBAMchannel, self).__init__()
        self.name = 'CBAMchannel'
        self.AvgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.MaxPool = nn.AdaptiveMaxPool2d((1, 1))
        self.MLP1 = nn.Linear(num_channels, num_channels/reduce_rate)
        self.MLP2 = nn.Linear(num_channels/reduce_rate, num_channels)
        
    def forward(self, x):
        avgpool = self.AvgPool(x)
        avgpool = avgpool.view(avgpool.size(0), -1)
        maxpool = self.MaxPool(x)
        maxpool = maxpool.view(maxpool.size(0), -1)
        avgMLP = self.MLP1(avgpool)
        avgMLP = torch.relu(avgMLP)
        avgMLP = self.MLP2(avgMLP)
        maxMLP = self.MLP1(maxpool)
        maxMLP = torch.relu(maxMLP)
        maxMLP = self.MLP2(maxMLP)

        channels = avgMLP + maxMLP
        channels = torch.sigmoid(channels)
        channels = channels.view(channels.size(0), -1, 1, 1)
        channels = channels * x
        return channels


class CBAMspatial(nn.Module):
    name = 'CBAMspatial'
    
    def __init__(self, num_channels):
        super(CBAMspatial, self).__init__()
        self.name = 'CBAMspatial'
        self.conv = nn.Conv2d(2, 1, 7, padding=3)
        
    def forward(self, x):
        maxpoolChannel = torch.max(x, 1, keepdim=True)[0]
        avgpoolChannel = torch.mean(x, 1, keepdim=True)
        concatChannel = torch.cat((maxpoolChannel, avgpoolChannel), dim=1)
        spatial = self.conv(concatChannel)
        spatial = torch.sigmoid(spatial)
        spatial = spatial * x
        return spatial

    
class CBAM(nn.Module):
    name = 'CBAM'
    
    def __init__(self, num_channels, reduce_rate=16):
        super(CBAM, self).__init__()
        self.name = 'CBAM'
        self.channels = CBAMchannel(num_channels, reduce_rate)
        self.spatial = CBAMspatial(num_channels)
        
    def forward(self, x):
        cbam = self.channels(x)
        cbam = self.spatial(cbam)
        return cbam

