"""ResNet family with one pooled-vector gate, no stage-wise gating."""
from torch import nn
from torchvision.models import resnet18,ResNet18_Weights
class GatedResNet18(nn.Module):
    def __init__(self,pretrained=False):
        super().__init__()
        self.backbone=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.backbone.fc=nn.Identity()
        self.gate=nn.Sequential(nn.Linear(512,32,bias=False),nn.ReLU(),nn.Linear(32,512,bias=False),nn.Sigmoid())
        self.classifier=nn.Linear(512,7)
    def forward(self,x):
        z=self.backbone(x)
        return self.classifier(z*self.gate(z))
def build_model(variant='resnet18_gated',num_classes=7,pretrained=False):
    assert variant=='resnet18_gated' and num_classes==7
    return GatedResNet18(pretrained)
