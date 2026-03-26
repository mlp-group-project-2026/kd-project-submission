from torch import nn
import torch

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # pt = p if target=1, 1-p otherwise
        focal_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class WeightedFocalBCELoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, pos_weights=None):
        super().__init__()
        self.focal = FocalLoss(alpha, gamma)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        self.alpha = 0.7  # Weight between focal and BCE
        
    def forward(self, inputs, targets):
        focal_loss = self.focal(inputs, targets)
        bce_loss = self.bce(inputs, targets)
        return self.alpha * focal_loss + (1 - self.alpha) * bce_loss

# Calculate better class weights
def calculate_effective_weights(df, label_cols, beta=0.999):
    """Calculate more robust class weights using effective number of samples"""
    n = len(df)
    weights = []
    for col in label_cols:
        pos_count = df[col].sum()
        neg_count = n - pos_count
        
        # Effective number of samples (from Class-Balanced Loss paper)
        eff_pos = (1 - beta**pos_count) / (1 - beta) if pos_count > 0 else 0
        eff_neg = (1 - beta**neg_count) / (1 - beta) if neg_count > 0 else 0
        
        weight = eff_neg / (eff_pos + 1e-8)  # Avoid division by zero
        weights.append(min(weight, 10.0))  # Cap extreme weights
    
    return torch.tensor(weights).float()