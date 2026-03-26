import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss

class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.1, temperature=3.0, use_focal=False, focal_gamma=2.0):
        """
        Knowledge Distillation Loss for multi-label classification
        with optional correlation-aware distillation and focal loss.
        
        Args:
            alpha: Weight for KL distillation loss (1-alpha-beta for hard label loss)
                   0.0 = only hard labels, higher = more teacher knowledge
                   Recommended: 0.3-0.7
            beta: Weight for correlation loss (matching feature correlations)
                  0.0 = no correlation matching, higher = more correlation matching
                  Recommended: 0.0-0.3
            temperature: Temperature for softening probability distributions
                         Higher values create softer distributions
                         Recommended: 3.0-5.0
            use_focal: If True, use focal loss instead of BCE for the hard label loss.
                       Focal loss down-weights easy examples and focuses on hard ones,
                       which helps with class imbalance. Default: False
            focal_gamma: Focusing parameter for focal loss. Higher values increase
                         the focus on hard-to-classify examples.
                         0.0 = equivalent to standard BCE
                         Recommended: 1.0-3.0, Default: 2.0
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.use_focal = use_focal
        self.focal_gamma = focal_gamma
        self.ce_loss = nn.BCEWithLogitsLoss(reduction="mean")

    def focal_loss(self, logits, labels):
        """
        Compute sigmoid focal loss for multi-label classification using torchvision.
        
        Args:
            logits: Raw model outputs (batch_size, num_classes)
            labels: Binary ground truth labels (batch_size, num_classes)
            
        Returns:
            Scalar focal loss value
        """
        return sigmoid_focal_loss(logits, labels, gamma=self.focal_gamma, alpha=-1, reduction="mean")

    def correlation_loss(self, student_probs, teacher_probs):
        """
        Compute correlation-aware distillation loss.
        Matches the correlation structure between classes in student and teacher.
        
        Args:
            student_probs: Softened probabilities from student model (batch_size, num_classes)
            teacher_probs: Softened probabilities from teacher model (batch_size, num_classes)
        
        Returns:
            MSE loss between correlation matrices
        """
        # Compute correlation matrices (num_classes x num_classes)
        # Using centered probabilities
        
        # Removing standardization (z-score), keeping centering for covariance:
        z_s = student_probs - student_probs.mean(dim=0, keepdim=True)
        z_t = teacher_probs - teacher_probs.mean(dim=0, keepdim=True)
        
        # Compute similarity/covariance matrices (num_classes x num_classes)
        C_s = z_s.T @ z_s / z_s.size(0)
        C_t = z_t.T @ z_t / z_t.size(0)
        
        # MSE between matrices
        return F.mse_loss(C_s, C_t)

    def forward(self, student_logits, labels, teacher_logits=None):
        """
        Compute the combined distillation loss.
        
        Args:
            student_logits: Logits from student model (batch_size, num_classes)
            labels: Ground truth labels (batch_size, num_classes)
            teacher_logits: Logits from teacher model (batch_size, num_classes), optional
                            OR list/tuple/tensor of logits for ensemble distillation
        
        Returns:
            Combined loss value
        """
        # Hard label loss (BCE or focal)
        if self.use_focal:
            hard_loss = self.focal_loss(student_logits, labels)
        else:
            hard_loss = self.ce_loss(student_logits, labels)

        # If no teacher logits provided, or both alpha and beta are 0, return only hard loss
        if teacher_logits is None or (self.alpha == 0.0 and self.beta == 0.0):
            return hard_loss
        
        # Check if ensemble (list/tuple or 3D tensor)
        is_ensemble = isinstance(teacher_logits, (list, tuple)) or (torch.is_tensor(teacher_logits) and teacher_logits.ndim == 3)
        
        soft_loss = 0.0
        corr_loss = 0.0

        if is_ensemble:
            # Handle ensemble: averaging soft probabilities
            if isinstance(teacher_logits, (list, tuple)):
                teacher_stack = torch.stack(teacher_logits)
            else:
                teacher_stack = teacher_logits

            with torch.no_grad():
                # soft probabilities are calculated first
                teacher_probs = torch.sigmoid(teacher_stack / self.temperature)
                # then averaged
                teacher_representation = torch.mean(teacher_probs, dim=0)

            # calculate soft loss
            if self.alpha > 0:
                soft_loss = F.binary_cross_entropy_with_logits(
                    student_logits / self.temperature,
                    teacher_representation,
                    reduction="mean",
                ) * (self.temperature ** 2)
            
            # calculate correlation
            if self.beta > 0:
                # Use averaged probabilities for correlation loss
                student_probs = torch.sigmoid(student_logits / self.temperature)
                corr_loss = self.correlation_loss(student_probs, teacher_representation)

        else:
            # Standard single-teacher case
            with torch.no_grad():
                # Always compute teacher probs if alpha > 0 or beta > 0
                if self.alpha > 0 or self.beta > 0:
                    teacher_probs = torch.sigmoid(teacher_logits / self.temperature)

            if self.alpha > 0:
                soft_loss = F.binary_cross_entropy_with_logits(
                    student_logits / self.temperature,
                    teacher_probs,
                    reduction="mean",
                ) * (self.temperature ** 2)

            if self.beta > 0:
                student_probs = torch.sigmoid(student_logits / self.temperature)
                corr_loss = self.correlation_loss(student_probs, teacher_probs)

        return ((1.0 - self.alpha - self.beta) * hard_loss + self.alpha * soft_loss + self.beta * corr_loss), hard_loss, soft_loss
