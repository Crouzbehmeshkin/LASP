import torch
import torch.nn as nn


# SupConLoss computed for each feature separately
class SupConImportance:
    def __init__(self, temperature=0.07, base_temperature=0.07, reduction='mean', importance_smoothing=1):
        self.target_labels = None
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.reduction = reduction
        self.importance_smoothing = importance_smoothing

    def __call__(self, features, labels):
        target_labels = self.target_labels
        assert target_labels is not None and len(
            target_labels) > 0, "Target labels should be given as a list of integer"

        device = features.device

        org_shape = features.shape
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, num_features],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        batch_size = features.shape[0]
        num_features = features.shape[2]
        # mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        # exploding features into a new axis
        contrast_feature = torch.stack(torch.unbind(contrast_feature, dim=1)).unsqueeze(2)
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        mask = mask.unsqueeze(0).repeat(num_features, 1, 1)
        logits_mask = logits_mask.unsqueeze(0).repeat(num_features, 1, 1)

        # compute logits for each feature
        # [num_features, bs x n_views, bs x n_views]
        anchor_dot_contrast = torch.div(
            torch.bmm(anchor_feature, torch.transpose(contrast_feature, 1, 2)),
            self.temperature)

        # best clustered version of anchor dot contrast. will be subtracted from importances as the
        # minimum importance
        bc_adc = anchor_dot_contrast.detach().clone()
        bc_adc[torch.where(mask == 1)] = 1

        # for numerical stability [not gonna do when computing saliency]
        # logits_max, _ = torch.max(anchor_dot_contrast, dim=2, keepdim=True)
        # logits = anchor_dot_contrast - logits_max.detach()
        # bc_logits = bc_adc - logits_max.detach()
        logits = anchor_dot_contrast
        bc_logits = bc_adc

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        exp_bc_logits = torch.exp(bc_logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(2, keepdim=True))
        bc_log_prob = logits - torch.log(exp_bc_logits.sum(2, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(2) / mask.sum(2)
        bc_mean_log_prob_pos = (mask * bc_log_prob).sum(2) / mask.sum(2)

        # loss
        importance = - (self.temperature / self.base_temperature) * (mean_log_prob_pos + bc_mean_log_prob_pos.detach())
        # importance = - (self.temperature / self.base_temperature) * (mean_log_prob_pos)
        importance = 1 / (1 + importance/self.importance_smoothing)
        # divide by batch_size*num_views since the importances will be accumulated for all images when
        # graident of weights is being computed: will be done later in the saliency function
        # importance = importance/importance.shape[1]

        importance = torch.permute(torch.stack(importance.split(batch_size, dim=1), dim=1), [2, 1, 0])
        importance = importance.view(org_shape)

        return importance


class GaussianImportance:
    """
    Computes importance for each feature separately based on the Gaussian function:
    sim(i, j) = exp( - (i-j)**2  / temperature)
    importance = mean(sim of positives) - mean(sim of negatives)
    importance is clamped to zero.
    """
    def __init__(self, temperature=0.1):
        self.temperature = temperature

    def __call__(self, features, labels):
        device = features.device

        org_shape = features.shape
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, num_features],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        batch_size = features.shape[0]
        num_features = features.shape[2]
        # mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        # exploding features into a new axis
        contrast_feature = torch.stack(torch.unbind(contrast_feature, dim=1)).unsqueeze(2)
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        positive_mask = mask * logits_mask
        negative_mask = (1-mask) * logits_mask
        positive_mask = positive_mask.unsqueeze(0).repeat(num_features, 1, 1)
        negative_mask = negative_mask.unsqueeze(0).repeat(num_features, 1, 1)

        # compute logits for each feature
        # [num_features, bs x n_views, bs x n_views]
        anchor_contrast_diff = (anchor_feature - torch.transpose(contrast_feature, 1, 2)).square()
        anchor_contrast_diff = torch.exp(-anchor_contrast_diff / self.temperature)

        importance = (anchor_contrast_diff*positive_mask).sum(dim=2) / positive_mask.sum(dim=2)\
                     - (anchor_contrast_diff*negative_mask).sum(dim=2) / negative_mask.sum(dim=2)

        # Multiplying by each feature absolute value. If a feature value is close to zero, it's not considered
        # importance
        # importance = contrast_feature[:, :, 0].abs() * importance

        importance = torch.clamp(importance, min=0)
        importance = torch.permute(torch.stack(importance.split(batch_size, dim=1), dim=1), [2, 1, 0])
        importance = importance.view(org_shape)

        # mask = importance <= 0.99
        # importance[~mask] = 1
        # importance[mask] = 0

        return importance


class PeakyImportance:
    def __init__(self, optimization_steps = 100, num_clones = 10, ld1 = 1):
        # class_means: (representation_size, num_classes)
        # new class data: list of (data_len, representation_size)
        self.class_means = None
        self.optimization_steps = optimization_steps
        self.ld1 = ld1
        self.num_clones = num_clones

    def __call__(self, features, labels):
        # features: (data_len, representation_size)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        feature_len, num_classes = features.shape[1], labels.max()+1
        self.class_means = torch.zeros((feature_len, num_classes), device=device)

        with torch.no_grad():
            uq_labels = torch.unique(labels)
            for label in uq_labels:
                cl_reps = features[labels == label]
                proto = cl_reps.mean(dim=0)
                proto = torch.nn.functional.normalize(proto, 2, dim=0)
                self.class_means[:, label] = proto

        # print(uq_labels)
        salience = torch.randn(self.num_clones, features.shape[1], requires_grad=True, device=device)
        optimizer = torch.optim.SGD([salience], lr=100)
        # criterion = torch.nn.CrossEntropyLoss()
        accs = [0 for s in salience]
        for step in range(self.optimization_steps):
            optimizer.zero_grad()
            select = torch.sigmoid(salience)
            loss = self.ld1 * torch.norm(select, 1)
            for i, s in enumerate(select):
                accs[i] = 0
                reps = features * s.unsqueeze(0)
                reps = torch.nn.functional.normalize(reps, 2, 1)
                protos = self.class_means * s.unsqueeze(0).T
                protos = torch.nn.functional.normalize(protos, 2, 0)
                sim = torch.mm(reps, protos)
                pred = torch.argmax(sim, dim=1)
                for label in uq_labels:
                    mask = labels == label
                    pred_l = pred[mask]
                    labels_l = labels[mask]
                    accs[i] += (pred_l == labels_l).sum() / pred_l.shape[0]
                accs[i] = accs[i] / len(uq_labels)
                # loss += criterion(sim, labels)
                loss -= sim[:, labels].mean()

            features.requires_grad_(False)
            self.class_means.requires_grad_(False)
            loss.backward()
            optimizer.step()
        select = torch.sigmoid(salience)
        select[select <= 0.5] = 0
        ret = select[0] * accs[0]
        best_acc = accs[0]
        for i, s in enumerate(select[1:]):
            if best_acc < accs[i]:
                ret = s * accs[i]
                best_acc = accs[i]
        print(f'selected representation accuracy: {best_acc}, no select {(ret>0).sum()}')
        return ret.detach().unsqueeze(0).repeat(features.shape[0], 1)

