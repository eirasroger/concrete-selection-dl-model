import torch

def listmle_loss(scores, ranks, mask):
    loss = 0.0
    for s, r, m in zip(scores, ranks, mask):
        valid_idx = m.nonzero(as_tuple=False).squeeze(1)
        s_i = s[valid_idx]
        rank_i = r[valid_idx]
        sorted_idx = rank_i.argsort()
        s_perm = s_i[sorted_idx]
        for t in range(len(s_perm)):
            rest = s_perm[t:]
            loss += -(s_perm[t] - torch.logsumexp(rest, dim=0))
    return loss / mask.sum().float()