"""Clean V4 Muon (DSv4 Algorithm 1) + min_sv sentinel."""
import torch
PHASE_A = (3.4445, -4.7750, 2.0315)
PHASE_B = (2.0, -1.5, 0.5)

@torch.no_grad()
def newton_schulz(X, n_a=8, n_b=2):
    orig = X.shape
    if X.dim() != 2: X = X.reshape(X.shape[0], -1)
    transposed = X.shape[0] > X.shape[1]
    if transposed: X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(n_a):
        XXt = X @ X.T
        a, b, c = PHASE_A
        X = a * X + b * (XXt @ X) + c * (XXt @ XXt @ X)
    for _ in range(n_b):
        XXt = X @ X.T
        a, b, c = PHASE_B
        X = a * X + b * (XXt @ X) + c * (XXt @ XXt @ X)
    if transposed: X = X.T
    return X.reshape(orig)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=2e-4, momentum=0.95, weight_decay=0.1,
                 gamma=0.18, ns_a=8, ns_b=2):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        gamma=gamma, ns_a=ns_a, ns_b=ns_b)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()
        for group in self.param_groups:
            lr, mu, wd = group['lr'], group['momentum'], group['weight_decay']
            gamma, ns_a, ns_b = group['gamma'], group['ns_a'], group['ns_b']
            for p in group['params']:
                if p.grad is None or p.grad.dim() < 2: continue
                g = p.grad
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p, dtype=torch.float32)
                buf = state['momentum_buffer']
                buf.mul_(mu).add_(g.float())
                ns_input = mu * buf + g.float()
                update = newton_schulz(ns_input, n_a=ns_a, n_b=ns_b)
                update = update * (max(p.shape) ** 0.5) * gamma
                if wd > 0: p.mul_(1.0 - lr * wd)
                p.add_(update.to(p.dtype), alpha=-lr)
                state['check_sv_step'] = state.get('check_sv_step', 0) + 1
                # Check spectrum every 1000 steps (skip step 0: random init has min/max ~ 1/sqrt(N)
                # for square N x N matrices, which would trigger false alarms on every restart).
                # Use a relative threshold so the warning fires only on genuine rank degeneration.
                if state['check_sv_step'] % 1000 == 0:
                    try:
                        sv = torch.linalg.svdvals(p.float())
                        sv_max, sv_min = sv[0].item(), sv[-1].item()
                        if sv_max > 0 and sv_min / sv_max < 1e-4:
                            print(f'  [Muon WARN] cond_recip={sv_min/sv_max:.2e} '
                                  f'(min_sv={sv_min:.5f}, max_sv={sv_max:.3f}) '
                                  f'on {tuple(p.shape)} -- rank loss')
                    except Exception: pass
        return loss


def split_params_for_muon(model):
    muon_p, adamw_p = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        is_router = '.mlp.gate' in name and 'gate_proj' not in name
        is_2d_hidden = (p.ndim >= 2
            and 'embed' not in name.lower()
            and 'norm' not in name.lower()
            and 'bias' not in name.lower()
            and not is_router)
        (muon_p if is_2d_hidden else adamw_p).append(p)
    return muon_p, adamw_p


def build_optimizers(model, peak_lr=2e-4, momentum=0.95, beta2=0.95, wd=0.1):
    muon_p, adamw_p = split_params_for_muon(model)
    muon_opt = Muon(muon_p, lr=peak_lr, momentum=momentum, weight_decay=wd) if muon_p else None
    adamw_opt = torch.optim.AdamW(adamw_p, lr=peak_lr, betas=(0.9, beta2),
                                   weight_decay=0.0, eps=1e-20) if adamw_p else None
    print(f'  Muon:  {len(muon_p)} tensors, {sum(p.numel() for p in muon_p)/1e6:.2f}M params')
    print(f'  AdamW: {len(adamw_p)} tensors, {sum(p.numel() for p in adamw_p)/1e6:.2f}M params')
    return muon_opt, adamw_opt
