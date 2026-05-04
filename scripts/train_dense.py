"""Phase 1/2/3 trainer for dense Qwen3 50M with FIM data augmentation
and Muon V4 + AdamW hybrid optimizer.

FIM (Bavarian et al. 2022) is enabled via --fim-rate. With probability p,
each document is permuted into PSM order:
    [<|fim_prefix|>] prefix [<|fim_suffix|>] suffix [<|fim_middle|>] middle
The model then learns infilling for free without sacrificing left-to-right.

Phase 1 (pretrain) uses fim_rate=0.5; phases 2 (CPT) and 3 (SFT) use 0.0.
"""
import argparse, json, math, os, random, re, sys, time
import torch


_DR = re.compile(r'\d{2,}')
def per_digit_wrap(t):
    return _DR.sub(lambda m: ' '.join(m.group()), t)


def fim_permute(ids, fim_pre, fim_suf, fim_mid, prob, rng):
    """PSM (prefix-suffix-middle) reorder.
    With prob p, transform ids = [A B C] -> [<PRE> A <SUF> C <MID> B]
    where the splits are at uniformly random points. Sequences too short
    to give all three regions a meaningful slice are passed through.
    """
    if rng.random() >= prob:
        return ids
    n = len(ids)
    if n < 64:
        return ids
    a = rng.randint(8, n - 16)
    b = rng.randint(a + 4, n - 4)
    return [fim_pre] + ids[:a] + [fim_suf] + ids[b:] + [fim_mid] + ids[a:b]


class JsonlStream:
    def __init__(self, path, apply_pd=True, max_in_mem=1_000_000):
        self.path, self.pd = path, apply_pd
        self.records, self.streaming, self._fh = [], False, None
        sz = os.path.getsize(path)
        if sz < max_in_mem * 200:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        t = json.loads(line).get('text', '')
                        if t: self.records.append(t)
                    except json.JSONDecodeError: continue
            random.shuffle(self.records); self.idx = 0
        else:
            self.streaming = True
    def __iter__(self): return self
    def __next__(self):
        if self.streaming:
            while True:
                if self._fh is None:
                    self._fh = open(self.path, 'r', encoding='utf-8')
                line = self._fh.readline()
                if not line:
                    self._fh.close(); self._fh = None; continue
                try:
                    t = json.loads(line).get('text', '')
                    return per_digit_wrap(t) if self.pd else t
                except json.JSONDecodeError: continue
        if not self.records: raise StopIteration
        r = self.records[self.idx]
        self.idx = (self.idx + 1) % len(self.records)
        return per_digit_wrap(r) if self.pd else r


def make_mixed(cache_dir, weights, tok, seq_len, fim_rate=0.0, apply_pd=True, rng=None):
    if rng is None: rng = random
    streams, aw = {}, {}
    for n, w in weights.items():
        p = os.path.join(cache_dir, f'{n}.jsonl')
        if os.path.exists(p) and os.path.getsize(p) > 1000:
            streams[n] = iter(JsonlStream(p, apply_pd))
            aw[n] = w
        else:
            print(f'  WARN: {p} missing -- dropping {n}')
    if not aw: raise RuntimeError('no data')
    total = sum(aw.values()); aw = {k: v/total for k, v in aw.items()}
    names, probs = list(aw.keys()), [aw[n] for n in aw]

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    eos_id = tok.eos_token_id if tok.eos_token_id is not None else pad_id
    fim_pre = tok.convert_tokens_to_ids('<|fim_prefix|>')
    fim_suf = tok.convert_tokens_to_ids('<|fim_suffix|>')
    fim_mid = tok.convert_tokens_to_ids('<|fim_middle|>')
    if fim_rate > 0:
        if any(x is None or x == tok.unk_token_id for x in (fim_pre, fim_suf, fim_mid)):
            print('  WARN: FIM tokens missing in vocab -- disabling FIM')
            fim_rate = 0.0
    print(f'  active mix: {aw}  fim_rate={fim_rate}  pad_id={pad_id} eos_id={eos_id}')

    buf = []
    while True:
        n = rng.choices(names, weights=probs, k=1)[0]
        try: text = next(streams[n])
        except StopIteration:
            streams[n] = iter(JsonlStream(os.path.join(cache_dir, f'{n}.jsonl'), apply_pd))
            text = next(streams[n])
        ids = tok.encode(text, add_special_tokens=False)
        if fim_rate > 0:
            ids = fim_permute(ids, fim_pre, fim_suf, fim_mid, fim_rate, rng)
        buf.extend(ids); buf.append(eos_id)
        while len(buf) >= seq_len + 1:
            chunk = buf[:seq_len + 1]; buf = buf[seq_len + 1:]
            inp = torch.tensor([chunk[:-1]], dtype=torch.long)
            tgt = torch.tensor([chunk[1:]], dtype=torch.long)
            tgt[tgt == pad_id] = -100
            yield inp, tgt


def batch_iter(stream, bs):
    while True:
        inps, tgts = [], []
        for _ in range(bs):
            inp, tgt = next(stream); inps.append(inp); tgts.append(tgt)
        yield torch.cat(inps, dim=0), torch.cat(tgts, dim=0)


def wsd_lr(step, total, peak, mn, w=0.015, d=0.20):
    ws = int(total * w); ds = int(total * (1.0 - d))
    if step < ws: return mn + (peak - mn) * (step / max(ws, 1))
    if step < ds: return peak
    p = (step - ds) / max(total - ds, 1)
    return mn + (peak - mn) * (1.0 - math.sqrt(p))


def cooldown_b2(step, total, b2_s=0.95, b2_e=0.97, w=0.015):
    ws = int(total * w)
    if step < ws: return b2_s
    p = (step - ws) / max(total - ws, 1)
    return b2_s + (b2_e - b2_s) * min(max(p, 0.0), 1.0)


def set_lr(opts, lr):
    for o in opts:
        if o is None: continue
        for pg in o.param_groups: pg['lr'] = lr


def set_b2(opts, b2):
    for o in opts:
        if isinstance(o, torch.optim.AdamW):
            for pg in o.param_groups: pg['betas'] = (pg['betas'][0], b2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', required=True, choices=['phase1', 'cpt_16k', 'sft_4k'])
    p.add_argument('--resume', required=True, help='HF dir to load model + tokenizer from')
    p.add_argument('--output', required=True)
    p.add_argument('--cache-dir', required=True)
    p.add_argument('--steps', type=int, required=True)
    p.add_argument('--batch-size', type=int, required=True)
    p.add_argument('--grad-accum', type=int, default=1)
    p.add_argument('--seq-len', type=int, required=True)
    p.add_argument('--peak-lr', type=float, required=True)
    p.add_argument('--min-lr', type=float, required=True)
    p.add_argument('--warmup-frac', type=float, default=0.015)
    p.add_argument('--decay-frac', type=float, default=0.20)
    p.add_argument('--z-loss', type=float, default=1e-4)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--fim-rate', type=float, default=0.0,
                   help='FIM (PSM) permutation probability per document')
    p.add_argument('--ckpt-every', type=int, default=2500)
    p.add_argument('--log-every', type=int, default=20)
    p.add_argument('--no-per-digit', action='store_true')
    p.add_argument('--no-liger', action='store_true')
    p.add_argument('--seed', type=int, default=20260504)
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)
    torch.manual_seed(args.seed); random.seed(args.seed)
    rng = random.Random(args.seed)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from muon import build_optimizers

    from transformers import AutoTokenizer, Qwen3ForCausalLM
    tok = AutoTokenizer.from_pretrained(args.resume, use_fast=True)
    print(f'  Loading from {args.resume}')
    model = Qwen3ForCausalLM.from_pretrained(args.resume, torch_dtype=torch.bfloat16)

    if not args.no_liger:
        try:
            from liger_kernel.transformers import apply_liger_kernel_to_qwen3
            apply_liger_kernel_to_qwen3(model)
            print(f'  Liger Kernel applied (fused CE)')
        except ImportError:
            print(f'  liger-kernel not available; using stock CE')
        except Exception as e:
            print(f'  liger apply failed: {e}; using stock CE')

    model.cuda(); model.train()
    model.gradient_checkpointing_enable()
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Dense Qwen3: {n_params/1e6:.2f}M parameters')

    muon_opt, adamw_opt = build_optimizers(model, peak_lr=args.peak_lr)
    optimizers = [o for o in [muon_opt, adamw_opt] if o is not None]

    # Domain mix: web foundation 40%, then reasoning specialization 60%
    # split across math/lang/code. Trainer auto-drops any domain whose
    # combined .jsonl is missing (with a WARN), so this works even if e.g.
    # web.jsonl wasn't built for a particular run.
    MIX = {'web': 0.40, 'math': 0.25, 'lang': 0.20, 'code': 0.15}
    apply_pd = not args.no_per_digit
    stream = make_mixed(args.cache_dir, MIX, tok, args.seq_len,
                        fim_rate=args.fim_rate, apply_pd=apply_pd, rng=rng)
    bi = batch_iter(stream, args.batch_size)

    use_wandb = bool(os.environ.get('WANDB_API_KEY'))
    if use_wandb:
        try:
            import wandb
            wandb.init(project=os.environ.get('WANDB_PROJECT', 'crowfeather-50m'),
                       name=f'{args.phase}-{int(time.time())}', config=vars(args))
        except Exception as e:
            print(f'  wandb init failed: {e}'); use_wandb = False

    eff = args.batch_size * args.grad_accum
    print(f'\n=== {args.phase} begins ({args.steps} steps, B={args.batch_size} '
          f'accum={args.grad_accum} eff={eff}, T={args.seq_len}) ===')
    log_f = open(os.path.join(args.output, 'train.log'), 'a', buffering=1)
    t_start = time.time()
    losses = []
    for step in range(1, args.steps + 1):
        lr = wsd_lr(step, args.steps, args.peak_lr, args.min_lr,
                    args.warmup_frac, args.decay_frac)
        b2 = cooldown_b2(step, args.steps, 0.95, 0.97, args.warmup_frac)
        set_lr(optimizers, lr); set_b2(optimizers, b2)

        for o in optimizers: o.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for accum in range(args.grad_accum):
            try: inp, tgt = next(bi)
            except StopIteration:
                print(f'  data exhausted'); return
            inp, tgt = inp.cuda(), tgt.cuda()
            out = model(input_ids=inp, labels=tgt)
            loss = out.loss / args.grad_accum
            if args.z_loss > 0:
                lse = torch.logsumexp(out.logits.float(), dim=-1).mean()
                loss = loss + (args.z_loss / args.grad_accum) * lse.pow(2)
            if not torch.isfinite(loss):
                print(f'  step {step}: NaN/Inf -- abort'); return
            loss.backward()
            accum_loss += loss.item() * args.grad_accum
            del out, loss, inp, tgt
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        for o in optimizers: o.step()
        avg_loss = accum_loss / args.grad_accum
        losses.append(avg_loss)
        if len(losses) > 100: losses.pop(0)

        if step % args.log_every == 0:
            avg = sum(losses) / len(losses)
            el = (time.time() - t_start) / 60
            gn_v = gn.item() if hasattr(gn, 'item') else float(gn)
            line = f'  step {step:6d}/{args.steps}  loss={avg_loss:.4f} avg100={avg:.4f} lr={lr:.2e} b2={b2:.4f} gn={gn_v:.3f} elapsed={el:.1f}m'
            print(line); log_f.write(line + chr(10))
            if use_wandb:
                try:
                    import wandb
                    wandb.log({f'{args.phase}/loss': avg_loss, f'{args.phase}/avg100': avg,
                               f'{args.phase}/lr': lr, f'{args.phase}/b2': b2,
                               f'{args.phase}/gn': gn_v}, step=step)
                except Exception: pass

        if step % args.ckpt_every == 0 or step == args.steps:
            ckpt = os.path.join(args.output, f'step_{step:06d}')
            model.save_pretrained(ckpt, safe_serialization=True)
            tok.save_pretrained(ckpt)
            print(f'    HF ckpt: {ckpt}')

    final = os.path.join(args.output, 'final')
    model.save_pretrained(final, safe_serialization=True)
    tok.save_pretrained(final)
    print(f'\n=== {args.phase} final saved: {final} ===')
    log_f.close()
    if use_wandb:
        try:
            import wandb; wandb.finish()
        except Exception: pass


if __name__ == '__main__':
    main()
