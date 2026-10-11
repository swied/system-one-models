"""Head-only/full Laya training with validation stopping and fp16 CUDA AMP."""
import argparse
import contextlib
import copy
import json
import random
from pathlib import Path

from system_one_models.finetuning import encode_examples, verify_dataset

MODEL_REVISION = '7b928d828b7b0e022f929d9bd2e44165aa270148'


def inputs(batch, device):
    return {k: batch[k].to(device) for k in
            ('input_ids', 'attention_mask', 'marker_pos', 'marker_mask', 'qtype')}


def save_checkpoint(agent, directory):
    from safetensors.torch import save_file
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in agent.model.state_dict().items()},
              str(directory / 'model.safetensors'))
    agent.model.encoder.config.save_pretrained(directory / 'encoder')
    agent.tok.save_pretrained(directory / 'tokenizer')
    (directory / 'rl_agent_config.json').write_text(json.dumps(agent.cfg, indent=2) + '\n')


def rlcd_loss(logits, target, sigma, group_size):
    """Gaussian-logit policy gradient with leave-one-out proper-score baseline.

    This is an experimental RLCD-style objective, not a reproduction of vendor training.
    CE anchors the noisy policy; no act/escalate supervision is available here.
    """
    import torch
    if sigma <= 0 or group_size < 2:
        raise ValueError('RLCD needs positive noise and at least two samples')
    noisy = logits.detach()[None] + sigma * torch.randn(
        (group_size, *logits.shape), device=logits.device)
    p = noisy.softmax(-1)
    rewards = (target[None] * noisy.log_softmax(-1)).sum(-1)
    rewards += (target[None] * p).sum(-1) / p.square().sum(-1).sqrt().clamp_min(1e-8)
    baseline = (rewards.sum(0, keepdim=True) - rewards) / (group_size - 1)
    advantage = (rewards - baseline).detach()
    log_density = -((noisy - logits[None]) ** 2).sum(-1) / (2 * sigma ** 2)
    ce = -(target * logits.log_softmax(-1)).sum(-1).mean()
    return ce - (advantage * log_density).mean()


def train(args):
    import laya
    import numpy as np
    import torch
    from laya.common import collate_items, ece_score
    from system_one_models.finetuning import CheckpointBackend

    if min(args.epochs, args.batch_size, args.accumulation, args.patience) < 1:
        raise ValueError('Training counts must be positive')
    if args.lr_head <= 0 or args.lr_encoder <= 0:
        raise ValueError('Learning rates must be positive')
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new run directory to preserve existing checkpoints')
    splits = verify_dataset(args.data)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    backend = CheckpointBackend('training', args.model, device=args.device,
                                revision=None if Path(args.model).is_dir() else args.revision)
    agent = backend.agent
    agent.cfg = copy.deepcopy(agent.cfg)
    agent.cfg['max_len'] = min(512, agent.cfg['max_len'])
    agent.cfg['temperature'] = [1.0, 1.0, 1.0]
    for key in ('temperature_by_options', 'lang_temperatures', 'binning_map'):
        agent.cfg.pop(key, None)
    agent.cfg['fine_tuned'] = True
    agent.cfg['model_name'] = f'week03-ticket-{args.regime}'
    training = encode_examples(agent, splits['train'])
    validation = encode_examples(agent, splits['validation'])
    model = agent.model.float()
    for name, p in model.named_parameters():
        p.requires_grad_(not name.startswith('act_head.') and
                         (args.regime == 'full' or not name.startswith('encoder.')))
    model.head_checkpointing = True
    if args.regime == 'full':
        model.encoder.gradient_checkpointing_enable()
    groups = []
    for encoder, lr in ((True, args.lr_encoder), (False, args.lr_head)):
        params = [p for n, p in model.named_parameters()
                  if p.requires_grad and n.startswith('encoder.') == encoder]
        if params:
            groups.append({'params': params, 'lr': lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01, foreach=False)
    cuda = agent.device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=cuda)
    def amp():
        return torch.autocast('cuda', dtype=torch.float16) if cuda else contextlib.nullcontext()
    def batch_of(items):
        return collate_items([items], agent.tok.pad_token_id)
    def validate():
        model.eval()
        correct, confidence, loss = [], [], []
        with torch.inference_mode():
            for start in range(0, len(validation), args.batch_size):
                batch = batch_of(validation[start:start + args.batch_size])
                with amp():
                    logits, _ = model(**inputs(batch, agent.device))
                p = logits.softmax(-1)
                labels = batch['label'].to(agent.device)
                correct.extend((p.argmax(-1) == labels).cpu().tolist())
                confidence.extend(p.max(-1).values.cpu().tolist())
                loss.extend((-logits.log_softmax(-1).gather(1, labels[:, None])[:, 0]).cpu().tolist())
        return {'accuracy': float(np.mean(correct)), 'ece': float(ece_score(
            np.asarray(confidence), np.asarray(correct))), 'nll': float(np.mean(loss))}
    history, best, stale = [], -1.0, 0
    output.mkdir(parents=True, exist_ok=True)
    metadata = {'arguments': vars(args), 'dataset_manifest': json.loads(
        (Path(args.data) / 'manifest.json').read_text()), 'torch': torch.__version__,
        'laya': getattr(laya, '__version__', 'unknown'), 'device': str(agent.device),
        'hardware': torch.cuda.get_device_name(agent.device) if cuda else 'CPU',
        'precision': 'fp16 autocast, fp32 parameters/optimizer' if cuda else 'fp32',
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'initial_validation': validate()}
    agent.cfg['week03_training'] = metadata
    for epoch in range(args.epochs):
        model.train()
        if args.regime == 'head':
            model.encoder.eval()
        model.act_head.eval()
        random.shuffle(training)
        total_loss = 0.0
        window = args.batch_size * args.accumulation
        for start in range(0, len(training), window):
            examples = training[start:start + window]
            optimizer.zero_grad(set_to_none=True)
            for offset in range(0, len(examples), args.batch_size):
                items = examples[offset:offset + args.batch_size]
                batch = batch_of(items)
                with amp():
                    logits, _ = model(**inputs(batch, agent.device))
                    target = batch['target'].to(agent.device)
                    if args.objective == 'rlcd':
                        sigma = 0.4 + (0.1 - 0.4) * epoch / max(1, args.epochs - 1)
                        loss = rlcd_loss(logits, target, sigma, 4)
                    else:
                        loss = -(target * logits.log_softmax(-1)).sum(-1).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite loss; reduce learning rate or batch size')
                scaler.scale(loss * len(items) / len(examples)).backward()
                total_loss += float(loss.detach()) * len(items)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(optimizer)
            scaler.update()
        metrics = validate()
        history.append({'epoch': epoch + 1, 'training_loss': total_loss / len(training), **metrics})
        print(json.dumps(history[-1]))
        if metrics['accuracy'] > best:
            best, stale = metrics['accuracy'], 0
            metadata['best_epoch'] = epoch + 1
            save_checkpoint(agent, output / 'best')
        else:
            stale += 1
        # Each epoch is recoverable; best is used for calibration/evaluation.
        save_checkpoint(agent, output / 'last')
        torch.save({'epoch': epoch + 1, 'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(), 'random_state': random.getstate(),
                    'torch_rng_state': torch.get_rng_state(),
                    'cuda_rng_state': torch.cuda.get_rng_state_all() if cuda else None},
                   output / 'training-state.pt')
        (output / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
        (output / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
        if stale >= args.patience:
            break
    if cuda:
        print('Peak allocated GiB:', torch.cuda.max_memory_allocated() / 2**30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', default='convaiinnovations/laya')
    parser.add_argument('--revision', default=MODEL_REVISION)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--regime', choices=['head', 'full'], default='head')
    parser.add_argument('--objective', choices=['ce', 'rlcd'], default='ce')
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--accumulation', type=int, default=8)
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--lr-head', type=float, default=1e-4)
    parser.add_argument('--lr-encoder', type=float, default=2.5e-5)
    parser.add_argument('--seed', type=int, default=42)
    train(parser.parse_args())


if __name__ == '__main__':
    main()
