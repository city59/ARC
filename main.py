"""Train ARC: personalized relation information-bottleneck recommendation."""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from modules.arc import (ContextEncoder, NeighborSampler,
                              ARC, parameter_penalty)
from utils.ib_data import load_dataset


ROOT = Path(__file__).resolve().parent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['movie', 'music', 'book'], default='music')
    parser.add_argument('--data-root', '--data_path', type=Path, default=ROOT / 'data')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'runs')
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda, cuda:0')
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--validation-ratio', type=float, default=0.1)
    parser.add_argument('--no-inverse', action='store_true')
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--context-layers', type=int, default=2)
    parser.add_argument('--codes', type=int, default=4)
    parser.add_argument('--blocks', type=int, default=4)
    parser.add_argument('--keep-blocks', type=int, default=2)
    parser.add_argument('--fanout', type=int, default=8)
    parser.add_argument('--batch-size', '--batch_size', type=int, default=128)
    parser.add_argument('--context-epochs', type=int, default=5)
    parser.add_argument('--epochs', '--epoch', type=int, default=100)
    parser.add_argument('--steps-per-epoch', type=int, default=0,
                        help='0: ceil(number of train positives / batch size)')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--beta-context', type=float, default=1e-3)
    parser.add_argument('--beta-relation', type=float, default=1e-2)
    parser.add_argument('--beta-message', type=float, default=1e-3)
    parser.add_argument('--l2', type=float, default=1e-6)
    parser.add_argument('--grad-clip', type=float, default=5.0)
    parser.add_argument('--mask-temperature', type=float, default=1.0)
    parser.add_argument('--code-temperature', type=float, default=1.0)
    parser.add_argument('--gumbel-temperature', type=float, default=1.0)
    parser.add_argument('--context-samples', type=int, default=2)
    parser.add_argument('--eval-samples', type=int, default=2)
    parser.add_argument('--eval-every', type=int, default=5)
    parser.add_argument('--eval-users', type=int, default=0,
                        help='0: every eligible user; nonzero is a diagnostic subset')
    parser.add_argument('--ks', type=int, nargs='+', default=[5, 10, 20])
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--log-every', type=int, default=50)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--smoke-test', action='store_true',
                        help='Real data, small architecture, three updates per stage')
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args(argv)
    args.model_name = 'ARC'
    if args.smoke_test:
        args.dim, args.batch_size, args.fanout = 16, 32, 4
        args.blocks, args.keep_blocks = 4, 2
        args.context_epochs = args.epochs = args.eval_every = 1
        args.steps_per_epoch, args.eval_users = 3, 2
        args.context_samples, args.eval_samples = 1, 2
        args.log_every = 1
    positive = ['dim', 'layers', 'context_layers', 'codes', 'blocks', 'fanout',
                'batch_size', 'context_epochs', 'epochs', 'context_samples',
                'eval_samples', 'eval_every', 'patience', 'log_every', 'threads']
    for name in positive:
        if getattr(args, name) <= 0:
            parser.error('--' + name.replace('_', '-') + ' must be positive')
    if args.dim % args.blocks or not 0 < args.keep_blocks < args.blocks:
        parser.error('dim must be divisible by blocks; 0 < keep-blocks < blocks')
    if args.codes < 2 or args.steps_per_epoch < 0 or args.eval_users < 0:
        parser.error('codes >= 2; steps-per-epoch and eval-users >= 0')
    if not 0 < args.validation_ratio < 1 or any(k <= 0 for k in args.ks):
        parser.error('0 < validation-ratio < 1; all ks must be positive')
    if args.lr <= 0 or args.grad_clip <= 0:
        parser.error('lr and grad-clip must be positive')
    for name in ['mask_temperature', 'code_temperature', 'gumbel_temperature']:
        if getattr(args, name) <= 0:
            parser.error(name + ' must be positive')
    for name in ['beta_context', 'beta_relation', 'beta_message', 'l2']:
        if getattr(args, name) < 0:
            parser.error(name + ' must be nonnegative')
    if args.eval_only and args.checkpoint is None:
        parser.error('--eval-only requires --checkpoint')
    args.ks = sorted(set(args.ks))
    return args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_json(path, value):
    path.write_text(json.dumps(jsonable(value), indent=2, ensure_ascii=False),
                    encoding='utf-8')


def emit(record, log_path=None):
    record = {'model_name': 'ARC', **record}
    line = json.dumps(jsonable(record), ensure_ascii=False)
    print(line, flush=True)
    if log_path is not None:
        with log_path.open('a', encoding='utf-8') as stream:
            stream.write(line + '\n')


@torch.no_grad()
def evaluate(model, data, sampler, split='valid', ks=(5, 10, 20),
             max_users=0, samples=2, seed=2024):
    """Full-catalog ranking, positive-only labels, no test-driven selection."""
    if split not in ('valid', 'test'):
        raise ValueError('split must be valid or test')
    targets = data.valid_user_items if split == 'valid' else data.test_user_items
    users = np.array(sorted(u for u, items in targets.items() if items), dtype=np.int64)
    if max_users and max_users < len(users):
        users = np.sort(np.random.default_rng(seed).choice(users, max_users, replace=False))
    if len(users) == 0:
        raise ValueError('No eligible users in the ' + split + ' split')
    model.eval()
    metrics = {name + '@' + str(k): 0.0
               for k in ks for name in ('recall', 'ndcg', 'precision', 'hit')}
    groups = {name: dict(users=0, metrics={key: 0.0 for key in metrics})
              for name in ('warm', 'cold')}
    device = model.e_node.weight.device
    devices = [device.index or 0] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        for ordinal, user_value in enumerate(users):
            user = int(user_value)
            torch.manual_seed(seed + user)
            if device.type == 'cuda':
                torch.cuda.manual_seed(seed + user)
            scores = model.score_all_items(user, sampler, samples=samples).cpu().numpy()
            if not np.isfinite(scores).all():
                raise FloatingPointError('Non-finite evaluation scores')
            seen = set(data.train_user_items.get(user, set()))
            if split == 'test':
                seen.update(data.valid_user_items.get(user, set()))
            relevant = set(targets[user]) - seen
            if not relevant:
                raise ValueError('Evaluation positives overlap all excluded items')
            candidates = np.setdiff1d(np.arange(data.n_items), list(seen))
            ranked = candidates[np.lexsort((candidates, -scores[candidates]))][:max(ks)]
            hits = np.array([item in relevant for item in ranked], dtype=np.float64)
            group = groups['warm' if data.train_user_items.get(user) else 'cold']
            group['users'] += 1
            for k in ks:
                count = hits[:k].sum()
                dcg = (hits[:k] / np.log2(np.arange(len(hits[:k])) + 2)).sum()
                ideal = (1 / np.log2(np.arange(min(k, len(relevant))) + 2)).sum()
                for name, value in [('recall', count / len(relevant)),
                                    ('ndcg', dcg / ideal), ('precision', count / k),
                                    ('hit', count > 0)]:
                    key = name + '@' + str(k)
                    metrics[key] += float(value)
                    group['metrics'][key] += float(value)
            if (ordinal + 1) % 100 == 0:
                emit({'stage': 'evaluation', 'split': split,
                      'users_done': ordinal + 1, 'users_total': len(users)})
    result = {key: value / len(users) for key, value in metrics.items()}
    for group in groups.values():
        group['metrics'] = ({key: value / group['users'] for key, value in group['metrics'].items()}
                            if group['users'] else None)
    result.update(split=split, users=len(users), eligible_users=len(targets),
                  full_catalog=True, user_subset=bool(max_users and max_users < len(targets)),
                  mc_samples=samples, user_groups=groups)
    return result


def train_epoch(model, data, sampler, optimizer, args, rng, stage, log_path):
    model.train()
    steps = args.steps_per_epoch or math.ceil(len(data.train_pairs) / args.batch_size)
    total = np.zeros(4, dtype=np.float64)
    start = time.time()
    for step in range(steps):
        users, positives, negatives = data.sample_triples(args.batch_size, rng)
        if stage == 'context':
            pos, neg, message_rate = model.pair_scores(
                users, positives, negatives, data.n_users, sampler)
            relation_rate = message_rate * 0.0
            beta = args.beta_context
        else:
            pos, neg, message_rate, relation_rate = model.pair_scores(
                users, positives, negatives, sampler)
            beta = args.beta_message
        ranking = F.softplus(neg - pos).mean()
        loss = (ranking + beta * message_rate + args.beta_relation * relation_rate
                + args.l2 * parameter_penalty(model))
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite training loss in ' + stage)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if not torch.isfinite(norm):
            raise FloatingPointError('Non-finite gradients in ' + stage)
        optimizer.step()
        total += [loss.item(), ranking.item(), message_rate.item(), relation_rate.item()]
        if (step + 1) % args.log_every == 0 or step + 1 == steps:
            emit({'stage': stage, 'step': step + 1, 'steps': steps,
                  'loss': total[0] / (step + 1), 'seconds': time.time() - start}, log_path)
    return dict(zip(('loss', 'bpr', 'message_rate', 'relation_rate'), total / steps))


def model_config(args):
    return dict(dim=args.dim, layers=args.layers, n_codes=args.codes,
                n_blocks=args.blocks, keep_blocks=args.keep_blocks,
                mask_temperature=args.mask_temperature, code_temperature=args.code_temperature,
                gumbel_temperature=args.gumbel_temperature)


def make_model(data, context, config, device):
    return ARC(data.n_users, data.n_items, data.n_entities,
                                     data.n_relations, data.rho, context,
                                     **config).to(device)


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu')
                          if args.device == 'auto' else args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use --device cpu')
    checkpoint = None
    if args.eval_only:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        saved = checkpoint['args']
        args.dataset = saved['dataset']
        args.seed = saved['seed']
        args.validation_ratio = saved['validation_ratio']
        args.no_inverse = saved['no_inverse']
        args.fanout = saved['fanout']
        seed_everything(args.seed)
    data = load_dataset(args.data_root, args.dataset, seed=args.seed,
                        val_ratio=args.validation_ratio, inverse=not args.no_inverse)
    output = args.output_dir / args.dataset / ('smoke' if args.smoke_test else 'train')
    if args.eval_only:
        output = args.checkpoint.resolve().parent
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / ('evaluation.jsonl' if args.eval_only else 'training.jsonl')
    emit({'stage': 'data', 'dataset': args.dataset, 'device': str(device),
          'torch': torch.__version__, 'stats': {key: value for key, value in data.stats.items()
                                               if key not in ('rho', 'relation_counts')}}, log_path)
    eval_sampler = NeighborSampler(data.joint_graph, args.fanout, args.seed + 70000)
    if args.eval_only:
        if checkpoint['data_stats'] != jsonable(data.stats):
            raise ValueError('Checkpoint data/split fingerprint does not match loaded dataset')
        model = make_model(data, checkpoint['model']['e_context'],
                           checkpoint['model_config'], device)
        model.load_state_dict(checkpoint['model'])
        result = evaluate(model, data, eval_sampler, split='test', ks=args.ks,
                          max_users=args.eval_users, samples=args.eval_samples, seed=args.seed)
        result = dict(model_name='ARC', **result)
        emit(result, log_path)
        write_json(output / 'evaluation.json', result)
        return result
    if (output / 'best.pt').exists():
        raise FileExistsError('Run already exists: ' + str(output)
                              + '. Select a new --output-dir.')
    write_json(output / 'config.json', vars(args))
    write_json(output / 'data_stats.json', data.stats)
    rng = np.random.default_rng(args.seed)
    context_model = ContextEncoder(data.n_users + data.n_items, args.dim, args.context_layers).to(device)
    optimizer = torch.optim.Adam(context_model.parameters(), lr=args.lr)
    for epoch in range(args.context_epochs):
        sampler = NeighborSampler(data.ui_graph, args.fanout, args.seed + epoch)
        metrics = train_epoch(context_model, data, sampler, optimizer, args, rng, 'context', log_path)
        emit(dict(stage='context_epoch', epoch=epoch + 1, **metrics), log_path)
    emit({'stage': 'cache_context', 'users': data.n_users}, log_path)
    context_sampler = NeighborSampler(data.ui_graph, args.fanout, args.seed + 60000)
    context = context_model.cache_context(data.n_users, context_sampler,
                                          samples=args.context_samples)
    torch.save({'model_name': 'ARC', 'stage': 'context',
                'model': context_model.state_dict(), 'context': context.cpu(),
                'args': jsonable(vars(args)), 'data_stats': jsonable(data.stats)}, output / 'context.pt')
    del optimizer, context_model, sampler, context_sampler
    model = make_model(data, context, model_config(args), device)
    del context
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_score, bad_evaluations = -math.inf, 0
    selection_key = 'recall@' + str(20 if 20 in args.ks else max(args.ks))
    for epoch in range(args.epochs):
        sampler = NeighborSampler(data.joint_graph, args.fanout, args.seed + 10000 + epoch)
        metrics = train_epoch(model, data, sampler, optimizer, args, rng, 'recommendation', log_path)
        emit(dict(stage='recommendation_epoch', epoch=epoch + 1, **metrics), log_path)
        if (epoch + 1) % args.eval_every != 0 and epoch + 1 != args.epochs:
            continue
        validation = evaluate(model, data, eval_sampler, split='valid', ks=args.ks,
                              max_users=args.eval_users, samples=args.eval_samples, seed=args.seed)
        emit(dict(stage='validation', epoch=epoch + 1, **validation), log_path)
        improved = validation[selection_key] > best_score
        if improved:
            best_score, bad_evaluations = validation[selection_key], 0
        else:
            bad_evaluations += 1
        checkpoint = dict(model_name='ARC', model=model.state_dict(), model_config=model_config(args),
                          optimizer=optimizer.state_dict(), args=jsonable(vars(args)),
                          data_stats=jsonable(data.stats), epoch=epoch + 1,
                          validation=validation, best_score=best_score)
        torch.save(checkpoint, output / 'last.pt')
        if improved:
            torch.save(checkpoint, output / 'best.pt')
        if bad_evaluations >= args.patience:
            break
    checkpoint = torch.load(output / 'best.pt', map_location=device)
    model.load_state_dict(checkpoint['model'])
    test = evaluate(model, data, eval_sampler, split='test', ks=args.ks,
                    max_users=args.eval_users, samples=args.eval_samples, seed=args.seed)
    result = dict(model_name='ARC', dataset=args.dataset, smoke_test=args.smoke_test,
                  best_epoch=checkpoint['epoch'], validation=checkpoint['validation'], test=test)
    write_json(output / 'results.json', result)
    emit(dict(stage='finished', **result), log_path)
    return result


if __name__ == '__main__':
    main()
