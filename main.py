"""Train ARC: personalized relation information-bottleneck recommendation."""
import argparse
import copy
import hashlib
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
ARCHITECTURE_VERSION = 'arc-paper-v2'
VARIANTS = ('arc', 'no-context-ib', 'global', 'no-mask', 'random-mask',
            'no-relation-ib', 'no-message-ib')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['movie', 'music', 'book'], default='music')
    parser.add_argument('--data-root', '--data_path', type=Path, default=ROOT / 'data')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'runs')
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda, cuda:0')
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--runs', type=int, default=10,
                        help='Independent training runs; seeds are seed + run index')
    parser.add_argument('--seeds', type=int, nargs='+',
                        help='Explicit distinct training seeds, overriding --runs')
    parser.add_argument('--split-seed', type=int, default=2024,
                        help='Fixed data split seed shared across training runs/methods')
    parser.add_argument('--split-mode', choices=['random', 'provided'], default='random',
                        help='random: fixed 80/20 positive split; provided: archived split')
    parser.add_argument('--validation-ratio', type=float, default=0.0,
                        help='Optional holdout from training; 0 follows the paper 80/20 protocol')
    inverse = parser.add_mutually_exclusive_group()
    inverse.add_argument('--inverse', action='store_true',
                         help='Non-paper extension: separate inverse relation types')
    inverse.add_argument('--no-inverse', dest='inverse', action='store_false',
                         help=argparse.SUPPRESS)
    parser.set_defaults(inverse=False)
    parser.add_argument('--variant', choices=VARIANTS, default='arc')
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--context-layers', type=int, default=2)
    capacity = parser.add_mutually_exclusive_group()
    capacity.add_argument('--codes', type=int,
                          help='Explicit codebook size B >= 1')
    capacity.add_argument('--code-divisor', type=int, choices=[2, 4, 8, 16],
                          help='ceil(original relation count / divisor); default divisor is 4')
    parser.add_argument('--blocks', type=int, default=4)
    parser.add_argument('--keep-blocks', type=int, default=2)
    parser.add_argument('--fanout', type=int, default=0,
                        help='0: exact full-graph propagation; positive: sampled approximation')
    parser.add_argument('--batch-size', '--batch_size', type=int, default=1024)
    parser.add_argument('--query-microbatch', type=int, default=1,
                        help='Unique target users per recommendation forward/backward pass')
    parser.add_argument('--context-epochs', type=int, default=5)
    parser.add_argument('--epochs', '--epoch', type=int, default=100)
    parser.add_argument('--steps-per-epoch', type=int, default=0,
                        help='0: ceil(number of train positives / batch size)')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--beta-context', type=float, default=1e-3)
    parser.add_argument('--beta-relation', type=float, default=1e-2)
    parser.add_argument('--beta-message', type=float, default=1e-3)
    parser.add_argument('--l2', type=float, default=1e-6)
    parser.add_argument('--grad-clip', type=float, default=0.0,
                        help='0: disabled; positive: optional gradient norm clipping')
    parser.add_argument('--mask-temperature', type=float, default=1.0)
    parser.add_argument('--code-temperature', type=float, default=1.0)
    parser.add_argument('--gumbel-temperature', type=float, default=1.0)
    parser.add_argument('--context-samples', type=int, default=2)
    parser.add_argument('--eval-samples', type=int, default=2)
    parser.add_argument('--eval-every', type=int, default=5)
    parser.add_argument('--eval-users', type=int, default=0,
                        help='0: every eligible user; nonzero is a diagnostic subset')
    parser.add_argument('--ks', type=int, nargs='+', default=[20])
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--log-every', type=int, default=50)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--smoke-test', action='store_true',
                        help='Real data, small architecture, three updates per stage')
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--paired-baseline', type=Path,
                        help='Real matched-seed summary JSON for two-sided paired t-tests')
    args = parser.parse_args(argv)
    args.model_name = 'ARC'
    if args.codes is None and args.code_divisor is None:
        args.code_divisor = 4
    if args.smoke_test:
        args.dim, args.batch_size, args.fanout = 16, 32, 4
        args.blocks, args.keep_blocks = 4, 2
        args.context_epochs = args.epochs = args.eval_every = 1
        args.steps_per_epoch, args.eval_users = 3, 2
        args.context_samples, args.eval_samples = 1, 2
        args.log_every = 1
        args.runs, args.seeds = 1, None
    if args.seeds is not None:
        args.runs = len(args.seeds)
    args.training_seeds = (list(args.seeds) if args.seeds is not None
                           else list(range(args.seed, args.seed + args.runs)))
    args.relation_context = 'global' if args.variant == 'global' else 'personalized'
    args.mask_mode = {'no-mask': 'all', 'random-mask': 'random'}.get(args.variant, 'learned')
    if args.variant == 'no-context-ib':
        args.beta_context = 0.0
    elif args.variant == 'no-relation-ib':
        args.beta_relation = 0.0
    elif args.variant == 'no-message-ib':
        args.beta_message = 0.0
    positive = ['dim', 'layers', 'context_layers', 'blocks', 'runs',
                'batch_size', 'context_epochs', 'epochs', 'context_samples',
                'eval_samples', 'eval_every', 'patience', 'log_every', 'threads',
                'query_microbatch']
    for name in positive:
        if getattr(args, name) <= 0:
            parser.error('--' + name.replace('_', '-') + ' must be positive')
    if args.dim % args.blocks or not 0 < args.keep_blocks <= args.blocks:
        parser.error('dim must be divisible by blocks; 0 < keep-blocks <= blocks')
    if args.mask_mode != 'all' and args.keep_blocks == args.blocks:
        parser.error('learned/random masks require keep-blocks < blocks; use --variant no-mask for all blocks')
    if args.codes is not None and args.codes < 1:
        parser.error('codes must be >= 1')
    if args.steps_per_epoch < 0 or args.eval_users < 0 or args.fanout < 0:
        parser.error('steps-per-epoch, eval-users and fanout must be nonnegative')
    if not 0 <= args.validation_ratio < 1 or any(k <= 0 for k in args.ks):
        parser.error('0 <= validation-ratio < 1; all ks must be positive')
    if args.lr <= 0 or args.grad_clip < 0:
        parser.error('lr must be positive and grad-clip nonnegative')
    if len(set(args.training_seeds)) != len(args.training_seeds):
        parser.error('independent training seeds must be distinct')
    if any(seed < 0 or seed >= 2 ** 32 for seed in args.training_seeds + [args.split_seed]):
        parser.error('training and split seeds must be in [0, 2**32)')
    for name in ['mask_temperature', 'code_temperature', 'gumbel_temperature']:
        if getattr(args, name) <= 0:
            parser.error(name + ' must be positive')
    for name in ['beta_context', 'beta_relation', 'beta_message', 'l2']:
        if getattr(args, name) < 0:
            parser.error(name + ' must be nonnegative')
    if args.eval_only and args.checkpoint is None:
        parser.error('--eval-only requires --checkpoint')
    if args.paired_baseline is not None and (args.smoke_test or args.eval_only):
        parser.error('--paired-baseline requires independent training runs, not smoke/eval-only')
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
                  mc_samples=samples, user_groups=groups,
                  evaluation_users_sha256=hashlib.sha256(users.tobytes()).hexdigest())
    return result


def train_epoch(model, data, sampler, optimizer, args, rng, stage, log_path):
    model.train()
    steps = args.steps_per_epoch or math.ceil(len(data.train_pairs) / args.batch_size)
    total = np.zeros(4, dtype=np.float64)
    start = time.time()
    for step in range(steps):
        users, positives, negatives = data.sample_triples(args.batch_size, rng)
        optimizer.zero_grad(set_to_none=True)
        step_metrics = np.zeros(4, dtype=np.float64)
        if stage == 'context':
            batches = [np.arange(len(users))]
        elif stage == 'recommendation':
            batches = query_microbatches(users, args.query_microbatch)
        else:
            raise ValueError('Unknown training stage: ' + stage)
        for indices in batches:
            weight = len(indices) / len(users)
            if stage == 'context':
                pos, neg, message_rate = model.pair_scores(
                    users[indices], positives[indices], negatives[indices], data.n_users, sampler)
                relation_rate = message_rate * 0.0
                beta = args.beta_context
            else:
                pos, neg, message_rate, _ = model.pair_scores(
                    users[indices], positives[indices], negatives[indices], sampler)
                relation_rate = message_rate * 0.0
                beta = args.beta_message
            ranking = F.softplus(neg - pos).mean()
            objective = ranking + beta * message_rate + args.beta_relation * relation_rate
            if not torch.isfinite(objective):
                raise FloatingPointError('Non-finite training loss in ' + stage)
            # Each sampled target user keeps a single (u,r) draw per pass.
            # Weight by triples, not the number of microbatches or unique users.
            (weight * objective).backward()
            step_metrics += weight * np.array([
                objective.item(), ranking.item(), message_rate.item(), relation_rate.item()])
        if stage == 'recommendation':
            # Eq. (6) samples uniformly from all users, including users with no
            # training positives. BPR's eligible-user population is narrower.
            rate_users = rng.integers(data.n_users, size=args.batch_size)
            relation_state = model.relation(
                model.relation_contexts(rate_users), sample=False,
                user_ids=torch.as_tensor(rate_users, dtype=torch.long,
                                         device=model.e_node.weight.device))
            relation_rate = relation_state['rate']
            if not torch.isfinite(relation_rate):
                raise FloatingPointError('Non-finite relation information cost')
            (args.beta_relation * relation_rate).backward()
            step_metrics[0] += args.beta_relation * relation_rate.item()
            step_metrics[3] = relation_rate.item()
        # Eq. (4) pretrains context using BPR + beta_c * rate only. Eq. (8)
        # regularizes recommendation parameters once per optimizer update.
        if stage == 'recommendation' and args.l2:
            regularizer = args.l2 * parameter_penalty(model)
            if not torch.isfinite(regularizer):
                raise FloatingPointError('Non-finite L2 regularizer')
            regularizer.backward()
            step_metrics[0] += regularizer.item()
        if any(not torch.isfinite(parameter.grad).all()
               for parameter in model.parameters() if parameter.grad is not None):
            raise FloatingPointError('Non-finite gradients in ' + stage)
        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        total += step_metrics
        if (step + 1) % args.log_every == 0 or step + 1 == steps:
            emit({'stage': stage, 'step': step + 1, 'steps': steps,
                  'loss': total[0] / (step + 1), 'seconds': time.time() - start}, log_path)
    return dict(zip(('loss', 'bpr', 'message_rate', 'relation_rate'), total / steps))


def query_microbatches(users, max_users):
    """Keep repeated target users together while limiting personalized graph copies."""
    unique_users = np.unique(users)
    for start in range(0, len(unique_users), max_users):
        yield np.flatnonzero(np.isin(users, unique_users[start:start + max_users]))


def model_config(args):
    return dict(dim=args.dim, layers=args.layers, n_codes=args.codes,
                n_blocks=args.blocks, keep_blocks=args.keep_blocks,
                mask_temperature=args.mask_temperature, code_temperature=args.code_temperature,
                gumbel_temperature=args.gumbel_temperature,
                relation_context=args.relation_context, mask_mode=args.mask_mode,
                mask_seed=args.seed)


def make_model(data, context, config, device):
    return ARC(data.n_users, data.n_items, data.n_entities,
                                     data.n_relations, data.rho, context,
                                     **config).to(device)


def check_checkpoint(checkpoint):
    if checkpoint.get('architecture_version') != ARCHITECTURE_VERSION:
        raise ValueError('Checkpoint architecture is incompatible with ARC paper v2. '
                         'Train a new checkpoint using the current implementation.')
    if checkpoint.get('stage') != 'recommendation':
        raise ValueError('Evaluation requires a recommendation-stage checkpoint')


def run_training(args, data, device, output):
    """One independent run, with a fixed split and no test-driven selection."""
    seed_everything(args.seed)
    if any((output / name).exists() for name in ('config.json', 'context.pt', 'last.pt', 'best.pt')):
        raise FileExistsError('Run already exists: ' + str(output)
                              + '. Select a new --output-dir.')
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / 'training.jsonl'
    emit({'stage': 'data', 'dataset': args.dataset, 'device': str(device),
          'seed': args.seed, 'split_seed': args.split_seed,
          'propagation': 'full_graph' if args.fanout == 0 else 'sampled_approximation',
          'torch': torch.__version__, 'stats': {key: value for key, value in data.stats.items()
                                               if key not in ('rho', 'relation_counts')}}, log_path)
    eval_sampler = NeighborSampler(data.joint_graph, args.fanout, args.seed + 70000)
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
                'architecture_version': ARCHITECTURE_VERSION,
                'model': context_model.state_dict(), 'context': context.cpu(),
                'args': jsonable(vars(args)), 'data_stats': jsonable(data.stats)}, output / 'context.pt')
    del optimizer, context_model, sampler, context_sampler
    model = make_model(data, context, model_config(args), device)
    del context
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_score, bad_evaluations = -math.inf, 0
    use_validation = args.validation_ratio > 0
    if use_validation and not data.valid_user_items:
        raise ValueError('The requested validation holdout produced no eligible users')
    selection_key = 'recall@' + str(20 if 20 in args.ks else max(args.ks))
    validation = None
    for epoch in range(args.epochs):
        sampler = NeighborSampler(data.joint_graph, args.fanout, args.seed + 10000 + epoch)
        metrics = train_epoch(model, data, sampler, optimizer, args, rng, 'recommendation', log_path)
        emit(dict(stage='recommendation_epoch', epoch=epoch + 1, **metrics), log_path)
        improved = False
        if use_validation:
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
        checkpoint = dict(model_name='ARC', stage='recommendation',
                          architecture_version=ARCHITECTURE_VERSION,
                          model=model.state_dict(), model_config=model_config(args),
                          optimizer=optimizer.state_dict(), args=jsonable(vars(args)),
                          data_stats=jsonable(data.stats), epoch=epoch + 1,
                          validation=validation, best_score=best_score if use_validation else None,
                          selection='validation' if use_validation else 'final_epoch')
        torch.save(checkpoint, output / 'last.pt')
        if improved:
            torch.save(checkpoint, output / 'best.pt')
        if use_validation and bad_evaluations >= args.patience:
            break
    selected_path = output / ('best.pt' if use_validation else 'last.pt')
    checkpoint = torch.load(selected_path, map_location=device)
    check_checkpoint(checkpoint)
    model.load_state_dict(checkpoint['model'])
    test = evaluate(model, data, eval_sampler, split='test', ks=args.ks,
                    max_users=args.eval_users, samples=args.eval_samples, seed=args.seed)
    result = dict(model_name='ARC', dataset=args.dataset, smoke_test=args.smoke_test,
                  architecture_version=ARCHITECTURE_VERSION, variant=args.variant, seed=args.seed,
                  data_fingerprint=data.fingerprint, split_seed=args.split_seed,
                  split_mode=args.split_mode, validation_ratio=args.validation_ratio,
                  propagation='full_graph' if args.fanout == 0 else 'sampled_approximation',
                  diagnostic=bool(args.smoke_test or args.steps_per_epoch or args.eval_users),
                  selected_epoch=checkpoint['epoch'], checkpoint=str(selected_path),
                  selection=checkpoint['selection'], validation=checkpoint['validation'], test=test)
    write_json(output / 'results.json', result)
    emit(dict(stage='finished', **result), log_path)
    return result


def _beta_fraction(a, b, x):
    """Continued fraction for the regularized incomplete beta (Lentz method)."""
    tiny, tolerance = 1e-300, 3e-14
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    value = d
    for index in range(1, 1001):
        twice = 2 * index
        coefficient = index * (b - index) * x / ((qam + twice) * (a + twice))
        d = 1.0 + coefficient * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + coefficient / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        value *= d * c
        coefficient = -(a + index) * (qab + index) * x / ((a + twice) * (qap + twice))
        d = 1.0 + coefficient * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + coefficient / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        change = d * c
        value *= change
        if abs(change - 1.0) < tolerance:
            return value
    raise ArithmeticError('Incomplete beta continued fraction failed to converge')


def paired_t_test(values, baseline):
    """Two-sided paired Student t-test of real per-seed scores, without SciPy."""
    values, baseline = np.asarray(values, dtype=float), np.asarray(baseline, dtype=float)
    if values.ndim != 1 or baseline.shape != values.shape or len(values) < 2:
        raise ValueError('A paired t-test needs at least two matched run scores')
    if not (np.isfinite(values).all() and np.isfinite(baseline).all()):
        raise ValueError('Paired scores must be finite')
    difference = values - baseline
    mean, std = float(difference.mean()), float(difference.std(ddof=1))
    result = dict(n_pairs=len(values), mean_difference=mean, df=len(values) - 1,
                  alternative='two-sided')
    if std == 0:
        # Zero variance makes Student's t statistic undefined; do not invent p-values.
        return dict(result, t=None, p_value=None, status='undefined_zero_variance')
    statistic = mean / (std / math.sqrt(len(values)))
    a, b = (len(values) - 1) / 2.0, 0.5
    x = (len(values) - 1) / ((len(values) - 1) + statistic * statistic)
    if x >= 1.0:
        probability = 1.0
    elif x <= 0.0:
        probability = 0.0
    else:
        scale = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                         + a * math.log(x) + b * math.log1p(-x))
        if x < (a + 1.0) / (a + b + 2.0):
            probability = scale * _beta_fraction(a, b, x) / a
        else:
            probability = 1.0 - scale * _beta_fraction(b, a, 1.0 - x) / b
    return dict(result, t=statistic, p_value=min(1.0, max(0.0, probability)), status='ok')


def summarize_runs(results, baseline_path=None):
    """Save every seed result and aggregate only observed, comparable scores."""
    if not results:
        raise ValueError('Cannot summarize an empty run list')
    seeds = [run['seed'] for run in results]
    if len(set(seeds)) != len(seeds):
        raise ValueError('Run seeds must be unique')
    if len({run['data_fingerprint'] for run in results}) != 1:
        raise ValueError('Independent runs must share a fixed data split')
    metric_names = sorted(key for key in results[0]['test'] if '@' in key)
    metrics = {}
    for metric in metric_names:
        scores = np.asarray([run['test'][metric] for run in results], dtype=float)
        if not np.isfinite(scores).all():
            raise ValueError('Non-finite run score: ' + metric)
        metrics[metric] = dict(mean=float(scores.mean()),
                               std=float(scores.std(ddof=1)) if len(scores) > 1 else None)
    summary = dict(model_name='ARC', architecture_version=ARCHITECTURE_VERSION,
                   dataset=results[0]['dataset'], variant=results[0]['variant'],
                   data_fingerprint=results[0]['data_fingerprint'], seeds=seeds,
                   n_runs=len(results), metrics=metrics, std_definition='sample (ddof=1)',
                   runs=results)
    if baseline_path is not None:
        baseline = json.loads(Path(baseline_path).read_text(encoding='utf-8'))
        baseline_runs = baseline.get('runs', [])
        baseline_seeds = [run['seed'] for run in baseline_runs]
        if len(set(baseline_seeds)) != len(baseline_seeds) or set(baseline_seeds) != set(seeds):
            raise ValueError('Paired baseline must contain exactly the same distinct training seeds')
        by_seed = {run['seed']: run for run in baseline_runs}
        for run in results:
            other = by_seed[run['seed']]
            for field in ('dataset', 'data_fingerprint'):
                if run[field] != other.get(field):
                    raise ValueError('Paired baseline mismatch: ' + field)
            for compared in (run, other):
                if compared.get('diagnostic', True) or compared.get('smoke_test', False):
                    raise ValueError('Paired tests require complete runs, not diagnostic results')
                if not compared['test'].get('full_catalog') or compared['test'].get('user_subset'):
                    raise ValueError('Paired tests require full-catalog evaluation for all test users')
            for field in ('users', 'evaluation_users_sha256'):
                if not run['test'].get(field) or run['test'][field] != other['test'].get(field):
                    raise ValueError('Paired baseline evaluation-user mismatch: ' + field)
        summary['paired_baseline'] = str(baseline_path)
        summary['paired_tests'] = {
            metric: paired_t_test([run['test'][metric] for run in results],
                                  [by_seed[seed]['test'][metric] for seed in seeds])
            for metric in metric_names}
    return summary


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu')
                          if args.device == 'auto' else args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use --device cpu')
    checkpoint = None
    if args.eval_only:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        check_checkpoint(checkpoint)
        saved = checkpoint['args']
        for name in ('dataset', 'seed', 'split_seed', 'split_mode', 'validation_ratio',
                     'inverse', 'fanout'):
            setattr(args, name, saved[name])
        seed_everything(args.seed)
    data = load_dataset(args.data_root, args.dataset, seed=args.split_seed,
                        val_ratio=args.validation_ratio, inverse=args.inverse,
                        split_mode=args.split_mode)
    if args.codes is None:
        args.codes = math.ceil(data.stats['n_original_relations'] / args.code_divisor)
    if args.eval_only:
        if checkpoint['data_stats'] != jsonable(data.stats):
            raise ValueError('Checkpoint data/split fingerprint does not match loaded dataset')
        model = make_model(data, checkpoint['model']['e_context'],
                           checkpoint['model_config'], device)
        model.load_state_dict(checkpoint['model'], strict=True)
        sampler = NeighborSampler(data.joint_graph, args.fanout, args.seed + 70000)
        result = evaluate(model, data, sampler, split='test', ks=args.ks,
                          max_users=args.eval_users, samples=args.eval_samples, seed=args.seed)
        output = args.checkpoint.resolve().parent
        result = dict(model_name='ARC', seed=args.seed, data_fingerprint=data.fingerprint, **result)
        emit(result, output / 'evaluation.jsonl')
        write_json(output / 'evaluation.json', result)
        return result
    output = args.output_dir / args.dataset
    if args.variant != 'arc':
        output = output / args.variant
    output = output / ('smoke' if args.smoke_test else 'train')
    if (output / 'summary.json').exists():
        raise FileExistsError('Experiment already exists: ' + str(output))
    results = []
    for seed in args.training_seeds:
        run_args = copy.deepcopy(args)
        run_args.seed = seed
        run_output = output / ('seed_' + str(seed)) if args.runs > 1 else output
        results.append(run_training(run_args, data, device, run_output))
    summary = summarize_runs(results, args.paired_baseline)
    write_json(output / 'summary.json', summary)
    emit(dict(stage='experiment_finished', dataset=args.dataset, variant=args.variant,
              seeds=args.training_seeds, metrics=summary['metrics'], summary=str(output / 'summary.json')))
    return results[0] if args.runs == 1 else summary


if __name__ == '__main__':
    main()
