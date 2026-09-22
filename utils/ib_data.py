"""Leakage-aware data loading and CSR graph sampling for the IB recommender.

The paper protocol randomly splits unique positive interactions 80/20 with a
fixed split seed. The supplied split can alternatively be retained explicitly.
Only training positives enter a graph; optional validation uses training only.
Items occupy the first ``n_items`` entity indices, while graph node indices put
users first and all entities after them.  No SciPy/networkx dependency is needed.
"""

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import pickle
from typing import Dict, Iterator, Optional, Set, Tuple

import numpy as np


def _randint(rng, high, size=None):
    if hasattr(rng, "integers"):
        return rng.integers(high, size=size)
    return rng.randint(high, size=size)


@dataclass
class Graph:
    """CSR neighborhoods; relation -1 denotes a user-item interaction edge."""

    offsets: np.ndarray
    neighbors: np.ndarray
    relations: np.ndarray

    @property
    def n_nodes(self):
        return int(len(self.offsets) - 1)

    @property
    def n_edges(self):
        return int(len(self.neighbors))

    @classmethod
    def from_edges(cls, n_nodes, rows, neighbors, relations):
        rows = np.asarray(rows, dtype=np.int64)
        neighbors = np.asarray(neighbors, dtype=np.int64)
        relations = np.asarray(relations, dtype=np.int64)
        if not (len(rows) == len(neighbors) == len(relations)):
            raise ValueError("Edge row, neighbor, and relation lengths differ")
        if len(rows) and (rows.min() < 0 or neighbors.min() < 0
                          or rows.max() >= n_nodes or neighbors.max() >= n_nodes):
            raise ValueError("Graph edge endpoint is outside the node universe")
        order = np.argsort(rows, kind="stable")
        counts = np.bincount(rows, minlength=n_nodes)
        offsets = np.empty(n_nodes + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(counts, out=offsets[1:])
        return cls(offsets, neighbors[order], relations[order])

    def sample_table(self, fanout, rng):
        """Sample distinct incident edges, padding missing slots with invalids.

        Multiple relation edges to one neighbor remain distinct edges.  Padding
        points to the row node itself and must be ignored using ``valid``.
        """
        if fanout < 1:
            raise ValueError("fanout must be positive")
        neighbors = np.repeat(np.arange(self.n_nodes, dtype=np.int64)[:, None],
                              fanout, axis=1)
        relations = np.full((self.n_nodes, fanout), -1, dtype=np.int64)
        valid = np.zeros((self.n_nodes, fanout), dtype=np.bool_)
        for node in range(self.n_nodes):
            start, end = int(self.offsets[node]), int(self.offsets[node + 1])
            degree = end - start
            if not degree:
                continue
            count = min(degree, fanout)
            if degree <= fanout:
                indices = slice(start, end)
            else:
                indices = start + rng.choice(degree, size=fanout, replace=False)
            neighbors[node, :count] = self.neighbors[indices]
            relations[node, :count] = self.relations[indices]
            valid[node, :count] = True
        return neighbors, relations, valid


@dataclass
class Dataset:
    name: str
    n_users: int
    n_items: int
    n_entities: int
    n_relations: int
    train_pairs: np.ndarray
    valid_pairs: np.ndarray
    test_pairs: np.ndarray
    train_user_items: Dict[int, Set[int]]
    valid_user_items: Dict[int, Set[int]]
    test_user_items: Dict[int, Set[int]]
    rho: np.ndarray
    ui_graph: Graph
    joint_graph: Graph
    kg_triples: np.ndarray
    user_ids: np.ndarray
    item_ids: np.ndarray
    entity_ids: np.ndarray
    relation_ids: np.ndarray
    inverse_relation_offset: Optional[int]
    stats: dict
    fingerprint: str
    _eligible_users: np.ndarray = field(init=False, repr=False)
    _train_lists: dict = field(init=False, repr=False)
    _dense_negatives: dict = field(init=False, repr=False)

    def __post_init__(self):
        self._train_lists = {
            user: np.asarray(sorted(items), dtype=np.int64)
            for user, items in self.train_user_items.items() if items
        }
        self._eligible_users = np.asarray(
            [user for user, items in self._train_lists.items()
             if len(items) < self.n_items], dtype=np.int64)
        self._dense_negatives = {}
        if not len(self._eligible_users):
            raise ValueError("No user has both a training positive and an unseen item")

    @property
    def n_nodes(self):
        return self.n_users + self.n_entities

    @property
    def metadata(self):
        return self.stats

    def sample_triples(self, batch_size, rng):
        """Uniform-user BPR sampling using training labels only.

        The positive and negative belong to the same user.  Validation/test
        positives are intentionally not consulted by the negative sampler.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        users = self._eligible_users[
            _randint(rng, len(self._eligible_users), size=batch_size)]
        positives = np.empty(batch_size, dtype=np.int64)
        negatives = np.asarray(_randint(rng, self.n_items, size=batch_size),
                               dtype=np.int64)
        for index, user in enumerate(users):
            user = int(user)
            history = self._train_lists[user]
            positives[index] = history[int(_randint(rng, len(history)))]
            excluded = self.train_user_items[user]
            if len(excluded) > self.n_items // 2:
                if user not in self._dense_negatives:
                    self._dense_negatives[user] = np.asarray(
                        [item for item in range(self.n_items) if item not in excluded],
                        dtype=np.int64)
                pool = self._dense_negatives[user]
                negatives[index] = pool[int(_randint(rng, len(pool)))]
            else:
                while int(negatives[index]) in excluded:
                    negatives[index] = int(_randint(rng, self.n_items))
        return users, positives, negatives

    def iter_triples(self, batch_size, rng) -> Iterator[Tuple[np.ndarray, ...]]:
        """Yield ``len(train_pairs)`` uniform-user samples per epoch."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        for start in range(0, len(self.train_pairs), batch_size):
            yield self.sample_triples(min(batch_size, len(self.train_pairs) - start), rng)


def _integer_array(value, name, columns):
    array = np.asarray(value)
    if array.size == 0:
        return np.empty((0, columns), dtype=np.int64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] < columns:
        raise ValueError("{} must have at least {} columns".format(name, columns))
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError("{} contains nonnumeric values".format(name))
    if not np.all(np.isfinite(array)) or not np.all(array == np.floor(array)):
        raise ValueError("{} must contain finite integer values".format(name))
    return array.astype(np.int64, copy=False)


def _load_interactions(directory, split):
    # The Book .npy arrays contain the same rows as its pandas-backed pickles.
    # Prefer plain arrays so ARC does not require pandas merely to load data.
    path = directory / (split + "_data.npy")
    if path.exists():
        array = np.load(str(path), allow_pickle=False)
    else:
        path = directory / (split + "_data.pkl")
        if not path.exists():
            raise FileNotFoundError("Missing {}_data.npy/.pkl in {}".format(split, directory))
        # These are the user-supplied trusted archive's pickle files.
        with path.open("rb") as handle:
            array = pickle.load(handle, encoding="bytes")
    array = _integer_array(array, str(path), 2)
    if array.shape[1] not in (2, 3):
        raise ValueError("{} must be [user, item] or [user, item, binary_label]".format(path))
    if len(array) and np.any(array[:, :2] < 0):
        raise ValueError("{} contains negative user/item IDs".format(path))
    if array.shape[1] == 3:
        if not np.all(np.isin(array[:, 2], [0, 1])):
            raise ValueError("{} requires binary interaction labels 0/1".format(path))
        positive = array[array[:, 2] == 1, :2]
    else:
        positive = array[:, :2]
    return array, np.unique(positive, axis=0), path.name


def _user_items(pairs):
    result = {}
    for user, item in pairs:
        result.setdefault(int(user), set()).add(int(item))
    return result


def _pairs_from_user_items(user_items):
    pairs = [(user, item) for user in sorted(user_items)
             for item in sorted(user_items[user])]
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def _fingerprint(arrays, configuration):
    digest = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode("utf-8"))
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def load_dataset(data_root, dataset, seed=2024, val_ratio=0.0, inverse=False,
                 split_mode="random"):
    """Load the paper's 80/20 positive-interaction protocol.

    ``random`` splits the union of unique positive source interactions, using
    floor(0.8 * number of positives) training pairs and the remainder for test.
    ``provided`` preserves the supplied train/test assignment. ``seed`` governs
    the data split only and should stay fixed across independent training runs.
    Optional validation is held out only from the resulting training split;
    the paper default uses all 80% training positives without this holdout.

    Movie/music catalog files have columns ``external_item_id, entity_id``;
    their second column is aligned with the processed interactions and KG.
    Original IDs are exposed by ``*_ids`` arrays. By default the KG propagates
    in both directions using the SAME original relation identity. ``rho`` counts
    triples in the original KG, before adding reverse propagation edges, and
    ``kg_triples`` retains those original directed triples. Duplicate copies of
    a factual triple are removed because the KG is a set.

    Explicit ``inverse=True`` retains the legacy extension with distinct inverse
    relation identities and an equally weighted original/inverse relation prior.
    That extension changes the relation random variable and is not Eq. (6)'s
    original-KG protocol. ``relation_ids`` always contains the original types;
    legacy inverse types start at ``inverse_relation_offset``.
    The ID vocabulary may include test users/items, but test labels and edges
    never enter a training graph or sampler.
    """
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be in [0, 1)")
    if split_mode not in ("random", "provided"):
        raise ValueError("split_mode must be 'random' or 'provided'")
    root = Path(data_root).expanduser()
    directory = root / dataset
    if not directory.is_dir() and root.name == dataset:
        directory = root
    if not directory.is_dir():
        raise FileNotFoundError("Dataset directory does not exist: {}".format(directory))
    train_rows, original_train, train_source = _load_interactions(directory, "train")
    test_rows, original_test, test_source = _load_interactions(directory, "test")
    if not len(original_train):
        raise ValueError("No positive training interactions in {}".format(directory))
    original_overlap = set(map(tuple, original_train)) & set(map(tuple, original_test))
    if original_overlap:
        raise ValueError("{} positive pairs overlap provided train/test splits".format(
            len(original_overlap)))
    positive_pairs = np.unique(np.concatenate([original_train, original_test]), axis=0)
    if split_mode == "random":
        if len(positive_pairs) < 2:
            raise ValueError("The 80/20 split requires at least two positive interactions")
        permutation = np.random.default_rng(seed).permutation(len(positive_pairs))
        n_train = (4 * len(positive_pairs)) // 5
        split_train = positive_pairs[permutation[:n_train]]
        split_test = positive_pairs[permutation[n_train:]]
    else:
        split_train, split_test = original_train, original_test

    kg_path = directory / "kg_final.npy"
    if kg_path.exists():
        raw_kg = np.load(str(kg_path), allow_pickle=False)
    else:
        kg_path = directory / "kg_final.txt"
        raw_kg = np.loadtxt(str(kg_path), dtype=np.int64)
    raw_kg = _integer_array(raw_kg, str(kg_path), 3)
    if raw_kg.shape[1] != 3 or not len(raw_kg) or np.any(raw_kg < 0):
        raise ValueError("Knowledge graph must contain nonnegative [head, relation, tail] triples")
    kg_input_count = len(raw_kg)
    raw_kg = np.unique(raw_kg, axis=0)

    user_ids = np.unique(np.concatenate([train_rows[:, 0], test_rows[:, 0]]))
    observed_items = np.unique(np.concatenate([train_rows[:, 1], test_rows[:, 1]]))
    catalog_path = directory / "item_index2entity_id.txt"
    catalog_items = np.empty(0, dtype=np.int64)
    if catalog_path.exists():
        catalog = _integer_array(np.loadtxt(str(catalog_path), dtype=np.int64),
                                 str(catalog_path), 2)
        catalog_items = np.unique(catalog[:, 1])
        if len(catalog_items) and catalog_items.min() < 0:
            raise ValueError("Catalog contains negative entity IDs")
    item_ids = np.union1d(observed_items, catalog_items)
    kg_entities = np.unique(raw_kg[:, [0, 2]])
    entity_ids = np.concatenate([item_ids, np.setdiff1d(kg_entities, item_ids)])
    relation_ids = np.unique(raw_kg[:, 1])
    n_users, n_items, n_entities = len(user_ids), len(item_ids), len(entity_ids)
    n_base_relations = len(relation_ids)
    if n_items < 2:
        raise ValueError("At least two catalog items are required for BPR training")

    def remap_pairs(pairs):
        return np.column_stack([np.searchsorted(user_ids, pairs[:, 0]),
                                np.searchsorted(item_ids, pairs[:, 1])]).astype(np.int64)

    mapped_train = remap_pairs(split_train)
    mapped_test = remap_pairs(split_test)
    train_user_items = _user_items(mapped_train)
    valid_user_items = {}
    rng = np.random.default_rng(seed)
    if val_ratio:
        for user in sorted(train_user_items):
            items = np.asarray(sorted(train_user_items[user]), dtype=np.int64)
            if len(items) < 2:
                continue
            n_valid = min(len(items) - 1, max(1, int(round(len(items) * val_ratio))))
            held_out = set(map(int, rng.choice(items, n_valid, replace=False)))
            valid_user_items[user] = held_out
            train_user_items[user] -= held_out
    train_pairs = _pairs_from_user_items(train_user_items)
    valid_pairs = _pairs_from_user_items(valid_user_items)
    test_user_items = _user_items(mapped_test)
    test_pairs = _pairs_from_user_items(test_user_items)

    # entity_ids deliberately is not globally sorted: all candidate items first.
    entity_map = {int(raw): index for index, raw in enumerate(entity_ids)}
    kg = np.empty_like(raw_kg, dtype=np.int64)
    kg[:, 0] = np.fromiter((entity_map[int(x)] for x in raw_kg[:, 0]), dtype=np.int64)
    kg[:, 1] = np.searchsorted(relation_ids, raw_kg[:, 1])
    kg[:, 2] = np.fromiter((entity_map[int(x)] for x in raw_kg[:, 2]), dtype=np.int64)
    original_relation_counts = np.bincount(kg[:, 1], minlength=n_base_relations)
    inverse_offset = n_base_relations if inverse else None
    reverse_kg = kg[:, [2, 1, 0]].copy()
    if inverse:
        reverse_kg[:, 1] += n_base_relations
        kg = np.concatenate([kg, reverse_kg], axis=0)
        propagation_kg = kg
        relation_counts = np.tile(original_relation_counts, 2)
    else:
        # The KG is a set: a self-loop or an already-present reverse triple
        # should not receive a duplicate message when making it bidirectional.
        propagation_kg = np.unique(np.concatenate([kg, reverse_kg], axis=0), axis=0)
        relation_counts = original_relation_counts
    n_relations = n_base_relations * (2 if inverse else 1)
    rho = (relation_counts / float(relation_counts.sum())).astype(np.float32)

    ui_rows = np.concatenate([train_pairs[:, 0], n_users + train_pairs[:, 1]])
    ui_neighbors = np.concatenate([n_users + train_pairs[:, 1], train_pairs[:, 0]])
    ui_relations = np.full(len(ui_rows), -1, dtype=np.int64)
    # UI graph only needs user/item nodes; joint graph additionally holds KG-only entities.
    ui_graph = Graph.from_edges(n_users + n_items, ui_rows, ui_neighbors, ui_relations)
    joint_graph = Graph.from_edges(
        n_users + n_entities,
        np.concatenate([ui_rows, n_users + propagation_kg[:, 0]]),
        np.concatenate([ui_neighbors, n_users + propagation_kg[:, 2]]),
        np.concatenate([ui_relations, propagation_kg[:, 1]]))

    config = {"dataset": dataset, "seed": int(seed), "val_ratio": float(val_ratio),
              "inverse": bool(inverse), "split_mode": split_mode, "schema": 3}
    fingerprint = _fingerprint(
        [train_pairs, valid_pairs, test_pairs, kg, user_ids, item_ids, entity_ids,
         relation_ids], config)
    n_train_positive_rows = int(np.sum(train_rows[:, 2] == 1)) if train_rows.shape[1] == 3 else len(train_rows)
    n_test_positive_rows = int(np.sum(test_rows[:, 2] == 1)) if test_rows.shape[1] == 3 else len(test_rows)
    stats = dict(config)
    stats.update({
        "fingerprint": fingerprint,
        "source_files": {"train": train_source, "test": test_source, "kg": kg_path.name,
                         "catalog": catalog_path.name if catalog_path.exists() else None},
        "n_users": n_users, "n_items": n_items, "n_entities": n_entities,
        "n_nodes": n_users + n_entities, "n_relations": n_relations,
        "n_original_relations": n_base_relations,
        "source_train_rows": len(train_rows), "source_test_rows": len(test_rows),
        "source_train_positive_rows": n_train_positive_rows,
        "source_train_negative_rows": len(train_rows) - n_train_positive_rows,
        "source_test_positive_rows": n_test_positive_rows,
        "source_test_negative_rows": len(test_rows) - n_test_positive_rows,
        "duplicate_train_positives_removed": n_train_positive_rows - len(original_train),
        "duplicate_test_positives_removed": n_test_positive_rows - len(original_test),
        "provided_train_positive_pairs": len(original_train),
        "provided_test_positive_pairs": len(original_test),
        "total_unique_positive_pairs": len(positive_pairs),
        "source_raw_train_fraction": len(train_rows) / float(len(train_rows) + len(test_rows)),
        "source_positive_train_fraction": len(original_train) / float(len(positive_pairs)),
        "split_train_positive_pairs": len(split_train),
        "split_positive_train_fraction": len(split_train) / float(len(positive_pairs)),
        "train_positive_pairs": len(train_pairs), "valid_positive_pairs": len(valid_pairs),
        "test_positive_pairs": len(test_pairs), "train_users": len(train_user_items),
        "valid_users": len(valid_user_items), "test_users": len(test_user_items),
        "test_users_without_train_positives": len(set(test_user_items) - set(train_user_items)),
        "observed_item_ids": len(observed_items), "catalog_items": len(catalog_items),
        "kg_source_triples": kg_input_count, "kg_unique_original_triples": len(raw_kg),
        "kg_triples_with_inverse": len(kg),
        "kg_propagation_edges": len(propagation_kg),
        "kg_reverse_relations": "distinct inverse types (legacy)" if inverse else "same original type",
        "original_relation_counts": original_relation_counts.tolist(),
        "rho_source": "original KG and equally weighted inverse identities (legacy)" if inverse
                      else "unique original KG triples, before reverse propagation",
        "ui_directed_edges": ui_graph.n_edges,
        "joint_directed_edges": joint_graph.n_edges,
        "relation_counts": relation_counts.tolist(), "rho": rho.tolist(),
        "negative_exclusion": "training positives only; no validation/test labels",
    })
    return Dataset(dataset, n_users, n_items, n_entities, n_relations,
                   train_pairs, valid_pairs, test_pairs, train_user_items,
                   valid_user_items, test_user_items, rho, ui_graph, joint_graph,
                   kg, user_ids, item_ids, entity_ids, relation_ids, inverse_offset,
                   stats, fingerprint)
