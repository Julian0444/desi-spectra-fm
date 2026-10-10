"""Metadata-only stratified, group-preserving partitions. No model selection."""
from __future__ import annotations

from collections import Counter, defaultdict
import math
import numpy as np

SPLIT_SEED = 20261009
Z_EDGES = [0, .5, 1, 1.5, 2, 2.5]


def stratum(row):
    return f"z{int(np.searchsorted(Z_EDGES[1:], row['z'], side='right'))}:zwarn{int(row['zwarn'])}"


def quotas(counts, size):
    total = sum(counts.values())
    if not 0 <= size <= total:
        raise ValueError("Invalid split size")
    exact = {key: size * count / total for key, count in counts.items()}
    result = {key: math.floor(value) for key, value in exact.items()}
    order = sorted(counts, key=lambda key: (-(exact[key] - result[key]), key))
    for key in order[:size - sum(result.values())]:
        result[key] += 1
    return result


def make_splits(rows, protected_ids, *, relationships=None, seed=SPLIT_SEED,
                validation_size=10000, test_size=10000):
    """relationships maps ID to an externally evidenced group, never guesses it.

    Mixed groups use their modal stratum for allocation and retain all members.
    Within each stratum whole-group allocation minimizes distance to its quota.
    Stable tie breaks plus one fixed generator avoid selection by outcome.
    """
    ids = [r['object_id'] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate IDs")
    protected_ids = set(protected_ids)
    if not protected_ids <= set(ids):
        raise ValueError("Protected IDs missing from eligible population")
    if validation_size + test_size >= len(ids):
        raise ValueError("Split sizes leave no train objects")
    for r in rows:
        if not np.isfinite(r['z']) or r['z'] < 0 or r['zwarn'] not in (0, 1):
            raise ValueError("Ineligible metadata")
    relationships = relationships or {}
    if not set(relationships) <= set(ids):
        raise ValueError("Unknown relationship ID")
    groups = defaultdict(list)
    for row in sorted(rows, key=lambda r: r['object_id']):
        # Prefix prevents an external group identifier colliding with an object ID.
        group = 'related:' + str(relationships[row['object_id']]) if row['object_id'] in relationships else 'object:' + row['object_id']
        groups[group].append(row)
    buckets = defaultdict(list)
    mixed = []
    for gid, members in sorted(groups.items()):
        counts = Counter(stratum(row) for row in members)
        dominant = sorted(counts, key=lambda k: (-counts[k], k))[0]
        buckets[dominant].append(gid)
        if len(counts) > 1:
            mixed.append({"group": gid, "strata": dict(counts)})
    counts = {key: sum(len(groups[g]) for g in gs) for key, gs in buckets.items()}
    test_quota, val_quota = quotas(counts, test_size), quotas(counts, validation_size)
    rng = np.random.Generator(np.random.PCG64(seed))
    assignment = {}
    for key in sorted(buckets):
        order = [buckets[key][i] for i in rng.permutation(len(buckets[key]))]
        remaining = order
        for role, quota in [('test', test_quota[key]), ('validation', val_quota[key])]:
            selected, n = set(), 0
            for gid in remaining:
                members = groups[gid]
                if role == 'test' and any(r['object_id'] in protected_ids for r in members):
                    continue
                if abs(n + len(members) - quota) < abs(n - quota):
                    selected.add(gid)
                    n += len(members)
                    assignment[gid] = role
            remaining = [g for g in remaining if g not in selected]
        assignment.update({g: 'train' for g in remaining})
    splits = {k: [] for k in ('train', 'validation', 'test')}
    for gid, members in groups.items():
        for row in members:
            splits[assignment[gid]].append({**row, "group_id": gid, "stratum": stratum(row)})
    for role in splits:
        splits[role].sort(key=lambda row: row['object_id'])
    assert not protected_ids & {r['object_id'] for r in splits['test']}
    strata = {role: dict(sorted(Counter(r['stratum'] for r in records).items())) for role, records in splits.items()}
    rare = [k for k in sorted(counts) if len(buckets[k]) < 3 or test_quota[k] == 0 or val_quota[k] == 0]
    report = {"seed": seed, "requested": {"train": len(ids)-validation_size-test_size, "validation": validation_size, "test": test_size},
              "actual": {k: len(v) for k, v in splits.items()}, "strata": strata,
              "allocation_quotas": {"test": test_quota, "validation": val_quota}, "rare_strata": rare,
              "rare_definition": "fewer than 3 observed groups or zero validation/test quota; retained in all accounting",
              "observed_groups": len(groups), "mixed_stratum_groups": mixed,
              "protected_ids": len(protected_ids), "protected_groups": len({g for g, rs in groups.items() if any(r['object_id'] in protected_ids for r in rs)}),
              "physical_independence": "not established by these metadata"}
    return splits, report
