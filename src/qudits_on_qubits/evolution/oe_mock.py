"""Provide local models for OpenEvolve: a fixed test candidate and random skeleton mutations."""

import ast
import os
from pathlib import Path
import random
import re

from .oe_candidates import skeleton_program
from .skeleton_fit import MAX_SKELETON_CZ

# The template puts the parent program in the first code block after this heading.
_CURRENT = re.compile(r"# Current candidate\n```\w*\n(.*?)```", re.DOTALL)
MUTATIONS = ("remove", "retarget", "swap", "move", "insert")


class MockLLM:
    """Return the local test candidate without a network request."""

    def __init__(self, config):
        self.model = config.name

    async def generate(self, prompt, **kwargs):
        return "```python\n" + Path(os.environ["QOQ_OE_MOCK_PROGRAM"]).read_text() + "\n```"

    async def generate_with_context(self, system_message, messages, **kwargs):
        return await self.generate("")


def mutate_skeleton(width, pairs, rng, *, kind=None, qubits=None):
    """Apply one change to the CZ list. Restrict edited pairs to qubits when supplied."""
    pairs = list(pairs)
    kind = kind or rng.choice(MUTATIONS if len(pairs) > 1 else ("retarget", "insert"))
    if kind not in MUTATIONS:
        raise ValueError("The mutation operation is not supported")
    qubits = list(range(width)) if qubits is None else list(qubits)
    indices = [i for i, pair in enumerate(pairs) if all(q in qubits for q in pair)]
    if kind != "insert" and not indices:
        return pairs
    if kind == "remove":
        del pairs[rng.choice(indices)]
    elif kind == "retarget":
        index = rng.choice(indices)
        keep = pairs[index][rng.randrange(2)]
        pairs[index] = (keep, rng.choice([q for q in qubits if q != keep]))
    elif kind == "swap":
        adjacent = [i for i in indices if i + 1 in indices]
        if not adjacent:
            return pairs
        index = rng.choice(adjacent)
        pairs[index], pairs[index + 1] = pairs[index + 1], pairs[index]
    elif kind == "move":
        pairs.insert(rng.randrange(len(pairs)), pairs.pop(rng.choice(indices)))
    else:
        if len(pairs) >= MAX_SKELETON_CZ:
            return pairs
        pairs.insert(rng.randrange(len(pairs) + 1), tuple(rng.sample(qubits, 2)))
    return pairs


class SkeletonMutator:
    """Offline model: read the parent skeleton from the prompt and return a random mutation.

    It sends no network request. It gives a search baseline without an LLM.
    """

    def __init__(self, config):
        self.model = config.name
        self.rng = random.Random(config.random_seed)

    async def generate_with_context(self, system_message, messages, **kwargs):
        match = _CURRENT.search(messages[-1]["content"])
        if match is None:
            raise ValueError("The prompt has no current candidate")
        tree = ast.parse(match.group(1))
        data = ast.literal_eval(tree.body[-1].value)
        pairs = [tuple(pair) for pair in data["cz"]]
        for _ in range(1 + (self.rng.random() < 0.3)):
            pairs = mutate_skeleton(data["num_qubits"], pairs, self.rng)
        return "```python\n" + skeleton_program(data["num_qubits"], pairs) + "```"

    async def generate(self, prompt, **kwargs):
        return await self.generate_with_context("", [{"role": "user", "content": prompt}])
