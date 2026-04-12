"""
Skill Definition Language (SDL)

A formal language for defining agent skills with type signatures,
composition rules, and dependency management.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Union


# ---------------------------------------------------------------------------
# 1. Skill Types — Type system for skill inputs/outputs
# ---------------------------------------------------------------------------

class SkillType(ABC):
    """Abstract base class for all types in the skill type system."""

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        ...

    @abstractmethod
    def __hash__(self) -> int:
        ...

    @abstractmethod
    def __repr__(self) -> str:
        ...

    def is_primitive(self) -> bool:
        return isinstance(self, PrimitiveType)

    def is_composite(self) -> bool:
        return isinstance(self, CompositeType)

    def is_special(self) -> bool:
        return isinstance(self, SpecialType)


class PrimitiveType(SkillType):
    """Primitive types in the skill type system."""

    class Kind(Enum):
        INT = auto()
        FLOAT = auto()
        STRING = auto()
        BOOL = auto()
        BYTES = auto()
        OPCODE = auto()

    def __init__(self, kind: "PrimitiveType.Kind") -> None:
        self.kind = kind

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PrimitiveType) and self.kind == other.kind

    def __hash__(self) -> int:
        return hash(("PrimitiveType", self.kind))

    def __repr__(self) -> str:
        return self.kind.name


# Singleton primitive types
INT = PrimitiveType(PrimitiveType.Kind.INT)
FLOAT = PrimitiveType(PrimitiveType.Kind.FLOAT)
STRING = PrimitiveType(PrimitiveType.Kind.STRING)
BOOL = PrimitiveType(PrimitiveType.Kind.BOOL)
BYTES = PrimitiveType(PrimitiveType.Kind.BYTES)
OPCODE = PrimitiveType(PrimitiveType.Kind.OPCODE)


class CompositeType(SkillType):
    """Composite types built from other types."""

    class Kind(Enum):
        LIST = auto()
        MAP = auto()
        TUPLE = auto()
        OPTIONAL = auto()

    def __init__(self, kind: "CompositeType.Kind", params: List[SkillType]) -> None:
        self.kind = kind
        self.params = list(params)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, CompositeType)
            and self.kind == other.kind
            and self.params == other.params
        )

    def __hash__(self) -> int:
        return hash(("CompositeType", self.kind, tuple(self.params)))

    def __repr__(self) -> str:
        if self.kind == CompositeType.Kind.LIST:
            return f"LIST<{self.params[0]}>"
        elif self.kind == CompositeType.Kind.MAP:
            return f"MAP<{self.params[0]},{self.params[1]}>"
        elif self.kind == CompositeType.Kind.TUPLE:
            return f"TUPLE<{','.join(repr(p) for p in self.params)}>"
        elif self.kind == CompositeType.Kind.OPTIONAL:
            return f"OPTIONAL<{self.params[0]}>"
        return f"CompositeType({self.kind})"


def LIST(inner: SkillType) -> CompositeType:
    """Shorthand: LIST<T>"""
    return CompositeType(CompositeType.Kind.LIST, [inner])


def MAP(key: SkillType, value: SkillType) -> CompositeType:
    """Shorthand: MAP<K,V>"""
    return CompositeType(CompositeType.Kind.MAP, [key, value])


def TUPLE(*params: SkillType) -> CompositeType:
    """Shorthand: TUPLE<T1,...Tn>"""
    return CompositeType(CompositeType.Kind.TUPLE, list(params))


def OPTIONAL(inner: SkillType) -> CompositeType:
    """Shorthand: OPTIONAL<T>"""
    return CompositeType(CompositeType.Kind.OPTIONAL, [inner])


class SpecialType(SkillType):
    """Special domain-specific types."""

    class Kind(Enum):
        BYTECODE = auto()
        AGENT_ID = auto()
        TRUST_LEVEL = auto()
        SIGNAL_REF = auto()

    def __init__(self, kind: "SpecialType.Kind") -> None:
        self.kind = kind

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SpecialType) and self.kind == other.kind

    def __hash__(self) -> int:
        return hash(("SpecialType", self.kind))

    def __repr__(self) -> str:
        return self.kind.name


# Singleton special types
BYTECODE = SpecialType(SpecialType.Kind.BYTECODE)
AGENT_ID = SpecialType(SpecialType.Kind.AGENT_ID)
TRUST_LEVEL = SpecialType(SpecialType.Kind.TRUST_LEVEL)
SIGNAL_REF = SpecialType(SpecialType.Kind.SIGNAL_REF)


class TypeCheckError(Exception):
    """Raised when type checking fails."""
    pass


class TypeChecker:
    """Validate that types are well-formed."""

    @staticmethod
    def check(t: SkillType) -> List[str]:
        """Return a list of errors for the given type. Empty list means valid."""
        errors: List[str] = []
        TypeChecker._check_recursive(t, errors)
        return errors

    @staticmethod
    def _check_recursive(t: SkillType, errors: List[str]) -> None:
        if isinstance(t, PrimitiveType) or isinstance(t, SpecialType):
            return  # primitives and specials are always well-formed
        elif isinstance(t, CompositeType):
            if t.kind == CompositeType.Kind.LIST:
                if len(t.params) != 1:
                    errors.append(f"LIST requires exactly 1 type parameter, got {len(t.params)}")
                else:
                    TypeChecker._check_recursive(t.params[0], errors)
            elif t.kind == CompositeType.Kind.MAP:
                if len(t.params) != 2:
                    errors.append(f"MAP requires exactly 2 type parameters, got {len(t.params)}")
                else:
                    TypeChecker._check_recursive(t.params[0], errors)
                    TypeChecker._check_recursive(t.params[1], errors)
            elif t.kind == CompositeType.Kind.TUPLE:
                if len(t.params) < 1:
                    errors.append("TUPLE requires at least 1 type parameter")
                else:
                    for p in t.params:
                        TypeChecker._check_recursive(p, errors)
            elif t.kind == CompositeType.Kind.OPTIONAL:
                if len(t.params) != 1:
                    errors.append(f"OPTIONAL requires exactly 1 type parameter, got {len(t.params)}")
                else:
                    TypeChecker._check_recursive(t.params[0], errors)

    @staticmethod
    def is_well_formed(t: SkillType) -> bool:
        return len(TypeChecker.check(t)) == 0

    @staticmethod
    def is_assignable(target: SkillType, source: SkillType) -> bool:
        """
        Check if a value of `source` type can be assigned to a `target` type.
        Exact match for now (covariance could be added later).
        """
        return target == source

    @staticmethod
    def types_compatible(expected: Dict[str, SkillType], provided: Dict[str, SkillType]) -> List[str]:
        """Check that `provided` types match `expected` types (by name). Returns errors."""
        errors: List[str] = []
        for name, exp_type in expected.items():
            if name not in provided:
                errors.append(f"Missing required type: '{name}'")
            elif not TypeChecker.is_assignable(exp_type, provided[name]):
                errors.append(
                    f"Type mismatch for '{name}': expected {exp_type}, got {provided[name]}"
                )
        for name in provided:
            if name not in expected:
                errors.append(f"Unexpected type provided: '{name}'")
        return errors


# ---------------------------------------------------------------------------
# 2. Skill Signature — Function-like type signature
# ---------------------------------------------------------------------------

class SkillTag(Enum):
    PURE = auto()      # no side effects
    IO = auto()        # reads/writes external resources
    A2A = auto()       # inter-agent communication
    BLOCKING = auto()  # may block execution


@dataclass(frozen=True)
class SkillSignature:
    """Function-like type signature for a skill."""

    name: str
    inputs: Dict[str, SkillType] = field(default_factory=dict)
    outputs: Dict[str, SkillType] = field(default_factory=dict)
    description: str = ""
    tags: frozenset[SkillTag] = frozenset()

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError(f"Skill name must be a non-empty string, got: {self.name!r}")

    def input_names(self) -> List[str]:
        return list(self.inputs.keys())

    def output_names(self) -> List[str]:
        return list(self.outputs.keys())

    def has_tag(self, tag: SkillTag) -> bool:
        return tag in self.tags

    def __repr__(self) -> str:
        ins = ", ".join(f"{k}: {v}" for k, v in self.inputs.items())
        outs = ", ".join(f"{k}: {v}" for k, v in self.outputs.items())
        tags_str = f" [{','.join(t.name for t in self.tags)}]" if self.tags else ""
        return f"skill {self.name}({ins}) -> ({outs}){tags_str}"


# ---------------------------------------------------------------------------
# 3. Skill Definition — Complete skill definition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SkillDefinition:
    """Complete definition of a skill."""

    signature: SkillSignature
    dependencies: List[str] = field(default_factory=list)
    implementation_ref: str = ""
    version: str = "0.0.0"
    author: str = ""

    @property
    def name(self) -> str:
        return self.signature.name

    def __repr__(self) -> str:
        return (
            f"SkillDefinition({self.signature.name!r}, "
            f"version={self.version!r}, "
            f"deps={self.dependencies})"
        )


# ---------------------------------------------------------------------------
# 4. Skill Registry — Central registry of all skills
# ---------------------------------------------------------------------------

class SkillRegistry:
    """Central registry of all skill definitions."""

    def __init__(self) -> None:
        self._skills: Dict[str, SkillDefinition] = {}

    def register(self, skill_def: SkillDefinition) -> None:
        """Register a skill definition. Overwrites if name exists."""
        self._skills[skill_def.name] = skill_def

    def unregister(self, name: str) -> None:
        """Remove a skill from the registry."""
        if name in self._skills:
            del self._skills[name]

    def lookup(self, name: str) -> Optional[SkillDefinition]:
        """Look up a skill by name. Returns None if not found."""
        return self._skills.get(name)

    def find_by_tag(self, tag: SkillTag) -> List[SkillDefinition]:
        """Find all skills with a given tag."""
        return [s for s in self._skills.values() if s.signature.has_tag(tag)]

    def find_by_output_type(self, output_type: SkillType) -> List[SkillDefinition]:
        """Find all skills that produce a given output type."""
        results: List[SkillDefinition] = []
        for skill in self._skills.values():
            if output_type in skill.signature.outputs.values():
                results.append(skill)
        return results

    def all_skills(self) -> List[SkillDefinition]:
        """Return all registered skills."""
        return list(self._skills.values())

    def skill_names(self) -> List[str]:
        """Return all registered skill names."""
        return list(self._skills.keys())

    def validate_dependencies(self, skill: SkillDefinition) -> ValidationResult:
        """Check that all dependencies of a skill exist in the registry and are compatible."""
        errors: List[str] = []
        warnings: List[str] = []

        for dep_name in skill.dependencies:
            dep = self.lookup(dep_name)
            if dep is None:
                errors.append(f"Dependency '{dep_name}' not found in registry")
            else:
                # Check for cycles — if dep depends (transitively) on skill
                if self._has_transitive_dependency(dep_name, skill.name):
                    errors.append(
                        f"Circular dependency detected: '{dep_name}' transitively depends on '{skill.name}'"
                    )

        if not errors:
            return ValidationResult(valid=True, errors=[], warnings=warnings)
        return ValidationResult(valid=False, errors=errors, warnings=warnings)

    def _has_transitive_dependency(self, from_skill: str, target: str) -> bool:
        """Check if from_skill transitively depends on target."""
        visited: Set[str] = set()
        stack = list(self._get_deps(from_skill))
        while stack:
            current = stack.pop()
            if current == target:
                return True
            if current in visited:
                continue
            visited.add(current)
            stack.extend(self._get_deps(current))
        return False

    def _get_deps(self, name: str) -> List[str]:
        skill = self.lookup(name)
        if skill is None:
            return []
        return skill.dependencies

    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, name: str) -> bool:
        return name in self._skills


@dataclass
class ValidationResult:
    """Result of a validation check."""

    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 5. Composition Engine — Combine skills into pipelines
# ---------------------------------------------------------------------------

class CompositionNode(ABC):
    """Abstract base class for composition nodes."""

    @abstractmethod
    def output_types(self) -> Dict[str, SkillType]:
        """Return the combined output types of this node."""
        ...

    @abstractmethod
    def required_skills(self) -> Set[str]:
        """Return the set of skill names required by this node."""
        ...


class SkillRef(CompositionNode):
    """Reference to a registered skill."""

    def __init__(self, skill_name: str) -> None:
        self.skill_name = skill_name
        self._skill_def: Optional[SkillDefinition] = None

    def resolve(self, registry: SkillRegistry) -> None:
        """Resolve the reference against a registry."""
        self._skill_def = registry.lookup(self.skill_name)

    @property
    def skill_def(self) -> Optional[SkillDefinition]:
        return self._skill_def

    def output_types(self) -> Dict[str, SkillType]:
        if self._skill_def:
            return dict(self._skill_def.signature.outputs)
        return {}

    def required_skills(self) -> Set[str]:
        return {self.skill_name}

    def __repr__(self) -> str:
        return f"SkillRef({self.skill_name!r})"


class Pipeline(CompositionNode):
    """Ordered sequence of skills — outputs of one feed into the next."""

    def __init__(self, steps: List[CompositionNode]) -> None:
        self.steps = list(steps)

    def output_types(self) -> Dict[str, SkillType]:
        if not self.steps:
            return {}
        return self.steps[-1].output_types()

    def required_skills(self) -> Set[str]:
        skills: Set[str] = set()
        for step in self.steps:
            skills.update(step.required_skills())
        return skills

    def __repr__(self) -> str:
        return f"Pipeline(steps={self.steps})"


class FanOut(CompositionNode):
    """Run the same skill on multiple inputs (parallel map)."""

    def __init__(self, skill: CompositionNode, inputs: List[Dict[str, SkillType]]) -> None:
        self.skill = skill
        self.inputs = list(inputs)

    def output_types(self) -> Dict[str, SkillType]:
        """FanOut produces a LIST of the skill's output types."""
        skill_outs = self.skill.output_types()
        return {name: LIST(t) for name, t in skill_outs.items()}

    def required_skills(self) -> Set[str]:
        return self.skill.required_skills()

    def __repr__(self) -> str:
        return f"FanOut(skill={self.skill}, count={len(self.inputs)})"


class FanIn(CompositionNode):
    """Merge multiple sources into a single output."""

    class MergeStrategy(Enum):
        CONCAT = auto()       # concatenate lists
        FIRST = auto()        # take first source's output
        LAST = auto()         # take last source's output
        MERGE_MAP = auto()    # merge maps by key
        ZIP = auto()          # zip parallel results

    def __init__(
        self,
        merge_strategy: "FanIn.MergeStrategy",
        sources: List[CompositionNode],
    ) -> None:
        self.merge_strategy = merge_strategy
        self.sources = list(sources)

    def output_types(self) -> Dict[str, SkillType]:
        if not self.sources:
            return {}
        if self.merge_strategy == FanIn.MergeStrategy.CONCAT:
            # Concat: unwrap LIST<T> to LIST<T>, or wrap non-list in LIST
            result: Dict[str, SkillType] = {}
            for name, t in self.sources[0].output_types().items():
                if isinstance(t, CompositeType) and t.kind == CompositeType.Kind.LIST:
                    result[name] = t
                else:
                    result[name] = LIST(t)
            return result
        elif self.merge_strategy == FanIn.MergeStrategy.FIRST:
            return dict(self.sources[0].output_types())
        elif self.merge_strategy == FanIn.MergeStrategy.LAST:
            return dict(self.sources[-1].output_types())
        elif self.merge_strategy == FanIn.MergeStrategy.MERGE_MAP:
            return dict(self.sources[0].output_types())
        elif self.merge_strategy == FanIn.MergeStrategy.ZIP:
            return dict(self.sources[0].output_types())
        return {}

    def required_skills(self) -> Set[str]:
        skills: Set[str] = set()
        for s in self.sources:
            skills.update(s.required_skills())
        return skills

    def __repr__(self) -> str:
        return f"FanIn(strategy={self.merge_strategy.name}, sources={self.sources})"


class Conditional(CompositionNode):
    """Conditional execution — choose branch based on a condition skill."""

    def __init__(
        self,
        condition_skill: CompositionNode,
        true_branch: CompositionNode,
        false_branch: CompositionNode,
    ) -> None:
        self.condition_skill = condition_skill
        self.true_branch = true_branch
        self.false_branch = false_branch

    def output_types(self) -> Dict[str, SkillType]:
        # Both branches must produce the same output types
        true_outs = self.true_branch.output_types()
        false_outs = self.false_branch.output_types()
        # Return true branch types (should match false branch)
        return true_outs

    def required_skills(self) -> Set[str]:
        skills: Set[str] = set()
        skills.update(self.condition_skill.required_skills())
        skills.update(self.true_branch.required_skills())
        skills.update(self.false_branch.required_skills())
        return skills

    def __repr__(self) -> str:
        return (
            f"Conditional(condition={self.condition_skill}, "
            f"true={self.true_branch}, false={self.false_branch})"
        )


def _resolve_refs(node: CompositionNode, registry: SkillRegistry) -> None:
    """Recursively resolve all SkillRef nodes against the registry."""
    if isinstance(node, SkillRef):
        node.resolve(registry)
    elif isinstance(node, Pipeline):
        for step in node.steps:
            _resolve_refs(step, registry)
    elif isinstance(node, FanOut):
        _resolve_refs(node.skill, registry)
    elif isinstance(node, FanIn):
        for source in node.sources:
            _resolve_refs(source, registry)
    elif isinstance(node, Conditional):
        _resolve_refs(node.condition_skill, registry)
        _resolve_refs(node.true_branch, registry)
        _resolve_refs(node.false_branch, registry)


def validate_pipeline(pipeline: CompositionNode, registry: SkillRegistry) -> ValidationResult:
    """Type-check that a pipeline's outputs match the next step's inputs."""
    errors: List[str] = []
    warnings: List[str] = []

    # Resolve all SkillRef nodes first
    _resolve_refs(pipeline, registry)

    # Check all required skills exist
    required = pipeline.required_skills()
    for skill_name in required:
        if registry.lookup(skill_name) is None:
            errors.append(f"Skill '{skill_name}' referenced in pipeline but not found in registry")

    # For Pipeline nodes, check step-to-step type compatibility
    if isinstance(pipeline, Pipeline):
        for i in range(len(pipeline.steps) - 1):
            current_outs = pipeline.steps[i].output_types()
            next_node = pipeline.steps[i + 1]
            # If next step is a SkillRef, check against its inputs
            if isinstance(next_node, SkillRef):
                next_def = registry.lookup(next_node.skill_name)
                if next_def:
                    compat_errors = TypeChecker.types_compatible(
                        next_def.signature.inputs, current_outs
                    )
                    for err in compat_errors:
                        errors.append(f"Pipeline step {i} -> {i + 1}: {err}")
                    # Check for extra outputs that aren't consumed
                    for name in current_outs:
                        if name not in next_def.signature.inputs:
                            warnings.append(
                                f"Pipeline step {i}: output '{name}' not consumed by step {i + 1}"
                            )

    # For Conditional, check both branches have matching output types
    if isinstance(pipeline, Conditional):
        true_outs = pipeline.true_branch.output_types()
        false_outs = pipeline.false_branch.output_types()
        if true_outs != false_outs:
            errors.append(
                f"Conditional branches have mismatched output types: "
                f"true={true_outs}, false={false_outs}"
            )

    if not errors:
        return ValidationResult(valid=True, errors=[], warnings=warnings)
    return ValidationResult(valid=False, errors=errors, warnings=warnings)


# ---------------------------------------------------------------------------
# 6. Dependency Resolver — Topological sort and cycle detection
# ---------------------------------------------------------------------------

class DependencyResolver:
    """Topological sort of skill dependencies with cycle detection."""

    @staticmethod
    def resolve(
        skills: List[SkillDefinition],
        registry: Optional[SkillRegistry] = None,
    ) -> List[List[SkillDefinition]]:
        """
        Topologically sort skills into layers. Each layer contains skills
        that can run in parallel (no dependencies between them within a layer).
        """
        if registry is None:
            registry = SkillRegistry()
            for s in skills:
                registry.register(s)

        skill_map: Dict[str, SkillDefinition] = {s.name: s for s in skills}
        in_degree: Dict[str, int] = {s.name: 0 for s in skills}
        dependents: Dict[str, List[str]] = defaultdict(list)

        for s in skills:
            for dep in s.dependencies:
                if dep in skill_map:
                    in_degree[s.name] += 1
                    dependents[dep].append(s.name)

        layers: List[List[SkillDefinition]] = []
        queue = [name for name, deg in in_degree.items() if deg == 0]

        while queue:
            layer = [skill_map[name] for name in queue]
            layers.append(layer)
            next_queue: List[str] = []
            for name in queue:
                for dependent in dependents[name]:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        next_queue.append(dependent)
            queue = next_queue

        return layers

    @staticmethod
    def detect_cycles(skills: List[SkillDefinition]) -> List[List[str]]:
        """
        Detect circular dependencies. Returns a list of cycles,
        where each cycle is a list of skill names forming the cycle.
        """
        skill_map: Dict[str, SkillDefinition] = {s.name: s for s in skills}
        WHITE, GRAY, BLACK = 0, 1, 2
        color: Dict[str, int] = {s.name: WHITE for s in skills}
        parent: Dict[str, str] = {}
        cycles: List[List[str]] = []

        def dfs(node: str) -> None:
            color[node] = GRAY
            skill = skill_map.get(node)
            if skill is None:
                return
            for dep in skill.dependencies:
                if dep not in skill_map:
                    continue
                if color[dep] == GRAY:
                    # Found a cycle — extract it
                    cycle = [dep]
                    current = node
                    while current != dep:
                        cycle.append(current)
                        current = parent.get(current, dep)
                    cycle.append(dep)
                    cycle.reverse()
                    # Normalize: start from the smallest element
                    min_idx = cycle.index(min(cycle[:-1]))
                    normalized = cycle[min_idx:-1] + cycle[:min_idx] + [cycle[min_idx]]
                    cycles.append(normalized)
                elif color[dep] == WHITE:
                    parent[dep] = node
                    dfs(dep)
            color[node] = BLACK

        for s in skills:
            if color[s.name] == WHITE:
                dfs(s.name)

        return cycles

    @staticmethod
    def missing_dependencies(skill: SkillDefinition, registry: SkillRegistry) -> List[str]:
        """Return list of dependency names that are missing from the registry."""
        missing: List[str] = []
        for dep_name in skill.dependencies:
            if registry.lookup(dep_name) is None:
                missing.append(dep_name)
        return missing


# ---------------------------------------------------------------------------
# 7. Parser — Simple text format for skill definitions
# ---------------------------------------------------------------------------

class ParseError(Exception):
    """Raised when parsing fails."""
    pass


class SkillParser:
    """
    Parser for the Skill Definition Language text format.

    Grammar:
        skill_def := 'skill' NAME '(' params ')' '->' '(' params ')' NEWLINE metadata*
        params := (param (',' param)?)?
        param := NAME ':' TYPE
        metadata := ('tags' ':' tag_list) | ('version' ':' STRING) | ('depends' ':' '[' name_list ']') | ('description' ':' STRING) | ('author' ':' STRING)
        tag_list := TAG (',' TAG)*
        name_list := NAME (',' NAME)*
    """

    TAG_MAP = {
        "PURE": SkillTag.PURE,
        "IO": SkillTag.IO,
        "A2A": SkillTag.A2A,
        "BLOCKING": SkillTag.BLOCKING,
    }

    TYPE_MAP = {
        "INT": INT,
        "FLOAT": FLOAT,
        "STRING": STRING,
        "BOOL": BOOL,
        "BYTES": BYTES,
        "OPCODE": OPCODE,
        "BYTECODE": BYTECODE,
        "AGENT_ID": AGENT_ID,
        "TRUST_LEVEL": TRUST_LEVEL,
        "SIGNAL_REF": SIGNAL_REF,
    }

    def __init__(self) -> None:
        self._skills: List[SkillDefinition] = []

    def parse(self, text: str) -> List[SkillDefinition]:
        """Parse a multi-line SDL text and return a list of SkillDefinitions."""
        self._skills = []
        lines = text.strip().split("\n")
        i = 0
        while i < len(lines):
            line = lines[i].rstrip()
            stripped = line.strip()

            # Skip blank lines
            if not stripped:
                i += 1
                continue

            # Parse skill definition
            if stripped.startswith("skill "):
                skill_def, i = self._parse_skill(lines, i)
                self._skills.append(skill_def)
            else:
                i += 1

        return self._skills

    def _parse_skill(
        self, lines: List[str], start: int
    ) -> Tuple[SkillDefinition, int]:
        """Parse a single skill definition starting at line `start`."""
        line = lines[start].strip()

        # Pattern: skill name(params) -> (outputs)
        pattern = r"skill\s+(\w+)\s*\((.*?)\)\s*->\s*\((.*?)\)"
        match = re.match(pattern, line)
        if not match:
            raise ParseError(f"Invalid skill definition at line {start + 1}: {line}")

        name = match.group(1)
        inputs_str = match.group(2).strip()
        outputs_str = match.group(3).strip()

        inputs = self._parse_params(inputs_str) if inputs_str else {}
        outputs = self._parse_params(outputs_str) if outputs_str else {}

        # Parse metadata from subsequent indented lines
        i = start + 1
        tags: Set[SkillTag] = set()
        version: str = "0.0.0"
        author: str = ""
        description: str = ""
        depends: List[str] = []
        implementation_ref: str = ""

        while i < len(lines):
            line = lines[i].rstrip()
            if not line or not line[0].isspace():
                break
            stripped = line.strip()

            if stripped.startswith("tags:"):
                tag_str = stripped[len("tags:"):].strip()
                for tag_name in self._split_csv(tag_str):
                    tag_name = tag_name.strip()
                    if tag_name in self.TAG_MAP:
                        tags.add(self.TAG_MAP[tag_name])
                    else:
                        raise ParseError(f"Unknown tag '{tag_name}' at line {i + 1}")
            elif stripped.startswith("version:"):
                version = self._parse_string_value(stripped[len("version:"):].strip())
            elif stripped.startswith("author:"):
                author = self._parse_string_value(stripped[len("author:"):].strip())
            elif stripped.startswith("description:"):
                description = self._parse_string_value(stripped[len("description:"):].strip())
            elif stripped.startswith("depends:"):
                dep_str = stripped[len("depends:"):].strip()
                depends = [d.strip() for d in self._parse_bracket_list(dep_str)]
            elif stripped.startswith("implementation_ref:"):
                implementation_ref = self._parse_string_value(
                    stripped[len("implementation_ref:"):].strip()
                )
            else:
                raise ParseError(f"Unknown metadata '{stripped}' at line {i + 1}")

            i += 1

        signature = SkillSignature(
            name=name,
            inputs=inputs,
            outputs=outputs,
            description=description,
            tags=frozenset(tags),
        )

        skill_def = SkillDefinition(
            signature=signature,
            dependencies=depends,
            implementation_ref=implementation_ref,
            version=version,
            author=author,
        )

        return skill_def, i

    def _parse_params(self, params_str: str) -> Dict[str, SkillType]:
        """Parse parameter list like 'text: STRING, count: INT'."""
        result: Dict[str, SkillType] = {}
        if not params_str:
            return result

        # Split by comma, but respect nested angle brackets
        parts = self._split_params(params_str)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if ":" not in part:
                raise ParseError(f"Invalid parameter (missing ':'): {part}")
            name, type_str = part.split(":", 1)
            name = name.strip()
            type_str = type_str.strip()
            result[name] = self._parse_type(type_str)

        return result

    def _split_params(self, s: str) -> List[str]:
        """Split parameters by commas, respecting nested angle brackets."""
        parts: List[str] = []
        depth = 0
        current: List[str] = []
        for ch in s:
            if ch == "," and depth == 0:
                parts.append("".join(current))
                current = []
            else:
                if ch == "<":
                    depth += 1
                elif ch == ">":
                    depth -= 1
                current.append(ch)
        if current:
            parts.append("".join(current))
        return parts

    def _parse_type(self, type_str: str) -> SkillType:
        """Parse a type string like 'LIST<STRING>' or 'MAP<STRING,INT>'."""
        type_str = type_str.strip()

        # Check for composite types with angle brackets
        bracket_match = re.match(r"^(\w+)\s*<(.+)>$", type_str)
        if bracket_match:
            type_name = bracket_match.group(1)
            inner_str = bracket_match.group(2).strip()

            if type_name == "LIST":
                inner = self._parse_type(inner_str)
                return LIST(inner)
            elif type_name == "OPTIONAL":
                inner = self._parse_type(inner_str)
                return OPTIONAL(inner)
            elif type_name == "TUPLE":
                inner_types = [self._parse_type(t.strip()) for t in self._split_params(inner_str)]
                return TUPLE(*inner_types)
            elif type_name == "MAP":
                inner_parts = self._split_params(inner_str)
                if len(inner_parts) != 2:
                    raise ParseError(f"MAP requires exactly 2 type parameters, got: {inner_str}")
                key_type = self._parse_type(inner_parts[0].strip())
                val_type = self._parse_type(inner_parts[1].strip())
                return MAP(key_type, val_type)
            else:
                raise ParseError(f"Unknown composite type: {type_name}")

        # Check for primitive/special types
        if type_str in self.TYPE_MAP:
            return self.TYPE_MAP[type_str]

        raise ParseError(f"Unknown type: {type_str}")

    def _parse_string_value(self, s: str) -> str:
        """Parse a quoted or unquoted string value."""
        s = s.strip()
        if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
            return s[1:-1]
        return s

    def _parse_bracket_list(self, s: str) -> List[str]:
        """Parse a bracketed list like '[a, b, c]'."""
        s = s.strip()
        if s.startswith("[") and s.endswith("]"):
            inner = s[1:-1].strip()
            if not inner:
                return []
            return self._split_csv(inner)
        return []

    @staticmethod
    def _split_csv(s: str) -> List[str]:
        """Split a comma-separated string."""
        return [part.strip() for part in s.split(",") if part.strip()]

    @staticmethod
    def serialize(skill_def: SkillDefinition) -> str:
        """Serialize a SkillDefinition back to SDL text format."""
        sig = skill_def.signature
        ins = ", ".join(f"{k}: {v}" for k, v in sig.inputs.items())
        outs = ", ".join(f"{k}: {v}" for k, v in sig.outputs.items())
        lines = [f"skill {sig.name}({ins}) -> ({outs})"]

        if sig.tags:
            tag_names = sorted(t.name for t in sig.tags)
            lines.append(f"  tags: {', '.join(tag_names)}")

        if skill_def.version != "0.0.0":
            lines.append(f'  version: "{skill_def.version}"')

        if skill_def.author:
            lines.append(f'  author: "{skill_def.author}"')

        if sig.description:
            lines.append(f'  description: "{sig.description}"')

        if skill_def.implementation_ref:
            lines.append(f'  implementation_ref: "{skill_def.implementation_ref}"')

        if skill_def.dependencies:
            lines.append(f"  depends: [{', '.join(skill_def.dependencies)}]")

        return "\n".join(lines)
