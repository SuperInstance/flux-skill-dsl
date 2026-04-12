"""Comprehensive tests for the Skill Definition Language (SDL)."""

from typing import Optional

import pytest
from skill_dsl.dsl import (
    # Types
    INT, FLOAT, STRING, BOOL, BYTES, OPCODE,
    BYTECODE, AGENT_ID, TRUST_LEVEL, SIGNAL_REF,
    PrimitiveType, CompositeType, SpecialType, SkillType,
    LIST, MAP, TUPLE, OPTIONAL,
    TypeChecker, TypeCheckError,
    # Signature & Definition
    SkillTag, SkillSignature, SkillDefinition,
    # Registry
    SkillRegistry, ValidationResult,
    # Composition
    CompositionNode, SkillRef, Pipeline, FanOut, FanIn, Conditional,
    validate_pipeline,
    # Dependency
    DependencyResolver,
    # Parser
    SkillParser, ParseError,
)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Primitive Types
# ═══════════════════════════════════════════════════════════════════════════

class TestPrimitiveTypes:

    def test_int_is_primitive(self):
        assert INT.is_primitive()
        assert not INT.is_composite()
        assert not INT.is_special()

    def test_float_is_primitive(self):
        assert FLOAT.is_primitive()
        assert repr(FLOAT) == "FLOAT"

    def test_string_is_primitive(self):
        assert STRING.is_primitive()
        assert repr(STRING) == "STRING"

    def test_bool_is_primitive(self):
        assert BOOL.is_primitive()

    def test_bytes_is_primitive(self):
        assert BYTES.is_primitive()

    def test_opcode_is_primitive(self):
        assert OPCODE.is_primitive()

    def test_primitive_equality(self):
        assert INT == PrimitiveType(PrimitiveType.Kind.INT)
        assert INT != FLOAT
        assert INT == INT

    def test_primitive_hash(self):
        s = {INT, FLOAT, STRING}
        assert len(s) == 3
        assert INT in s

    def test_all_primitive_types_distinct(self):
        primitives = [INT, FLOAT, STRING, BOOL, BYTES, OPCODE]
        kinds = set()
        for p in primitives:
            kinds.add(p.kind)
        assert len(kinds) == 6


# ═══════════════════════════════════════════════════════════════════════════
# 2. Special Types
# ═══════════════════════════════════════════════════════════════════════════

class TestSpecialTypes:

    def test_bytecode_is_special(self):
        assert BYTECODE.is_special()
        assert not BYTECODE.is_primitive()
        assert repr(BYTECODE) == "BYTECODE"

    def test_agent_id_is_special(self):
        assert AGENT_ID.is_special()

    def test_trust_level_is_special(self):
        assert TRUST_LEVEL.is_special()

    def test_signal_ref_is_special(self):
        assert SIGNAL_REF.is_special()

    def test_special_equality(self):
        assert AGENT_ID == SpecialType(SpecialType.Kind.AGENT_ID)
        assert AGENT_ID != BYTECODE

    def test_special_hash(self):
        s = {BYTECODE, AGENT_ID, TRUST_LEVEL, SIGNAL_REF}
        assert len(s) == 4

    def test_special_distinct_from_primitive(self):
        assert INT != BYTECODE


# ═══════════════════════════════════════════════════════════════════════════
# 3. Composite Types
# ═══════════════════════════════════════════════════════════════════════════

class TestCompositeTypes:

    def test_list_is_composite(self):
        t = LIST(STRING)
        assert t.is_composite()
        assert not t.is_primitive()
        assert not t.is_special()

    def test_list_repr(self):
        assert repr(LIST(STRING)) == "LIST<STRING>"
        assert repr(LIST(LIST(INT))) == "LIST<LIST<INT>>"

    def test_list_equality(self):
        assert LIST(STRING) == LIST(STRING)
        assert LIST(STRING) != LIST(INT)

    def test_map_repr(self):
        assert repr(MAP(STRING, INT)) == "MAP<STRING,INT>"
        assert repr(MAP(STRING, LIST(FLOAT))) == "MAP<STRING,LIST<FLOAT>>"

    def test_map_equality(self):
        assert MAP(STRING, INT) == MAP(STRING, INT)
        assert MAP(STRING, INT) != MAP(INT, STRING)

    def test_tuple_repr(self):
        assert repr(TUPLE(STRING, INT)) == "TUPLE<STRING,INT>"
        assert repr(TUPLE(BOOL, FLOAT, STRING)) == "TUPLE<BOOL,FLOAT,STRING>"

    def test_tuple_single_element(self):
        t = TUPLE(INT)
        assert t.is_composite()
        assert repr(t) == "TUPLE<INT>"

    def test_tuple_equality(self):
        assert TUPLE(INT, STRING) == TUPLE(INT, STRING)
        assert TUPLE(INT, STRING) != TUPLE(STRING, INT)

    def test_optional_repr(self):
        assert repr(OPTIONAL(STRING)) == "OPTIONAL<STRING>"
        assert repr(OPTIONAL(LIST(INT))) == "OPTIONAL<LIST<INT>>"

    def test_optional_equality(self):
        assert OPTIONAL(INT) == OPTIONAL(INT)
        assert OPTIONAL(INT) != OPTIONAL(FLOAT)

    def test_nested_composite(self):
        t = LIST(MAP(STRING, OPTIONAL(FLOAT)))
        assert repr(t) == "LIST<MAP<STRING,OPTIONAL<FLOAT>>>"
        assert t.is_composite()

    def test_composite_hash(self):
        s = {LIST(INT), LIST(STRING), MAP(STRING, INT)}
        assert len(s) == 3


# ═══════════════════════════════════════════════════════════════════════════
# 4. TypeChecker
# ═══════════════════════════════════════════════════════════════════════════

class TestTypeChecker:

    def test_primitive_always_well_formed(self):
        assert TypeChecker.is_well_formed(INT)
        assert TypeChecker.is_well_formed(STRING)
        assert TypeChecker.is_well_formed(OPCODE)

    def test_special_always_well_formed(self):
        assert TypeChecker.is_well_formed(BYTECODE)
        assert TypeChecker.is_well_formed(AGENT_ID)

    def test_list_well_formed(self):
        assert TypeChecker.is_well_formed(LIST(STRING))
        assert TypeChecker.is_well_formed(LIST(LIST(INT)))

    def test_list_empty_params_not_well_formed(self):
        bad = CompositeType(CompositeType.Kind.LIST, [])
        assert not TypeChecker.is_well_formed(bad)
        errors = TypeChecker.check(bad)
        assert len(errors) == 1
        assert "exactly 1" in errors[0]

    def test_map_well_formed(self):
        assert TypeChecker.is_well_formed(MAP(STRING, INT))

    def test_map_wrong_params(self):
        bad = CompositeType(CompositeType.Kind.MAP, [STRING])
        assert not TypeChecker.is_well_formed(bad)

    def test_tuple_well_formed(self):
        assert TypeChecker.is_well_formed(TUPLE(INT, STRING))

    def test_tuple_empty_params_not_well_formed(self):
        bad = CompositeType(CompositeType.Kind.TUPLE, [])
        assert not TypeChecker.is_well_formed(bad)

    def test_optional_well_formed(self):
        assert TypeChecker.is_well_formed(OPTIONAL(FLOAT))

    def test_optional_wrong_params(self):
        bad = CompositeType(CompositeType.Kind.OPTIONAL, [])
        assert not TypeChecker.is_well_formed(bad)

    def test_nested_type_check(self):
        # LIST<MAP<STRING, TUPLE<INT, FLOAT>>>
        t = LIST(MAP(STRING, TUPLE(INT, FLOAT)))
        assert TypeChecker.is_well_formed(t)

    def test_is_assignable_exact_match(self):
        assert TypeChecker.is_assignable(STRING, STRING)
        assert not TypeChecker.is_assignable(STRING, INT)

    def test_is_assignable_composite(self):
        assert TypeChecker.is_assignable(LIST(STRING), LIST(STRING))
        assert not TypeChecker.is_assignable(LIST(STRING), LIST(INT))

    def test_types_compatible_exact_match(self):
        errors = TypeChecker.types_compatible(
            {"text": STRING, "count": INT},
            {"text": STRING, "count": INT},
        )
        assert errors == []

    def test_types_compatible_missing_field(self):
        errors = TypeChecker.types_compatible(
            {"text": STRING, "count": INT},
            {"text": STRING},
        )
        assert len(errors) == 1
        assert "Missing" in errors[0]

    def test_types_compatible_type_mismatch(self):
        errors = TypeChecker.types_compatible(
            {"count": INT},
            {"count": STRING},
        )
        assert len(errors) == 1
        assert "Type mismatch" in errors[0]

    def test_types_compatible_extra_field(self):
        errors = TypeChecker.types_compatible(
            {"text": STRING},
            {"text": STRING, "extra": INT},
        )
        assert len(errors) == 1
        assert "Unexpected" in errors[0]


# ═══════════════════════════════════════════════════════════════════════════
# 5. SkillSignature
# ═══════════════════════════════════════════════════════════════════════════

class TestSkillSignature:

    def test_basic_signature(self):
        sig = SkillSignature(
            name="echo",
            inputs={"text": STRING},
            outputs={"result": STRING},
        )
        assert sig.name == "echo"
        assert sig.input_names() == ["text"]
        assert sig.output_names() == ["result"]

    def test_empty_signature(self):
        sig = SkillSignature(name="noop")
        assert sig.inputs == {}
        assert sig.outputs == {}
        assert sig.input_names() == []
        assert sig.output_names() == []

    def test_signature_with_tags(self):
        sig = SkillSignature(
            name="read_file",
            inputs={"path": STRING},
            outputs={"content": STRING},
            tags=frozenset({SkillTag.IO, SkillTag.BLOCKING}),
        )
        assert sig.has_tag(SkillTag.IO)
        assert sig.has_tag(SkillTag.BLOCKING)
        assert not sig.has_tag(SkillTag.PURE)

    def test_signature_frozen(self):
        sig = SkillSignature(name="test", inputs={"x": INT})
        with pytest.raises(AttributeError):
            sig.name = "other"  # type: ignore

    def test_invalid_name_raises(self):
        with pytest.raises(ValueError):
            SkillSignature(name="")

    def test_signature_with_description(self):
        sig = SkillSignature(
            name="add",
            inputs={"a": INT, "b": INT},
            outputs={"sum": INT},
            description="Add two integers",
        )
        assert sig.description == "Add two integers"

    def test_signature_repr(self):
        sig = SkillSignature(
            name="echo",
            inputs={"text": STRING},
            outputs={"result": STRING},
        )
        r = repr(sig)
        assert "echo" in r
        assert "STRING" in r

    def test_all_tags(self):
        tags = [SkillTag.PURE, SkillTag.IO, SkillTag.A2A, SkillTag.BLOCKING]
        sig = SkillSignature(
            name="complex",
            tags=frozenset(tags),
        )
        for tag in tags:
            assert sig.has_tag(tag)


# ═══════════════════════════════════════════════════════════════════════════
# 6. SkillDefinition
# ═══════════════════════════════════════════════════════════════════════════

class TestSkillDefinition:

    def _make_sig(self, name: str = "test", **kwargs) -> SkillSignature:
        return SkillSignature(name=name, **kwargs)

    def test_basic_definition(self):
        sig = self._make_sig("add", inputs={"a": INT, "b": INT}, outputs={"sum": INT})
        defn = SkillDefinition(signature=sig, version="1.0.0", author="quill")
        assert defn.name == "add"
        assert defn.version == "1.0.0"
        assert defn.author == "quill"
        assert defn.dependencies == []
        assert defn.implementation_ref == ""

    def test_definition_with_deps(self):
        sig = self._make_sig("analyze", inputs={"text": STRING}, outputs={"score": FLOAT})
        defn = SkillDefinition(
            signature=sig,
            dependencies=["tokenize", "classify"],
            implementation_ref="modules/analyze.bc",
        )
        assert defn.dependencies == ["tokenize", "classify"]
        assert defn.implementation_ref == "modules/analyze.bc"

    def test_definition_default_version(self):
        sig = self._make_sig("basic")
        defn = SkillDefinition(signature=sig)
        assert defn.version == "0.0.0"

    def test_definition_frozen(self):
        sig = self._make_sig("frozen")
        defn = SkillDefinition(signature=sig)
        with pytest.raises(AttributeError):
            defn.version = "2.0"  # type: ignore

    def test_definition_repr(self):
        sig = self._make_sig("test")
        defn = SkillDefinition(signature=sig, version="1.0")
        r = repr(defn)
        assert "test" in r
        assert "1.0" in r


# ═══════════════════════════════════════════════════════════════════════════
# 7. SkillRegistry
# ═══════════════════════════════════════════════════════════════════════════

class TestSkillRegistry:

    def _make_skill(self, name: str, **kwargs) -> SkillDefinition:
        sig = SkillSignature(name=name, **kwargs)
        return SkillDefinition(signature=sig)

    def test_register_and_lookup(self):
        reg = SkillRegistry()
        skill = self._make_skill("echo", inputs={"x": STRING}, outputs={"y": STRING})
        reg.register(skill)
        assert reg.lookup("echo") == skill
        assert reg.lookup("nonexistent") is None

    def test_register_overwrites(self):
        reg = SkillRegistry()
        s1 = self._make_skill("echo", inputs={"x": STRING}, outputs={"y": STRING})
        s2 = self._make_skill("echo", inputs={"x": INT}, outputs={"y": INT})
        reg.register(s1)
        reg.register(s2)
        assert reg.lookup("echo") == s2
        assert len(reg) == 1

    def test_unregister(self):
        reg = SkillRegistry()
        skill = self._make_skill("temp")
        reg.register(skill)
        reg.unregister("temp")
        assert reg.lookup("temp") is None
        assert len(reg) == 0

    def test_find_by_tag(self):
        reg = SkillRegistry()
        reg.register(SkillDefinition(
            signature=SkillSignature("read", tags=frozenset({SkillTag.IO}))
        ))
        reg.register(SkillDefinition(
            signature=SkillSignature("compute", tags=frozenset({SkillTag.PURE}))
        ))
        io_skills = reg.find_by_tag(SkillTag.IO)
        assert len(io_skills) == 1
        assert io_skills[0].name == "read"

    def test_find_by_tag_no_results(self):
        reg = SkillRegistry()
        reg.register(SkillDefinition(
            signature=SkillSignature("read", tags=frozenset({SkillTag.IO}))
        ))
        assert reg.find_by_tag(SkillTag.PURE) == []

    def test_find_by_output_type(self):
        reg = SkillRegistry()
        reg.register(SkillDefinition(
            signature=SkillSignature("to_str", inputs={"x": INT}, outputs={"result": STRING})
        ))
        reg.register(SkillDefinition(
            signature=SkillSignature("to_int", inputs={"x": STRING}, outputs={"result": INT})
        ))
        str_producers = reg.find_by_output_type(STRING)
        assert len(str_producers) == 1
        assert str_producers[0].name == "to_str"

    def test_find_by_output_type_multiple(self):
        reg = SkillRegistry()
        reg.register(SkillDefinition(
            signature=SkillSignature("f1", outputs={"a": STRING})
        ))
        reg.register(SkillDefinition(
            signature=SkillSignature("f2", outputs={"b": STRING})
        ))
        reg.register(SkillDefinition(
            signature=SkillSignature("f3", outputs={"c": INT})
        ))
        assert len(reg.find_by_output_type(STRING)) == 2

    def test_all_skills(self):
        reg = SkillRegistry()
        reg.register(self._make_skill("a"))
        reg.register(self._make_skill("b"))
        assert len(reg.all_skills()) == 2

    def test_skill_names(self):
        reg = SkillRegistry()
        reg.register(self._make_skill("alpha"))
        reg.register(self._make_skill("beta"))
        names = reg.skill_names()
        assert "alpha" in names
        assert "beta" in names

    def test_contains(self):
        reg = SkillRegistry()
        reg.register(self._make_skill("exists"))
        assert "exists" in reg
        assert "missing" not in reg

    def test_validate_dependencies_all_found(self):
        reg = SkillRegistry()
        reg.register(self._make_skill("dep_a"))
        reg.register(self._make_skill("dep_b"))
        skill = SkillDefinition(
            signature=SkillSignature("main"),
            dependencies=["dep_a", "dep_b"],
        )
        result = reg.validate_dependencies(skill)
        assert result.valid

    def test_validate_dependencies_missing(self):
        reg = SkillRegistry()
        skill = SkillDefinition(
            signature=SkillSignature("main"),
            dependencies=["missing_dep"],
        )
        result = reg.validate_dependencies(skill)
        assert not result.valid
        assert len(result.errors) == 1
        assert "missing_dep" in result.errors[0]

    def test_validate_dependencies_with_cycle(self):
        reg = SkillRegistry()
        reg.register(SkillDefinition(
            signature=SkillSignature("a"),
            dependencies=["b"],
        ))
        skill = SkillDefinition(
            signature=SkillSignature("b"),
            dependencies=["a"],
        )
        result = reg.validate_dependencies(skill)
        assert not result.valid
        assert any("Circular" in e for e in result.errors)

    def test_validate_empty_dependencies(self):
        reg = SkillRegistry()
        skill = self._make_skill("solo")
        result = reg.validate_dependencies(skill)
        assert result.valid


# ═══════════════════════════════════════════════════════════════════════════
# 8. Composition Engine
# ═══════════════════════════════════════════════════════════════════════════

class TestSkillRef:
    def test_basic_ref(self):
        ref = SkillRef("echo")
        assert ref.skill_name == "echo"
        assert ref.required_skills() == {"echo"}

    def test_resolve(self):
        reg = SkillRegistry()
        skill = SkillDefinition(
            signature=SkillSignature("echo", inputs={"x": STRING}, outputs={"y": STRING})
        )
        reg.register(skill)
        ref = SkillRef("echo")
        ref.resolve(reg)
        assert ref.skill_def == skill
        assert ref.output_types() == {"y": STRING}

    def test_unresolved_output_types(self):
        ref = SkillRef("unknown")
        assert ref.output_types() == {}


class TestPipeline:
    def test_empty_pipeline(self):
        p = Pipeline([])
        assert p.output_types() == {}
        assert p.required_skills() == set()

    def test_single_step(self):
        step = SkillRef("echo")
        p = Pipeline([step])
        assert p.required_skills() == {"echo"}

    def test_multi_step(self):
        p = Pipeline([SkillRef("a"), SkillRef("b"), SkillRef("c")])
        assert p.required_skills() == {"a", "b", "c"}

    def test_nested_pipeline(self):
        inner = Pipeline([SkillRef("step1"), SkillRef("step2")])
        outer = Pipeline([SkillRef("pre"), inner, SkillRef("post")])
        assert outer.required_skills() == {"pre", "step1", "step2", "post"}


class TestFanOut:
    def test_basic_fan_out(self):
        skill = SkillRef("classify")
        fan = FanOut(skill, [{"text": STRING}, {"text": STRING}])
        assert fan.required_skills() == {"classify"}

    def test_fan_out_output_types(self):
        reg = SkillRegistry()
        reg.register(SkillDefinition(
            signature=SkillSignature("classify", outputs={"label": STRING, "score": FLOAT})
        ))
        ref = SkillRef("classify")
        ref.resolve(reg)
        fan = FanOut(ref, [{"text": STRING}])
        outs = fan.output_types()
        assert outs == {"label": LIST(STRING), "score": LIST(FLOAT)}


class TestFanIn:
    def test_concat_strategy(self):
        reg = SkillRegistry()
        reg.register(SkillDefinition(
            signature=SkillSignature("fetch", outputs={"items": LIST(INT)})
        ))
        ref1 = SkillRef("fetch")
        ref1.resolve(reg)
        ref2 = SkillRef("fetch")
        ref2.resolve(reg)
        fan = FanIn(FanIn.MergeStrategy.CONCAT, [ref1, ref2])
        outs = fan.output_types()
        assert outs == {"items": LIST(INT)}

    def test_first_strategy(self):
        reg = SkillRegistry()
        reg.register(SkillDefinition(
            signature=SkillSignature("a", outputs={"x": STRING})
        ))
        reg.register(SkillDefinition(
            signature=SkillSignature("b", outputs={"y": INT})
        ))
        ref_a = SkillRef("a"); ref_a.resolve(reg)
        ref_b = SkillRef("b"); ref_b.resolve(reg)
        fan = FanIn(FanIn.MergeStrategy.FIRST, [ref_a, ref_b])
        assert fan.output_types() == {"x": STRING}

    def test_last_strategy(self):
        reg = SkillRegistry()
        reg.register(SkillDefinition(
            signature=SkillSignature("a", outputs={"x": STRING})
        ))
        reg.register(SkillDefinition(
            signature=SkillSignature("b", outputs={"y": INT})
        ))
        ref_a = SkillRef("a"); ref_a.resolve(reg)
        ref_b = SkillRef("b"); ref_b.resolve(reg)
        fan = FanIn(FanIn.MergeStrategy.LAST, [ref_a, ref_b])
        assert fan.output_types() == {"y": INT}

    def test_empty_sources(self):
        fan = FanIn(FanIn.MergeStrategy.FIRST, [])
        assert fan.output_types() == {}

    def test_fan_in_required_skills(self):
        fan = FanIn(FanIn.MergeStrategy.FIRST, [SkillRef("a"), SkillRef("b")])
        assert fan.required_skills() == {"a", "b"}


class TestConditional:
    def test_basic_conditional(self):
        cond = Conditional(SkillRef("check"), SkillRef("on_true"), SkillRef("on_false"))
        assert cond.required_skills() == {"check", "on_true", "on_false"}

    def test_conditional_output_types(self):
        reg = SkillRegistry()
        reg.register(SkillDefinition(
            signature=SkillSignature("on_true", outputs={"result": STRING})
        ))
        reg.register(SkillDefinition(
            signature=SkillSignature("on_false", outputs={"result": STRING})
        ))
        ref_t = SkillRef("on_true"); ref_t.resolve(reg)
        ref_f = SkillRef("on_false"); ref_f.resolve(reg)
        cond = Conditional(SkillRef("check"), ref_t, ref_f)
        assert cond.output_types() == {"result": STRING}


class TestValidatePipeline:
    def _make_reg(self) -> SkillRegistry:
        reg = SkillRegistry()
        reg.register(SkillDefinition(
            signature=SkillSignature("tokenize", inputs={"text": STRING}, outputs={"tokens": LIST(STRING)})
        ))
        reg.register(SkillDefinition(
            signature=SkillSignature("count", inputs={"tokens": LIST(STRING)}, outputs={"n": INT})
        ))
        reg.register(SkillDefinition(
            signature=SkillSignature("classify", outputs={"label": STRING})
        ))
        return reg

    def test_valid_pipeline(self):
        reg = self._make_reg()
        p = Pipeline([SkillRef("tokenize"), SkillRef("count")])
        result = validate_pipeline(p, reg)
        assert result.valid

    def test_missing_skill(self):
        reg = SkillRegistry()
        p = Pipeline([SkillRef("nonexistent")])
        result = validate_pipeline(p, reg)
        assert not result.valid
        assert any("nonexistent" in e for e in result.errors)

    def test_type_mismatch_in_pipeline(self):
        reg = self._make_reg()
        # classify outputs STRING, count expects LIST<STRING> — mismatch
        p = Pipeline([SkillRef("classify"), SkillRef("count")])
        result = validate_pipeline(p, reg)
        assert not result.valid
        assert any("Type mismatch" in e or "Missing" in e for e in result.errors)

    def test_conditional_branch_mismatch(self):
        reg = SkillRegistry()
        reg.register(SkillDefinition(
            signature=SkillSignature("branch_a", outputs={"result": STRING})
        ))
        reg.register(SkillDefinition(
            signature=SkillSignature("branch_b", outputs={"result": INT})
        ))
        cond = Conditional(
            SkillRef("check"),
            SkillRef("branch_a"),
            SkillRef("branch_b"),
        )
        result = validate_pipeline(cond, reg)
        assert not result.valid
        assert any("mismatched" in e.lower() for e in result.errors)

    def test_valid_conditional(self):
        reg = SkillRegistry()
        reg.register(SkillDefinition(
            signature=SkillSignature("branch_a", outputs={"result": STRING})
        ))
        reg.register(SkillDefinition(
            signature=SkillSignature("branch_b", outputs={"result": STRING})
        ))
        cond = Conditional(
            SkillRef("check"),
            SkillRef("branch_a"),
            SkillRef("branch_b"),
        )
        result = validate_pipeline(cond, reg)
        # Should have error for missing "check" but no branch mismatch
        mismatch_errors = [e for e in result.errors if "mismatched" in e.lower()]
        assert len(mismatch_errors) == 0


# ═══════════════════════════════════════════════════════════════════════════
# 9. Dependency Resolver
# ═══════════════════════════════════════════════════════════════════════════

class TestDependencyResolver:

    def _make_skill(self, name: str, deps: Optional[list] = None) -> SkillDefinition:
        return SkillDefinition(
            signature=SkillSignature(name),
            dependencies=deps or [],
        )

    def test_resolve_no_deps(self):
        skills = [self._make_skill("a"), self._make_skill("b")]
        layers = DependencyResolver.resolve(skills)
        assert len(layers) == 1
        assert len(layers[0]) == 2

    def test_resolve_linear_chain(self):
        skills = [
            self._make_skill("c", ["b"]),
            self._make_skill("b", ["a"]),
            self._make_skill("a"),
        ]
        layers = DependencyResolver.resolve(skills)
        assert len(layers) == 3
        assert layers[0][0].name == "a"
        assert layers[1][0].name == "b"
        assert layers[2][0].name == "c"

    def test_resolve_parallel_layers(self):
        skills = [
            self._make_skill("result", ["a", "b"]),
            self._make_skill("a"),
            self._make_skill("b"),
        ]
        layers = DependencyResolver.resolve(skills)
        assert len(layers) == 2
        assert len(layers[0]) == 2  # a and b in parallel
        assert layers[1][0].name == "result"

    def test_resolve_complex_dag(self):
        skills = [
            self._make_skill("final", ["mid1", "mid2"]),
            self._make_skill("mid1", ["base"]),
            self._make_skill("mid2", ["base"]),
            self._make_skill("base"),
        ]
        layers = DependencyResolver.resolve(skills)
        assert len(layers) == 3
        assert {s.name for s in layers[0]} == {"base"}
        assert {s.name for s in layers[1]} == {"mid1", "mid2"}
        assert {s.name for s in layers[2]} == {"final"}

    def test_detect_no_cycles(self):
        skills = [
            self._make_skill("a", ["b"]),
            self._make_skill("b"),
        ]
        cycles = DependencyResolver.detect_cycles(skills)
        assert cycles == []

    def test_detect_simple_cycle(self):
        skills = [
            self._make_skill("a", ["b"]),
            self._make_skill("b", ["a"]),
        ]
        cycles = DependencyResolver.detect_cycles(skills)
        assert len(cycles) > 0

    def test_detect_three_node_cycle(self):
        skills = [
            self._make_skill("a", ["c"]),
            self._make_skill("b", ["a"]),
            self._make_skill("c", ["b"]),
        ]
        cycles = DependencyResolver.detect_cycles(skills)
        assert len(cycles) > 0

    def test_missing_dependencies(self):
        reg = SkillRegistry()
        reg.register(self._make_skill("existing"))
        skill = SkillDefinition(
            signature=SkillSignature("test"),
            dependencies=["existing", "missing1", "missing2"],
        )
        missing = DependencyResolver.missing_dependencies(skill, reg)
        assert set(missing) == {"missing1", "missing2"}

    def test_no_missing_dependencies(self):
        reg = SkillRegistry()
        reg.register(self._make_skill("dep_a"))
        reg.register(self._make_skill("dep_b"))
        skill = SkillDefinition(
            signature=SkillSignature("main"),
            dependencies=["dep_a", "dep_b"],
        )
        assert DependencyResolver.missing_dependencies(skill, reg) == []

    def test_resolve_with_external_dep_ignored(self):
        """Deps not in the skill list should be ignored by resolve."""
        skills = [
            self._make_skill("a", ["external"]),
        ]
        layers = DependencyResolver.resolve(skills)
        assert len(layers) == 1
        assert layers[0][0].name == "a"


# ═══════════════════════════════════════════════════════════════════════════
# 10. Parser
# ═══════════════════════════════════════════════════════════════════════════

class TestSkillParser:

    def test_parse_simple_skill(self):
        text = 'skill echo(text: STRING) -> (result: STRING)'
        parser = SkillParser()
        skills = parser.parse(text)
        assert len(skills) == 1
        assert skills[0].name == "echo"
        assert skills[0].signature.inputs == {"text": STRING}
        assert skills[0].signature.outputs == {"result": STRING}

    def test_parse_skill_with_tags(self):
        text = """skill read_file(path: STRING) -> (content: STRING)
  tags: IO, BLOCKING"""
        parser = SkillParser()
        skills = parser.parse(text)
        assert SkillTag.IO in skills[0].signature.tags
        assert SkillTag.BLOCKING in skills[0].signature.tags

    def test_parse_skill_with_version(self):
        text = """skill add(a: INT, b: INT) -> (sum: INT)
  version: "1.2.0\""""
        parser = SkillParser()
        skills = parser.parse(text)
        assert skills[0].version == "1.2.0"

    def test_parse_skill_with_depends(self):
        text = """skill analyze(text: STRING) -> (score: FLOAT)
  depends: [tokenize, classify]
  version: "1.0.0\""""
        parser = SkillParser()
        skills = parser.parse(text)
        assert skills[0].dependencies == ["tokenize", "classify"]

    def test_parse_skill_with_author(self):
        text = """skill greet(name: STRING) -> (msg: STRING)
  author: "quill\""""
        parser = SkillParser()
        skills = parser.parse(text)
        assert skills[0].author == "quill"

    def test_parse_skill_with_description(self):
        text = """skill greet(name: STRING) -> (msg: STRING)
  description: "Greets the user by name\""""
        parser = SkillParser()
        skills = parser.parse(text)
        assert skills[0].signature.description == "Greets the user by name"

    def test_parse_composite_types(self):
        text = """skill process(items: LIST<STRING>) -> (counts: MAP<STRING,INT>)"""
        parser = SkillParser()
        skills = parser.parse(text)
        assert skills[0].signature.inputs["items"] == LIST(STRING)
        assert skills[0].signature.outputs["counts"] == MAP(STRING, INT)

    def test_parse_nested_composite_types(self):
        text = """skill nest(data: LIST<MAP<STRING,FLOAT>>) -> (result: OPTIONAL<STRING>)"""
        parser = SkillParser()
        skills = parser.parse(text)
        assert skills[0].signature.inputs["data"] == LIST(MAP(STRING, FLOAT))
        assert skills[0].signature.outputs["result"] == OPTIONAL(STRING)

    def test_parse_tuple_type(self):
        text = """skill make_tuple(x: INT, y: STRING) -> (pair: TUPLE<INT,STRING>)"""
        parser = SkillParser()
        skills = parser.parse(text)
        assert skills[0].signature.outputs["pair"] == TUPLE(INT, STRING)

    def test_parse_special_types(self):
        text = """skill exec(code: BYTECODE, agent: AGENT_ID) -> (result: SIGNAL_REF)"""
        parser = SkillParser()
        skills = parser.parse(text)
        assert skills[0].signature.inputs["code"] == BYTECODE
        assert skills[0].signature.inputs["agent"] == AGENT_ID
        assert skills[0].signature.outputs["result"] == SIGNAL_REF

    def test_parse_multiple_skills(self):
        text = """skill tokenize(text: STRING) -> (tokens: LIST<STRING>)
  tags: PURE

skill classify(tokens: LIST<STRING>) -> (label: STRING)
  depends: [tokenize]"""
        parser = SkillParser()
        skills = parser.parse(text)
        assert len(skills) == 2
        assert skills[0].name == "tokenize"
        assert skills[1].name == "classify"
        assert skills[1].dependencies == ["tokenize"]

    def test_parse_empty_inputs_outputs(self):
        text = "skill noop() -> ()"
        parser = SkillParser()
        skills = parser.parse(text)
        assert skills[0].signature.inputs == {}
        assert skills[0].signature.outputs == {}

    def test_parse_all_tags(self):
        text = """skill complex(x: STRING) -> (y: STRING)
  tags: PURE, IO, A2A, BLOCKING"""
        parser = SkillParser()
        skills = parser.parse(text)
        assert len(skills[0].signature.tags) == 4

    def test_parse_invalid_syntax_raises(self):
        text = "skill bad_syntax_no_arrow(x: INT)"
        parser = SkillParser()
        with pytest.raises(ParseError):
            parser.parse(text)

    def test_parse_unknown_tag_raises(self):
        text = """skill test(x: INT) -> (y: INT)
  tags: UNKNOWN"""
        parser = SkillParser()
        with pytest.raises(ParseError, match="Unknown tag"):
            parser.parse(text)

    def test_parse_unknown_type_raises(self):
        text = "skill test(x: NOTATYPE) -> (y: INT)"
        parser = SkillParser()
        with pytest.raises(ParseError, match="Unknown type"):
            parser.parse(text)

    def test_parse_with_implementation_ref(self):
        text = """skill run(op: OPCODE) -> (out: BYTES)
  implementation_ref: "modules/run_v2.bc\""""
        parser = SkillParser()
        skills = parser.parse(text)
        assert skills[0].implementation_ref == "modules/run_v2.bc"

    def test_serialize_roundtrip(self):
        original = SkillDefinition(
            signature=SkillSignature(
                name="analyze_sentiment",
                inputs={"text": STRING},
                outputs={"sentiment": FLOAT, "confidence": FLOAT},
                description="Analyze text sentiment",
                tags=frozenset({SkillTag.PURE}),
            ),
            dependencies=["tokenize", "classify"],
            implementation_ref="modules/sentiment.bc",
            version="1.0",
            author="quill",
        )
        text = SkillParser.serialize(original)
        parser = SkillParser()
        parsed = parser.parse(text)
        assert len(parsed) == 1
        assert parsed[0].name == "analyze_sentiment"
        assert parsed[0].signature.inputs == {"text": STRING}
        assert parsed[0].signature.outputs == {"sentiment": FLOAT, "confidence": FLOAT}
        assert SkillTag.PURE in parsed[0].signature.tags
        assert parsed[0].dependencies == ["tokenize", "classify"]
        assert parsed[0].version == "1.0"
        assert parsed[0].author == "quill"

    def test_parse_trust_level_type(self):
        text = "skill check(trust: TRUST_LEVEL) -> (ok: BOOL)"
        parser = SkillParser()
        skills = parser.parse(text)
        assert skills[0].signature.inputs["trust"] == TRUST_LEVEL

    def test_parse_opcode_type(self):
        text = "skill decode(op: OPCODE) -> (value: INT)"
        parser = SkillParser()
        skills = parser.parse(text)
        assert skills[0].signature.inputs["op"] == OPCODE

    def test_serialize_minimal(self):
        skill = SkillDefinition(
            signature=SkillSignature(name="minimal"),
        )
        text = SkillParser.serialize(skill)
        assert "skill minimal" in text
        assert "()" in text

    def test_parse_blank_lines_between_skills(self):
        text = """skill a(x: INT) -> (y: INT)

skill b(s: STRING) -> (t: STRING)
"""
        parser = SkillParser()
        skills = parser.parse(text)
        assert len(skills) == 2

    def test_serialize_then_parse_complex(self):
        """Full roundtrip with complex types."""
        original = SkillDefinition(
            signature=SkillSignature(
                name="batch_process",
                inputs={"items": LIST(MAP(STRING, OPTIONAL(FLOAT)))},
                outputs={"results": LIST(TUPLE(STRING, BOOL))},
                tags=frozenset({SkillTag.IO, SkillTag.BLOCKING}),
            ),
            dependencies=["validate", "transform"],
            version="2.1.0",
            author="quill",
        )
        text = SkillParser.serialize(original)
        parsed = SkillParser().parse(text)
        assert len(parsed) == 1
        p = parsed[0]
        assert p.name == "batch_process"
        assert p.signature.inputs["items"] == LIST(MAP(STRING, OPTIONAL(FLOAT)))
        assert p.signature.outputs["results"] == LIST(TUPLE(STRING, BOOL))
        assert p.dependencies == ["validate", "transform"]
        assert p.version == "2.1.0"
