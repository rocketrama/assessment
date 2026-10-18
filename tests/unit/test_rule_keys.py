"""Rule keys: the same rule stated differently lands together; incompatible facts never merge."""

import numpy as np

from app.ingestion.extract_facts import Fact
from app.ingestion.rule_keys import assign_rule_keys, normalize_attribute, normalize_conditions

# Hand-made embedding space: near-synonyms get near-identical vectors.
_VECTORS = {
    "sick leave: days per year": [1.0, 0.0, 0.0],
    "paid sick leave: annual allowance": [0.99, 0.12, 0.0],
    "annual leave accrual: days per month": [0.0, 1.0, 0.0],
    "carry forward: deadline to use": [0.0, 0.0, 1.0],
}


def fake_embed(texts):
    vectors = np.array([_VECTORS[t] for t in texts], dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def F(fact_id, topic, attribute, value, unit="working days", conditions=None, doc="D"):
    return Fact(fact_id, fact_id.split("#")[0], doc, topic, attribute, value, unit, conditions, "q")


def keys(facts):
    return {f.fact_id: f.rule_key for f in assign_rule_keys(facts, fake_embed, threshold=0.15)}


def test_same_rule_with_different_values_and_wording_shares_a_key():
    k = keys(
        [
            F("V1::005#f00", "sick leave", "days per year", "10", doc="V1"),
            F("V2::007#f00", "paid sick leave", "annual allowance", "12", unit="days", doc="V2"),
            F("FAQ::001#f00", "sick leave", "days per year", "10", unit="Working Days", doc="FAQ"),
        ]
    )
    assert len(set(k.values())) == 1
    assert k["V1::005#f00"] == "sick-leave-days-per-year"


def test_different_tiers_never_merge_even_when_they_embed_identically():
    """Tiers embed ~0.09 apart in reality; here identically. Their numbers differ."""
    _VECTORS["annual leave accrual: days per month (less than 2 years)"] = [0.0, 1.0, 0.1]
    _VECTORS["annual leave accrual: days per month (2 years to less than 5 years)"] = [0.0, 1.0, 0.1]
    k = keys(
        [
            F("A::001#f00", "annual leave accrual", "days per month", "1.5", "days per month", "less than 2 years", doc="A"),
            F("B::001#f00", "annual leave accrual", "days per month", "1.75", "days per month", "2 years to less than 5 years", doc="B"),
        ]
    )
    assert len(set(k.values())) == 2


def test_tier_in_topic_meets_tier_in_conditions():
    """Eval finding: the schedule wrote the tier as a condition, v3 into the topic."""
    _VECTORS["annual leave: days earned per month (less than 2 years)"] = [0.0, 1.0, 0.3]
    _VECTORS["annual leave accrual: less than 2 years of service"] = [0.0, 1.0, 0.32]
    k = keys(
        [
            F("TBL::001#f00", "annual leave", "days earned per month", "1.5", None, "less than 2 years", doc="TBL"),
            F("V3::001#f00", "annual leave accrual", "less than 2 years of service", "1.5", None, None, doc="V3"),
        ]
    )
    assert k["TBL::001#f00"] == k["V3::001#f00"]


def test_unrelated_rules_stay_apart():
    k = keys(
        [
            F("A::000#f00", "sick leave", "days per year", "12"),
            F("A::000#f01", "carry forward", "deadline to use", "31 March", unit=None),
            F("A::000#f02", "annual leave accrual", "days per month", "1.5", unit=None),
        ]
    )
    assert len(set(k.values())) == 3


def test_deterministic_regardless_of_input_order():
    facts = [
        F("B::000#f00", "paid sick leave", "annual allowance", "12"),
        F("A::000#f00", "sick leave", "days per year", "10"),
        F("C::000#f00", "carry forward", "deadline to use", "31 March"),
    ]
    assert keys(facts) == keys(list(reversed(facts)))


def test_units_do_not_split_a_rule():
    """The LLM names units inconsistently; the same rule must still meet."""
    k = keys(
        [
            F("V2::007#f00", "sick leave", "days per year", "12", unit="working days", doc="V2"),
            F("FAQ::001#f00", "sick leave", "days per year", "10", unit="paid sick days", doc="FAQ"),
        ]
    )
    assert k["V2::007#f00"] == k["FAQ::001#f00"]


def test_condition_normalisation():
    assert normalize_conditions("N/A") == ""


import pytest  # noqa: E402


@pytest.mark.parametrize(
    "conditions",
    ["Employees in India", "employees in Germany", "Staff based in the Berlin office", "for employees of Veltrix Systems Inc.",
     "Bengaluru and Austin offices", "employees in the Berlin and Munich offices", "the London office"],
)
def test_where_only_conditions_are_no_condition(conditions):
    """Eval finding: the model writes the region into conditions despite the prompt,
    which split the India addendum away from the global rule it overrides."""
    assert normalize_conditions(conditions) == ""


@pytest.mark.parametrize(
    "conditions",
    ["levels L1-L3", "employees with 5 or more years of service", "primary caregiver", "employees in India after probation",
     # one row per office in a single table: different values, so different rules
     "Bengaluru, India", "home office", "remote offices"],
)
def test_real_conditions_are_kept(conditions):
    assert normalize_conditions(conditions) != ""
    assert normalize_conditions("Levels L1–L3") == normalize_conditions("levels l1-l3")
    assert normalize_conditions("All employees") == normalize_conditions(None) == ""


def _unit(rows):
    v = np.array(rows, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_average_linkage_merges_close_pairs_only():
    from app.ingestion.rule_keys import average_linkage_labels

    v = _unit([[1, 0, 0], [0.99, 0.1, 0], [0, 1, 0], [0, 0.99, 0.1], [0, 0, 1]])
    labels = average_linkage_labels(v, threshold=0.05)
    assert labels[0] == labels[1] and labels[2] == labels[3]
    assert len(set(labels)) == 3


def test_average_linkage_does_not_chain():
    """a~b and b~c are each within the threshold, but a and c are not: single
    linkage would chain all three; average linkage keeps c apart."""
    from app.ingestion.rule_keys import average_linkage_labels

    angles = np.radians([0, 20, 40])
    v = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    pair = 1 - np.cos(np.radians(20))  # distance between neighbours
    labels = average_linkage_labels(v, threshold=pair * 1.05)
    assert labels[0] == labels[1] != labels[2]


def test_average_linkage_trivial_inputs():
    from app.ingestion.rule_keys import average_linkage_labels

    assert average_linkage_labels(_unit([[1, 0]]), 0.1) == [0]
    assert average_linkage_labels(np.zeros((0, 2)), 0.1) == []


def test_same_document_different_keys_never_merge():
    """Primary and secondary caregiver weeks embed almost identically, but one
    document states both, so they are different rules."""
    global _VECTORS
    _VECTORS = {
        **_VECTORS,
        "parental leave: primary caregiver weeks": [0.6, 0.6, 0.5],
        "parental leave: secondary caregiver weeks": [0.6, 0.59, 0.52],
    }
    k = keys(
        [
            F("FAQ::004#f00", "parental leave", "primary caregiver weeks", "16", "weeks", doc="FAQ"),
            F("FAQ::004#f01", "parental leave", "secondary caregiver weeks", "2", "weeks", doc="FAQ"),
            F("V2::008#f00", "parental leave", "primary caregiver weeks", "20", "weeks", doc="V2"),
            F("V2::008#f01", "parental leave", "secondary caregiver weeks", "4", "weeks", doc="V2"),
        ]
    )
    assert k["FAQ::004#f00"] == k["V2::008#f00"] != k["FAQ::004#f01"] == k["V2::008#f01"]


def test_recommendation_never_merges_with_entitlement():
    """Eval finding: "encouraged to take at least ten days" shared a key with
    Germany's 30-day entitlement and was answered as India's entitlement."""
    global _VECTORS
    _VECTORS = {
        **_VECTORS,
        "annual leave: days per year": [0.5, 0.5, 0.7],
        "annual leave: recommended minimum days to take": [0.5, 0.5, 0.71],
        "annual leave: encouraged minimum days to take": [0.5, 0.5, 0.72],
    }
    k = keys(
        [
            F("DE::001#f00", "annual leave", "days per year", "30", doc="DE"),
            F("V1::005#f00", "annual leave", "encouraged minimum days to take", "ten working days", doc="V1"),
            F("V2::005#f00", "annual leave", "recommended minimum days to take", "ten working days", doc="V2"),
        ]
    )
    assert k["V1::005#f00"] == k["V2::005#f00"] != k["DE::001#f00"]


@pytest.mark.parametrize(
    "attribute, expected",
    [
        ("expiry", "deadline to use"),
        ("Expiration date", "deadline to use"),
        ("carry forward expiry", "carry forward deadline to use"),
        ("lapse", "deadline to use"),
        ("valid until", "deadline to use"),
        ("use-by date", "deadline to use"),
        ("deadline to use", "deadline to use"),
        ("days per year", "days per year"),  # unrelated attributes untouched
        ("forfeiture date for unused days", "forfeiture date for unused days"),  # a different rule
    ],
)
def test_validity_wording_is_normalised(attribute, expected):
    assert normalize_attribute(attribute) == expected


def test_negative_rule_worded_as_expiry_meets_the_deadline_it_denies():
    """B1 finding: India's "do not expire on a fixed date" was keyed "expiry", the global
    "31 March" "deadline to use"; precedence never compared them and India got "31 March"."""
    k = keys(
        [
            F("V2::004#f00", "carry forward", "deadline to use", "31 March", doc="V2"),
            F("IN::002#f00", "carry forward", "expiry", "do not expire on a fixed date", doc="IN"),
        ]
    )
    assert k["V2::004#f00"] == k["IN::002#f00"]


def test_cannot_link_survives_merges():
    from app.ingestion.rule_keys import average_linkage_labels

    v = _unit([[1, 0], [0.999, 0.04], [0.998, 0.06]])
    forbid = np.zeros((3, 3), dtype=bool)
    forbid[0, 2] = forbid[2, 0] = True
    labels = average_linkage_labels(v, threshold=0.5, cannot_link=forbid)
    # 1 and 2 are closest and merge first; 0 is then barred from that cluster
    # because it contains 2, however close 0 is to 1.
    assert labels[1] == labels[2] != labels[0]


# --- LLM judge for near misses ----------------------------------------------

def _at_distance(d, axis=2):
    """A unit vector at cosine distance `d` from [1, 0, 0], leaning towards `axis`."""
    v = [1 - d, 0.0, 0.0]
    v[axis] = float(np.sqrt(1 - (1 - d) ** 2))
    return v


# Eval finding: the WFH memo, the policy and the FAQ named one rule three ways.
_VECTORS["remote work days: maximum days per week"] = [1.0, 0.0, 0.0]
_VECTORS["hybrid work: days per week"] = _at_distance(0.18)
_VECTORS["home office days: maximum days per week"] = _at_distance(0.20, axis=1)
_VECTORS["learning days: days per year"] = _at_distance(0.30)


class FakeJudge:
    """Says "same" for pairs whose sides share a topic word listed in `same_topics`."""

    def __init__(self, same_topics=(), answer=True):
        self.same_topics, self.answer = set(same_topics), answer
        self.asked: list[tuple[set[str], set[str]]] = []

    def __call__(self, questions):
        out = []
        for a, b in questions:
            topics_a, topics_b = {f.topic for f in a}, {f.topic for f in b}
            self.asked.append((topics_a, topics_b))
            out.append(self.answer if self.answer is None else bool((topics_a | topics_b) <= self.same_topics))
        return out


def judged_keys(facts, judge):
    return {f.fact_id: f.rule_key for f in assign_rule_keys(facts, fake_embed, threshold=0.15, judge=judge)}


WFH = [
    F("POL::001#f00", "remote work days", "maximum days per week", "2", "days", doc="POL"),
    F("MEMO::000#f00", "hybrid work", "days per week", "3", "days", "Bengaluru and Austin offices", doc="MEMO"),
    F("FAQ::006#f00", "home office days", "maximum days per week", "2", "days", doc="FAQ"),
]


def test_near_misses_merge_when_the_judge_says_same_rule():
    judge = FakeJudge({"remote work days", "hybrid work", "home office days"})
    k = judged_keys(WFH, judge)
    assert len(set(k.values())) == 1
    assert judge.asked  # the embedder alone kept them apart


def test_near_misses_stay_apart_when_the_judge_says_no_or_gives_no_answer():
    assert len(set(judged_keys(WFH, FakeJudge(answer=False)).values())) == 3
    assert len(set(judged_keys(WFH, FakeJudge(answer=None)).values())) == 3
    assert judged_keys(WFH, FakeJudge(answer=None)) == keys(WFH)  # same as no judge at all


def test_pairs_outside_the_band_are_never_asked():
    judge = FakeJudge(answer=True)
    k = judged_keys([WFH[0], F("LRN::001#f00", "learning days", "days per year", "3", doc="LRN")], judge)
    assert not judge.asked  # 0.30 apart: beyond NEAR_MISS_MAX
    assert len(set(k.values())) == 2


def test_judge_cannot_override_cannot_link():
    """Two facts of one document with different keys never merge, whatever the judge says:
    the memo also states a hybrid rule, and the judge (wrongly) says all three are one rule."""
    memo_other = F("POL::002#f00", "hybrid work", "days per week", "3", "days", doc="POL")
    k = judged_keys([WFH[0], memo_other, WFH[2]], FakeJudge(answer=True))
    assert k["POL::001#f00"] != k["POL::002#f00"]


def test_judge_is_not_asked_about_different_numbers():
    judge = FakeJudge(answer=True)
    _VECTORS["annual leave accrual: days per month (less than 2 years)"] = [0.0, 1.0, 0.1]
    _VECTORS["annual leave: days per month (5 years or more)"] = [0.0, 0.85, 0.5]
    k = judged_keys(
        [
            F("A::001#f00", "annual leave accrual", "days per month", "1.5", None, "less than 2 years", doc="A"),
            F("B::001#f00", "annual leave", "days per month", "2", None, "5 years or more", doc="B"),
        ],
        judge,
    )
    assert not judge.asked
    assert len(set(k.values())) == 2
