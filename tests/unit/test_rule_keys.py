"""Rule keys: the same rule stated differently lands together; incompatible facts never merge."""

import numpy as np

from app.ingestion.extract_facts import Fact
from app.ingestion.rule_keys import assign_rule_keys, normalize_conditions, normalize_unit

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
    assert k["V1::005#f00"] == "sick-leave-days-per-year|day"


def test_different_conditions_never_merge():
    k = keys(
        [
            F("A::001#f00", "annual leave accrual", "days per month", "1.5", "days per month", "less than 2 years"),
            F("A::001#f01", "annual leave accrual", "days per month", "1.75", "days per month", "2 years to less than 5 years"),
        ]
    )
    assert len(set(k.values())) == 2


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


def test_unit_and_condition_normalisation():
    assert normalize_unit("Working Days") == normalize_unit("days") == normalize_unit("business day") == "day"
    assert normalize_unit("days per month") == "day per month"
    assert normalize_unit("calendar days") == "calendar day"
    assert normalize_unit(None) == normalize_unit("N/A") == ""
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


def test_cannot_link_survives_merges():
    from app.ingestion.rule_keys import average_linkage_labels

    v = _unit([[1, 0], [0.999, 0.04], [0.998, 0.06]])
    forbid = np.zeros((3, 3), dtype=bool)
    forbid[0, 2] = forbid[2, 0] = True
    labels = average_linkage_labels(v, threshold=0.5, cannot_link=forbid)
    # 1 and 2 are closest and merge first; 0 is then barred from that cluster
    # because it contains 2, however close 0 is to 1.
    assert labels[1] == labels[2] != labels[0]
