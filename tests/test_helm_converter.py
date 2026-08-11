from fennec.featurizer import helm_one_hot_encode
from fennec.helm_converter import create_helm_notation, helm_to_fennec


def assert_one_hot_equivalent(left: str, right: str):
    assert helm_one_hot_encode(left) == helm_one_hot_encode(right)


def test_round_trip_new_featurizer_modifications():
    guide = "VP i s Y34 L10"
    passenger = "Ada s Gams Y44"

    helm = create_helm_notation(guide, passenger)
    restored_guide, restored_passenger = helm_to_fennec(helm)

    assert helm.endswith("$$$$V2.0")
    assert restored_guide.startswith("VP ")
    assert_one_hot_equivalent(restored_guide, guide)
    assert_one_hot_equivalent(restored_passenger, passenger)


def test_round_trip_lowercase_deoxy_t():
    guide = "t s Uf"
    passenger = "u"

    helm = create_helm_notation(guide, passenger)
    restored_guide, restored_passenger = helm_to_fennec(helm)

    assert_one_hot_equivalent(restored_guide, guide)
    assert restored_passenger == passenger


def test_round_trip_leading_vp_plain_rna():
    guide = "VP U a a c dA c u"
    passenger = "a s u"

    helm = create_helm_notation(guide, passenger)
    restored_guide, restored_passenger = helm_to_fennec(helm)

    assert helm.startswith("RNA1{[VP].R(U)P")
    assert helm.endswith("$$$$V2.0")
    assert restored_guide.startswith("VP ")
    assert_one_hot_equivalent(restored_guide, guide)
    assert_one_hot_equivalent(restored_passenger, passenger)