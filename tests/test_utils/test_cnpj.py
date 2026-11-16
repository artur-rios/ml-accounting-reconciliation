import random
from reconciliacao.utils.cnpj import generate_cnpj, validate_cnpj, normalize_cnpj


def test_given_a_seeded_generator_when_a_cnpj_is_generated_then_it_has_14_digits():
    """Seeded, like every other generator call in the suite."""
    assert len(generate_cnpj(random.Random(7))) == 14


def test_given_generated_cnpjs_when_validated_then_each_is_valid():
    rng = random.Random(42)
    for _ in range(100):
        assert validate_cnpj(generate_cnpj(rng))


def test_given_the_same_seed_when_cnpjs_are_generated_then_they_match():
    rng1 = random.Random(42)
    rng2 = random.Random(42)
    assert generate_cnpj(rng1) == generate_cnpj(rng2)


def test_given_a_known_valid_cnpj_when_validated_then_it_passes():
    # 11.222.333/0001-81 — check digits: d1=8, d2=1
    assert validate_cnpj("11222333000181")


def test_given_a_known_invalid_cnpj_when_validated_then_it_fails():
    assert not validate_cnpj("11222333000182")


def test_given_a_formatted_cnpj_when_validated_then_it_passes():
    assert validate_cnpj("11.222.333/0001-81")


def test_given_a_cnpj_of_the_wrong_length_when_validated_then_it_fails():
    assert not validate_cnpj("1122233300018")


def test_given_a_formatted_cnpj_when_normalised_then_only_digits_remain():
    assert normalize_cnpj("11.222.333/0001-81") == "11222333000181"


def test_given_a_short_cnpj_when_normalised_then_it_is_zero_padded():
    # Already 14 digits but with leading zero stripped
    assert normalize_cnpj("1222333000181") == "01222333000181"
