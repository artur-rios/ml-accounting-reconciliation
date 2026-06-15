import random
from reconciliacao.utils.cnpj import generate_cnpj, validate_cnpj, normalize_cnpj


def test_generate_cnpj_has_14_digits():
    assert len(generate_cnpj()) == 14


def test_generate_cnpj_produces_valid_cnpj():
    rng = random.Random(42)
    for _ in range(100):
        assert validate_cnpj(generate_cnpj(rng))


def test_generate_cnpj_is_reproducible():
    rng1 = random.Random(42)
    rng2 = random.Random(42)
    assert generate_cnpj(rng1) == generate_cnpj(rng2)


def test_validate_cnpj_known_valid():
    # 11.222.333/0001-81 — check digits: d1=8, d2=1
    assert validate_cnpj("11222333000181")


def test_validate_cnpj_known_invalid():
    assert not validate_cnpj("11222333000182")


def test_validate_cnpj_with_formatting():
    assert validate_cnpj("11.222.333/0001-81")


def test_validate_cnpj_wrong_length():
    assert not validate_cnpj("1122233300018")


def test_normalize_cnpj_strips_formatting():
    assert normalize_cnpj("11.222.333/0001-81") == "11222333000181"


def test_normalize_cnpj_pads_short_string():
    # Already 14 digits but with leading zero stripped
    assert normalize_cnpj("1222333000181") == "01222333000181"
