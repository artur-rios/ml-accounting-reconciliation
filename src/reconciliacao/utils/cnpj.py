import random as _random


def _calc_check_digit(digits: list[int], weights: list[int]) -> int:
    s = sum(d * w for d, w in zip(digits, weights))
    r = s % 11
    return 0 if r < 2 else 11 - r


def generate_cnpj(rng: _random.Random | None = None) -> str:
    r = rng or _random.Random()
    digits = [r.randint(0, 9) for _ in range(12)]
    w1 = [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
    w2 = [6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
    d1 = _calc_check_digit(digits, w1)
    d2 = _calc_check_digit(digits + [d1], w2)
    return "".join(map(str, digits + [d1, d2]))


def validate_cnpj(cnpj: str) -> bool:
    digits_str = "".join(c for c in cnpj if c.isdigit())
    if len(digits_str) != 14:
        return False
    digits = [int(c) for c in digits_str]
    w1 = [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
    w2 = [6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
    d1 = _calc_check_digit(digits[:12], w1)
    d2 = _calc_check_digit(digits[:12] + [d1], w2)
    return digits[12] == d1 and digits[13] == d2


def normalize_cnpj(cnpj: str) -> str:
    return "".join(c for c in cnpj if c.isdigit()).zfill(14)
