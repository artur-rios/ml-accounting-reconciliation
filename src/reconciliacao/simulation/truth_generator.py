# src/reconciliacao/simulation/truth_generator.py
"""Generate a reconciliation dataset whose label comes from ground truth.

Every payment is paired with exactly one candidate invoice from the same
supplier, and the truth table says whether that pairing is genuine. Labels
therefore never derive from the observable deltas -- the decoupling is by
construction, which is what the original experiment lacked.
"""

import random
from datetime import timedelta

import pandas as pd
from faker import Faker

from reconciliacao.simulation.retentions import legal_combos
from reconciliacao.utils.cnpj import generate_cnpj

_LC116_CODES = ["1.01", "1.02", "1.03", "1.04", "1.05", "7.01", "7.02", "14.01"]
_MUNICIPIOS = ["3550308", "3304557", "4106902", "2304400", "5300108"]
_ACCENTS = str.maketrans("áàâãéêíóôõúüç", "aaaaeeiooouuc")


def _corrupt_text(text: str, rng: random.Random) -> str:
    """Abbreviate, transpose characters and strip accents, as a payment memo would."""
    out = []
    for word in text.split():
        r = rng.random()
        if r < 0.25 and len(word) > 4:
            out.append(word[:3])
        elif r < 0.40 and len(word) > 3:
            i = rng.randrange(len(word) - 1)
            out.append(word[:i] + word[i + 1] + word[i] + word[i + 2:])
        else:
            out.append(word)
    return " ".join(out).translate(_ACCENTS)


def _illegitimate_retention(
    rng: random.Random, combinations: dict[str, float], tolerance_pp: float
) -> float:
    """A withholding percentage that is clearly outside every legal band.

    The margin is 6x the tolerance, not 1x: valor_pago is rounded to cents
    afterwards, which shifts the effective percentage slightly, and a margin
    equal to the tolerance the tests assert against would make them flaky.
    """
    for _ in range(200):
        candidate = rng.uniform(0.0, 20.0)
        if all(abs(candidate - v) > tolerance_pp * 6 for v in combinations.values()):
            return candidate
    return 17.3


def _shifted_term(rng: random.Random, term: int, shift_range: list[int]) -> int:
    """Draw a payment term that differs from the supplier's own, clamped at zero.

    Both true pairs and negatives call this with the same ``shift_range``, so
    the *set* of atypical delta_dias values is identical between classes --
    only how *often* each class uses it differs. That keeps desvio_prazo_fornecedor
    informative without making any single delta_dias value class-unique, which a
    deep model could otherwise memorise.

    Redraws when clamping collapses the result back onto ``term`` (e.g. a
    supplier whose own term is already 0 and the draw is non-positive) --
    otherwise the configured atypical rate would silently under-deliver for
    those suppliers.
    """
    new_term = term
    while new_term == term:
        new_term = max(0, term + rng.randint(shift_range[0], shift_range[1]))
    return new_term


def generate_comparison_dataset(
    config: dict, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (df_pagamentos, df_nfse, df_verdade) for one seed."""
    rng = random.Random(seed)
    Faker.seed(seed)
    fake = Faker("pt_BR")

    n = config["n_records"]
    rates = config["retention_rates"]
    tolerance = config["retention_tolerance_pp"]
    csrf_threshold = config["csrf_threshold_brl"]

    cnpjs: list[str] = []
    seen: set[str] = set()
    while len(seen) < n:
        c = generate_cnpj(rng)
        if c not in seen:
            seen.add(c)
            cnpjs.append(c)

    supplier_payment_terms = {c: rng.choice(config["payment_terms_days"]) for c in cnpjs}
    supplier_municipality = {c: rng.choice(_MUNICIPIOS) for c in cnpjs}

    cost_centers = [f"CC-{i:03d}" for i in range(100, 200)]
    cost_center_municipality = {cc: rng.choice(_MUNICIPIOS) for cc in cost_centers}
    cost_centers_by_municipality: dict[str, list[str]] = {}
    for cc, m in cost_center_municipality.items():
        cost_centers_by_municipality.setdefault(m, []).append(cc)

    n_match = int(n * config["match_rate"])
    is_match = [True] * n_match + [False] * (n - n_match)
    rng.shuffle(is_match)

    n_neg = n - n_match
    n_hard = int(n_neg * config["hard_negative_rate"])
    hard_flags = [True] * n_hard + [False] * (n_neg - n_hard)
    rng.shuffle(hard_flags)
    hard_iterator = iter(hard_flags)

    buyer_cnpj = generate_cnpj(rng)
    buyer_company_name = fake.company()

    payments, invoices, ground_truth = [], [], []

    for i, cnpj in enumerate(cnpjs):
        service_value = round(rng.uniform(*config["invoice_value_range_brl"]), 2)
        tax_rate = round(rng.uniform(*config["iss_rate_range_pct"]), 2)
        issue_date = fake.date_between(start_date="-1y", end_date="-2m")
        description = fake.bs()
        invoice_number = f"{i + 1:06d}"
        iss_amount = round(service_value * tax_rate / 100, 2)

        invoices.append({
            "nfse_numero": invoice_number,
            "nfse_codigo_verificacao": "".join(
                rng.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", k=8)
            ),
            "nfse_data_emissao": issue_date.isoformat() + "T00:00:00",
            "nfse_competencia": issue_date.replace(day=1).isoformat() + "T00:00:00",
            "nfse_cnpj_prestador": cnpj,
            "nfse_razao_social_prestador": fake.company(),
            "nfse_cnpj_tomador": buyer_cnpj,
            "nfse_razao_social_tomador": buyer_company_name,
            "nfse_valor_servicos": service_value,
            "nfse_valor_iss": iss_amount,
            "nfse_aliquota": tax_rate,
            "nfse_valor_liquido": round(service_value - iss_amount, 2),
            "nfse_item_lista_servico": rng.choice(_LC116_CODES),
            "nfse_discriminacao": description,
            "nfse_codigo_municipio": supplier_municipality[cnpj],
        })

        combinations = legal_combos(service_value, tax_rate, rates, csrf_threshold)
        municipality_centers = cost_centers_by_municipality.get(supplier_municipality[cnpj], cost_centers)

        if is_match[i]:
            retention_rate = combinations[rng.choice(list(combinations))]
            term = supplier_payment_terms[cnpj]
            if rng.random() < config["atypical_term_rate"]:
                term = _shifted_term(rng, term, config["term_shift_range"])
            if rng.random() < config["atypical_description_rate"]:
                payment_description = fake.bs()
            else:
                payment_description = _corrupt_text(description, rng)
            if rng.random() < config["same_municipality_rate"]:
                cost_center = rng.choice(municipality_centers)
            else:
                cost_center = rng.choice(cost_centers)
            negative_type = ""
            true_invoice_number = invoice_number
        else:
            is_hard = next(hard_iterator)
            if is_hard:
                retention_rate = combinations[rng.choice(list(combinations))]
            else:
                retention_rate = _illegitimate_retention(rng, combinations, tolerance)

            dimensions = rng.sample(["prazo", "texto", "municipio"], k=rng.randint(1, 3))
            term = supplier_payment_terms[cnpj]
            if "prazo" in dimensions:
                term = _shifted_term(rng, term, config["term_shift_range"])
            payment_description = fake.bs() if "texto" in dimensions else _corrupt_text(description, rng)
            cost_center = rng.choice(cost_centers) if "municipio" in dimensions else rng.choice(municipality_centers)
            negative_type = "hard" if is_hard else "soft"
            true_invoice_number = ""

        payments.append({
            "id_pagamento": f"PAG-{i + 1:06d}",
            "cnpj_fornecedor": cnpj,
            "data_pagamento": issue_date + timedelta(days=term),
            "valor_pago": round(service_value * (1 - retention_rate / 100), 2),
            "descricao": payment_description,
            "centro_custo": cost_center,
            "municipio_centro_custo": cost_center_municipality[cost_center],
        })
        ground_truth.append({
            "id_pagamento": f"PAG-{i + 1:06d}",
            "nfse_numero": true_invoice_number,
            "tipo_negativo": negative_type,
        })

    return pd.DataFrame(payments), pd.DataFrame(invoices), pd.DataFrame(ground_truth)
