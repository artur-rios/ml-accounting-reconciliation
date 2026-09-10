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


def _corrupt_text(texto: str, rng: random.Random) -> str:
    """Abbreviate, transpose characters and strip accents, as a payment memo would."""
    out = []
    for palavra in texto.split():
        r = rng.random()
        if r < 0.25 and len(palavra) > 4:
            out.append(palavra[:3])
        elif r < 0.40 and len(palavra) > 3:
            i = rng.randrange(len(palavra) - 1)
            out.append(palavra[:i] + palavra[i + 1] + palavra[i] + palavra[i + 2:])
        else:
            out.append(palavra)
    return " ".join(out).translate(_ACCENTS)


def _illegitimate_retention(
    rng: random.Random, combos: dict[str, float], tol_pp: float
) -> float:
    """A withholding percentage that is clearly outside every legal band.

    The margin is 6x the tolerance, not 1x: valor_pago is rounded to cents
    afterwards, which shifts the effective percentage slightly, and a margin
    equal to the tolerance the tests assert against would make them flaky.
    """
    for _ in range(200):
        candidato = rng.uniform(0.0, 20.0)
        if all(abs(candidato - v) > tol_pp * 6 for v in combos.values()):
            return candidato
    return 17.3


def generate_comparison_dataset(
    cmp_cfg: dict, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (df_pagamentos, df_nfse, df_verdade) for one seed."""
    rng = random.Random(seed)
    Faker.seed(seed)
    fake = Faker("pt_BR")

    n = cmp_cfg["n_records"]
    rates = cmp_cfg["retention_rates"]
    tol = cmp_cfg["retention_tolerance_pp"]
    limite_csrf = cmp_cfg["csrf_threshold_brl"]

    cnpjs: list[str] = []
    vistos: set[str] = set()
    while len(vistos) < n:
        c = generate_cnpj(rng)
        if c not in vistos:
            vistos.add(c)
            cnpjs.append(c)

    prazo_fornecedor = {c: rng.choice(cmp_cfg["payment_terms_days"]) for c in cnpjs}
    municipio_fornecedor = {c: rng.choice(_MUNICIPIOS) for c in cnpjs}

    centros = [f"CC-{i:03d}" for i in range(100, 200)]
    municipio_centro = {cc: rng.choice(_MUNICIPIOS) for cc in centros}
    centros_por_municipio: dict[str, list[str]] = {}
    for cc, m in municipio_centro.items():
        centros_por_municipio.setdefault(m, []).append(cc)

    n_match = int(n * cmp_cfg["match_rate"])
    eh_par = [True] * n_match + [False] * (n - n_match)
    rng.shuffle(eh_par)

    n_neg = n - n_match
    n_hard = int(n_neg * cmp_cfg["hard_negative_rate"])
    flags_hard = [True] * n_hard + [False] * (n_neg - n_hard)
    rng.shuffle(flags_hard)
    iter_hard = iter(flags_hard)

    cnpj_tomador = generate_cnpj(rng)
    razao_tomador = fake.company()

    pagamentos, notas, verdade = [], [], []

    for i, cnpj in enumerate(cnpjs):
        valor_servicos = round(rng.uniform(500.0, 50_000.0), 2)
        aliquota = round(rng.uniform(2.0, 5.0), 2)
        emissao = fake.date_between(start_date="-1y", end_date="-2m")
        discriminacao = fake.bs()
        numero = f"{i + 1:06d}"
        valor_iss = round(valor_servicos * aliquota / 100, 2)

        notas.append({
            "nfse_numero": numero,
            "nfse_codigo_verificacao": "".join(
                rng.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", k=8)
            ),
            "nfse_data_emissao": emissao.isoformat() + "T00:00:00",
            "nfse_competencia": emissao.replace(day=1).isoformat() + "T00:00:00",
            "nfse_cnpj_prestador": cnpj,
            "nfse_razao_social_prestador": fake.company(),
            "nfse_cnpj_tomador": cnpj_tomador,
            "nfse_razao_social_tomador": razao_tomador,
            "nfse_valor_servicos": valor_servicos,
            "nfse_valor_iss": valor_iss,
            "nfse_aliquota": aliquota,
            "nfse_valor_liquido": round(valor_servicos - valor_iss, 2),
            "nfse_item_lista_servico": rng.choice(_LC116_CODES),
            "nfse_discriminacao": discriminacao,
            "nfse_codigo_municipio": municipio_fornecedor[cnpj],
        })

        combos = legal_combos(valor_servicos, aliquota, rates, limite_csrf)
        no_municipio = centros_por_municipio.get(municipio_fornecedor[cnpj], centros)

        if eh_par[i]:
            retencao = combos[rng.choice(list(combos))]
            prazo = prazo_fornecedor[cnpj]
            if rng.random() < cmp_cfg["atypical_term_rate"]:
                prazo = max(0, prazo + rng.choice([-10, -5, 7, 14, 21]))
            descricao = _corrupt_text(discriminacao, rng)
            if rng.random() < cmp_cfg["same_municipality_rate"]:
                centro = rng.choice(no_municipio)
            else:
                centro = rng.choice(centros)
            tipo_negativo = ""
            nfse_verdadeira = numero
        else:
            hard = next(iter_hard)
            if hard:
                retencao = combos[rng.choice(list(combos))]
            else:
                retencao = _illegitimate_retention(rng, combos, tol)

            dimensoes = rng.sample(["prazo", "texto", "municipio"], k=rng.randint(1, 3))
            prazo = prazo_fornecedor[cnpj]
            if "prazo" in dimensoes:
                prazo = max(0, prazo + rng.choice([-30, -20, 25, 40, 60]))
            descricao = fake.bs() if "texto" in dimensoes else _corrupt_text(discriminacao, rng)
            centro = rng.choice(centros) if "municipio" in dimensoes else rng.choice(no_municipio)
            tipo_negativo = "hard" if hard else "soft"
            nfse_verdadeira = ""

        pagamentos.append({
            "id_pagamento": f"PAG-{i + 1:06d}",
            "cnpj_fornecedor": cnpj,
            "data_pagamento": emissao + timedelta(days=prazo),
            "valor_pago": round(valor_servicos * (1 - retencao / 100), 2),
            "descricao": descricao,
            "centro_custo": centro,
            "municipio_centro_custo": municipio_centro[centro],
        })
        verdade.append({
            "id_pagamento": f"PAG-{i + 1:06d}",
            "nfse_numero": nfse_verdadeira,
            "tipo_negativo": tipo_negativo,
        })

    return pd.DataFrame(pagamentos), pd.DataFrame(notas), pd.DataFrame(verdade)
