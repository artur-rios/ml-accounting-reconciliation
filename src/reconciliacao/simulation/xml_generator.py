import random
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from faker import Faker
from lxml import etree

from reconciliacao.utils.cnpj import generate_cnpj

_ABRASF_NS = "http://www.abrasf.org.br/nfse.xsd"
_LC116_CODES = ["1.01", "1.02", "1.03", "1.04", "1.05", "7.01", "7.02", "14.01"]
_MUNICIPIOS = ["3550308", "3304557", "4106902", "2304400", "5300108"]


def _sub(parent: etree._Element, tag: str, text: str | None = None) -> etree._Element:
    el = etree.SubElement(parent, tag)
    if text is not None:
        el.text = str(text)
    return el


def generate_nfse(df_pag: pd.DataFrame, conciliation_rate: float, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    Faker.seed(seed)
    fake = Faker("pt_BR")

    n = len(df_pag)
    conciliated_indices = set(rng.sample(range(n), k=int(n * conciliation_rate)))
    mismatch_types = ["cnpj_error", "date_shift", "value_discrepancy", "ghost"]

    records = []
    for i, row in enumerate(df_pag.itertuples(index=False)):
        pag_date: date = row.data_pagamento
        pag_valor: float = row.valor_pago
        pag_cnpj: str = row.cnpj_fornecedor

        if i in conciliated_indices:
            cnpj = pag_cnpj
            emit_date = pag_date
            valor = pag_valor
        else:
            mismatch = rng.choice(mismatch_types)
            if mismatch == "cnpj_error":
                # Flip one of the first 12 digits to produce an invalid CNPJ
                digits = list(pag_cnpj)
                pos = rng.randint(0, 11)
                digits[pos] = str((int(digits[pos]) + rng.randint(1, 9)) % 10)
                cnpj = "".join(digits)
                emit_date = pag_date
                valor = pag_valor
            elif mismatch == "date_shift":
                cnpj = pag_cnpj
                shift = rng.randint(6, 30) * rng.choice([-1, 1])
                emit_date = pag_date + timedelta(days=shift)
                valor = pag_valor
            elif mismatch == "value_discrepancy":
                cnpj = pag_cnpj
                emit_date = pag_date
                factor = 1 + rng.uniform(0.03, 0.20) * rng.choice([-1, 1])
                valor = round(pag_valor * factor, 2)
            else:  # ghost — completely unrelated invoice
                cnpj = generate_cnpj(rng)
                emit_date = fake.date_between(start_date="-1y", end_date="today")
                valor = round(rng.uniform(500.0, 50_000.0), 2)

        aliquota = round(rng.uniform(2.0, 5.0), 2)
        valor_iss = round(valor * aliquota / 100, 2)

        records.append({
            "nfse_numero": f"{i + 1:06d}",
            "nfse_codigo_verificacao": "".join(
                rng.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", k=8)
            ),
            "nfse_data_emissao": emit_date.isoformat() + "T00:00:00",
            "nfse_competencia": emit_date.replace(day=1).isoformat() + "T00:00:00",
            "nfse_cnpj_prestador": cnpj,
            "nfse_razao_social_prestador": fake.company(),
            "nfse_cnpj_tomador": generate_cnpj(rng),
            "nfse_razao_social_tomador": fake.company(),
            "nfse_valor_servicos": valor,
            "nfse_valor_iss": valor_iss,
            "nfse_aliquota": aliquota,
            "nfse_valor_liquido": round(valor - valor_iss, 2),
            "nfse_item_lista_servico": rng.choice(_LC116_CODES),
            "nfse_discriminacao": fake.bs(),
            "nfse_codigo_municipio": rng.choice(_MUNICIPIOS),
        })

    return pd.DataFrame(records)


def write_xml(df: pd.DataFrame, path: str | Path) -> None:
    nsmap = {None: _ABRASF_NS}
    root = etree.Element("ListaNfse", nsmap=nsmap)

    for _, row in df.iterrows():
        comp = _sub(root, "CompNfse")
        nfse = _sub(comp, "Nfse")
        inf = _sub(nfse, "InfNfse")

        _sub(inf, "Numero", row["nfse_numero"])
        _sub(inf, "CodigoVerificacao", row["nfse_codigo_verificacao"])
        _sub(inf, "DataEmissao", row["nfse_data_emissao"])
        _sub(inf, "Competencia", row["nfse_competencia"])

        prest = _sub(inf, "PrestadorServico")
        id_prest = _sub(prest, "IdentificacaoPrestador")
        _sub(_sub(id_prest, "CpfCnpj"), "Cnpj", row["nfse_cnpj_prestador"])
        _sub(id_prest, "InscricaoMunicipal", "000001")
        _sub(prest, "RazaoSocial", row["nfse_razao_social_prestador"])

        tom = _sub(inf, "TomadorServico")
        id_tom = _sub(tom, "IdentificacaoTomador")
        _sub(_sub(id_tom, "CpfCnpj"), "Cnpj", row["nfse_cnpj_tomador"])
        _sub(tom, "RazaoSocial", row["nfse_razao_social_tomador"])

        servico = _sub(inf, "Servico")
        valores = _sub(servico, "Valores")
        _sub(valores, "ValorServicos", f"{row['nfse_valor_servicos']:.2f}")
        _sub(valores, "ValorIss", f"{row['nfse_valor_iss']:.2f}")
        _sub(valores, "Aliquota", f"{row['nfse_aliquota']:.2f}")
        _sub(valores, "ValorLiquidoNfse", f"{row['nfse_valor_liquido']:.2f}")
        _sub(servico, "ItemListaServico", row["nfse_item_lista_servico"])
        _sub(servico, "Discriminacao", row["nfse_discriminacao"])
        _sub(servico, "CodigoMunicipio", row["nfse_codigo_municipio"])

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    etree.ElementTree(root).write(
        str(path), xml_declaration=True, encoding="UTF-8", pretty_print=True
    )
