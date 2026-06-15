from pathlib import Path

import pandas as pd
from lxml import etree

_NS = {"ns": "http://www.abrasf.org.br/nfse.xsd"}


def _text(el: etree._Element, xpath: str) -> str:
    nodes = el.xpath(xpath, namespaces=_NS)
    return nodes[0].text.strip() if nodes and nodes[0].text else ""


def load_pagamentos(path: str | Path) -> pd.DataFrame:
    return pd.read_excel(path, engine="openpyxl", dtype=str)


def load_nfse(path: str | Path) -> pd.DataFrame:
    tree = etree.parse(str(path))
    records = []
    for comp in tree.xpath("//ns:CompNfse", namespaces=_NS):
        records.append({
            "nfse_numero": _text(comp, ".//ns:InfNfse/ns:Numero"),
            "nfse_codigo_verificacao": _text(comp, ".//ns:CodigoVerificacao"),
            "nfse_data_emissao": _text(comp, ".//ns:DataEmissao"),
            "nfse_competencia": _text(comp, ".//ns:Competencia"),
            "nfse_cnpj_prestador": _text(comp, ".//ns:PrestadorServico//ns:Cnpj"),
            "nfse_razao_social_prestador": _text(comp, ".//ns:PrestadorServico/ns:RazaoSocial"),
            "nfse_cnpj_tomador": _text(comp, ".//ns:TomadorServico//ns:Cnpj"),
            "nfse_razao_social_tomador": _text(comp, ".//ns:TomadorServico/ns:RazaoSocial"),
            "nfse_valor_servicos": _text(comp, ".//ns:ValorServicos"),
            "nfse_valor_iss": _text(comp, ".//ns:ValorIss"),
            "nfse_aliquota": _text(comp, ".//ns:Aliquota"),
            "nfse_valor_liquido": _text(comp, ".//ns:ValorLiquidoNfse"),
            "nfse_item_lista_servico": _text(comp, ".//ns:ItemListaServico"),
            "nfse_discriminacao": _text(comp, ".//ns:Discriminacao"),
            "nfse_codigo_municipio": _text(comp, ".//ns:CodigoMunicipio"),
        })
    return pd.DataFrame(records)
