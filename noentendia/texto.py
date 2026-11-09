"""Utilidades de texto en español: normalización, stopwords, raíz ligera y oraciones.

Se usan en la recuperación (TF-IDF), en el agrupamiento por temas y en el
modo local (sin LLM). Todo es determinista y explicable: no hay modelos ocultos.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict

_STOP = """
a al algo algun alguna algunas alguno algunos ante antes aqui asi aun bajo bien
cada casi como con contra cual cuales cualquier cuando cuanto de del desde donde
dos e el ella ellas ello ellos en entre era eran es esa esas ese eso esos esta
estas este esto estos fue fueron ha han hasta hay la las le les lo los mas me mi
mientras muy mucho muchos ni no nos o otra otras otro otros para pero poco por
porque pues que quien quienes se ser si sin sobre son su sus tal tambien tan
tanto te tiene tienen todo todos tu un una unas uno unos y ya yo puede pueden
segun cuya cuyo dicha dicho asimismo mediante cosa cosas tipo tipos forma formas
vez veces parte partes ejemplo usa usan usar uso cual
"""
# Palabras propias de la forma de preguntar: no aportan al tema buscado.
_PREGUNTA = """
explica explicame explicar define definir definicion significa significado dime
dame hablame habla cuentame resume resumen resumir sirve sirven funciona funcionan
consiste consisten importante quiero saber entiendo entender documento texto
apuntes pagina paginas tema temas pregunta preguntas sencillas palabras
"""


def quitar_tildes(t: str) -> str:
    t = unicodedata.normalize("NFD", t)
    t = "".join(c for c in t if unicodedata.category(c) != "Mn" or c == "̃")
    # Conservar la ñ (n + virgulilla) y quitar el resto de diacríticos.
    return unicodedata.normalize("NFC", t)


STOPWORDS = {quitar_tildes(w) for w in (_STOP + _PREGUNTA).split()}
_PALABRA = re.compile(r"[a-záéíóúüñ0-9]+")


def raiz(palabra: str) -> str:
    """Raíz ligera: une plurales y variantes cercanas (redes→red, clases→clas)."""
    if len(palabra) > 4 and palabra.endswith("ces"):
        palabra = palabra[:-3] + "z"
    elif len(palabra) > 3 and palabra.endswith("s"):
        palabra = palabra[:-1]
    if len(palabra) > 4 and palabra[-1] in "aeo":
        palabra = palabra[:-1]
    return palabra[:9]


class Tokenizador:
    """Tokeniza y recuerda la forma más común de cada raíz (para nombrar temas)."""

    def __init__(self) -> None:
        self._formas: dict[str, Counter] = defaultdict(Counter)

    def __call__(self, texto: str) -> list[str]:
        salida = []
        for original in _PALABRA.findall(texto.lower()):
            tok = quitar_tildes(original)
            if len(tok) < 3 or tok in STOPWORDS or tok.isdigit():
                continue
            r = raiz(tok)
            self._formas[r][original] += 1   # se recuerda con tildes para mostrarla
            salida.append(r)
        return salida

    def forma(self, r: str) -> str:
        c = self._formas.get(r)
        return c.most_common(1)[0][0] if c else r


def _es_titulo(linea: str) -> bool:
    return len(linea) < 70 and not re.search(r"[.!?;:,]$", linea)


def oraciones(texto: str) -> list[str]:
    """Oraciones completas, sin títulos y con los saltos de línea del PDF unidos."""
    lineas = [l.strip() for l in texto.splitlines()]
    parrafos, actual = [], []
    for i, linea in enumerate(lineas):
        siguiente = lineas[i + 1] if i + 1 < len(lineas) else ""
        if not linea:
            if actual:
                parrafos.append(" ".join(actual)); actual = []
            continue
        # Título: línea corta sin puntuación final que no continúa una oración.
        cierra = not actual or re.search(r"[.!?:]$", actual[-1])
        if cierra and _es_titulo(linea) and (not siguiente or siguiente[:1].isupper()):
            continue
        actual.append(linea)
    if actual:
        parrafos.append(" ".join(actual))
    salida = []
    for p in parrafos:
        for o in re.split(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ¿¡0-9•\-])", p):
            o = o.strip(" •-")
            if len(o) > 25:
                salida.append(o)
    return salida


def limpiar(texto: str) -> str:
    texto = texto.replace("­", "")
    texto = re.sub(r"(\w)-\n(\w)", r"\1\2", texto)  # palabras partidas con guion
    texto = re.sub(r"[ \t]+", " ", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()
