"""Organización del material por temas con aprendizaje no supervisado.

Agrupa los fragmentos con K-Means sobre sus vectores TF-IDF y elige el número
de temas con el coeficiente de silueta. Se reporta también el costo de cómputo
(tiempo), como pide la validación del curso.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from langchain_core.documents import Document

from .ml import kmeans, silueta as coef_silueta
from .recuperador import Indice


@dataclass
class Tema:
    numero: int
    palabras: list[str]
    fragmentos: list[Document]
    titulo: str = ""
    sugerencia: str = ""   # pregunta de ejemplo para el estudiante

    @property
    def etiquetas(self) -> list[str]:
        vistas = []
        for f in self.fragmentos:
            e = f.metadata["etiqueta"]
            if e not in vistas:
                vistas.append(e)
        return vistas

    @property
    def nombre(self) -> str:
        return self.titulo or ", ".join(self.palabras[:3]).capitalize()


@dataclass
class Agrupamiento:
    temas: list[Tema]
    k: int
    silueta: float | None
    segundos: float
    evaluados: dict[int, float] = field(default_factory=dict)  # k -> silueta


def _orden_lectura(f: Document) -> tuple:
    return (f.metadata.get("doc", 0), f.metadata.get("pagina", 0), f.metadata.get("id", 0))


def agrupar(indice: Indice, k_max: int = 8, semilla: int = 42) -> Agrupamiento:
    inicio = time.perf_counter()
    X = indice.matriz
    n = X.shape[0]
    terminos = indice.vectorizador.terminos()

    if n < 4:
        etiquetas, k, silueta, evaluados = np.zeros(n, dtype=int), 1, None, {}
        centros = X.mean(axis=0, keepdims=True)
    else:
        mejor, evaluados = None, {}
        for k in range(2, min(k_max, n // 2) + 1):
            grupos, centros_k = kmeans(X, k, semilla=semilla)
            if len(set(grupos.tolist())) < 2:
                continue
            s = coef_silueta(X, grupos)
            evaluados[k] = round(s, 3)
            if mejor is None or s > mejor[0]:
                mejor = (s, k, (grupos, centros_k))
        if mejor is None:
            etiquetas, k, silueta = np.zeros(n, dtype=int), 1, None
            centros = X.mean(axis=0, keepdims=True)
        else:
            silueta, k, (etiquetas, centros) = mejor

    temas = []
    for c in range(k):
        miembros = [indice.fragmentos[i] for i in np.where(etiquetas == c)[0]]
        if not miembros:
            continue
        top = np.argsort(-centros[c])[:6]
        palabras = [indice.tok.forma(t) for t in terminos[top]]
        temas.append(Tema(0, palabras, sorted(miembros, key=_orden_lectura)))
    # Los temas se presentan en el orden en que aparecen en el documento.
    temas.sort(key=lambda t: _orden_lectura(t.fragmentos[0]))
    for i, t in enumerate(temas, start=1):
        t.numero = i
    return Agrupamiento(temas, len(temas), silueta, time.perf_counter() - inicio, evaluados)
