"""Algoritmos de aprendizaje automático implementados con numpy.

Se implementan a mano (en vez de importar scikit-learn) por dos razones:
  * el despliegue serverless en Vercel tiene un límite de tamaño y
    scikit-learn + scipy lo superan;
  * cada paso queda visible y explicable en la sustentación.

Contenido:
  TfIdf     — vectorizador TF-IDF (tf sublineal, idf suavizado, norma L2).
  kmeans    — K-Means con inicialización k-means++ y varios reinicios.
  silueta   — coeficiente de silueta con distancia coseno.
"""
from __future__ import annotations

from collections import Counter
from typing import Callable

import numpy as np


class TfIdf:
    def __init__(self, tokenizador: Callable[[str], list[str]]) -> None:
        self.tokenizador = tokenizador
        self.vocabulario: dict[str, int] = {}
        self.idf: np.ndarray | None = None

    def _conteos(self, textos: list[str]) -> np.ndarray:
        m = np.zeros((len(textos), len(self.vocabulario)), dtype=np.float32)
        for i, t in enumerate(textos):
            for termino, n in Counter(self.tokenizador(t)).items():
                j = self.vocabulario.get(termino)
                if j is not None:
                    m[i, j] = n
        return m

    def ajustar_transformar(self, textos: list[str]) -> np.ndarray:
        tokens = [self.tokenizador(t) for t in textos]
        terminos = sorted({tok for ts in tokens for tok in ts})
        self.vocabulario = {t: j for j, t in enumerate(terminos)}
        conteos = self._conteos(textos)
        n = len(textos)
        df = (conteos > 0).sum(axis=0)
        self.idf = (np.log((1 + n) / (1 + df)) + 1).astype(np.float32)
        return self._pesar(conteos)

    def transformar(self, textos: list[str]) -> np.ndarray:
        return self._pesar(self._conteos(textos))

    def _pesar(self, conteos: np.ndarray) -> np.ndarray:
        tf = np.where(conteos > 0, 1 + np.log(np.maximum(conteos, 1)), 0).astype(np.float32)
        m = tf * self.idf
        normas = np.linalg.norm(m, axis=1, keepdims=True)
        return m / np.where(normas == 0, 1, normas)

    def terminos(self) -> np.ndarray:
        return np.array(sorted(self.vocabulario, key=self.vocabulario.get))


def _kmeans_pp(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    centros = [X[rng.integers(len(X))]]
    normas = (X ** 2).sum(1)
    d2 = np.full(len(X), np.inf)
    for _ in range(1, k):
        c = centros[-1]
        d2 = np.minimum(d2, np.maximum(normas - 2 * X @ c + c @ c, 0))
        total = d2.sum()
        i = rng.integers(len(X)) if total == 0 else rng.choice(len(X), p=d2 / total)
        centros.append(X[i])
    return np.array(centros)


def kmeans(X: np.ndarray, k: int, reinicios: int = 10, iteraciones: int = 100,
           semilla: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (etiquetas, centros) del reinicio con menor inercia."""
    rng = np.random.default_rng(semilla)
    mejor = None
    for _ in range(reinicios):
        centros = _kmeans_pp(X, k, rng)
        for _ in range(iteraciones):
            # ||x - c||² = ||x||² - 2 x·c + ||c||²
            d = (X ** 2).sum(1)[:, None] - 2 * X @ centros.T + (centros ** 2).sum(1)[None]
            etiquetas = d.argmin(axis=1)
            nuevos = np.array([X[etiquetas == c].mean(0) if np.any(etiquetas == c)
                               else centros[c] for c in range(k)])
            if np.allclose(nuevos, centros):
                break
            centros = nuevos
        inercia = float(d[np.arange(len(X)), etiquetas].sum())
        if mejor is None or inercia < mejor[0]:
            mejor = (inercia, etiquetas, centros)
    return mejor[1], mejor[2]


def silueta(X: np.ndarray, etiquetas: np.ndarray) -> float:
    """Coeficiente de silueta medio con distancia coseno (X ya normalizada L2)."""
    D = np.clip(1 - X @ X.T, 0, 2)
    grupos = np.unique(etiquetas)
    s = np.zeros(len(X))
    for i in range(len(X)):
        propio = etiquetas == etiquetas[i]
        if propio.sum() <= 1:
            continue  # por convención, silueta 0 para grupos de un solo elemento
        a = D[i, propio].sum() / (propio.sum() - 1)
        b = min(D[i, etiquetas == g].mean() for g in grupos if g != etiquetas[i])
        s[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0
    return float(s.mean())
