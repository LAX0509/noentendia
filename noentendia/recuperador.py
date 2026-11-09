"""Recuperación de fragmentos (la "R" de RAG) con TF-IDF y similitud coseno (implementación propia en ml.py).

Se eligió TF-IDF en lugar de embeddings neuronales porque:
  * funciona sin internet ni GPU y en milisegundos;
  * es explicable: se puede mostrar qué palabras hicieron coincidir la pregunta;
  * permite fijar un umbral claro para decir "eso no está en el documento".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from langchain_core.documents import Document
from .ml import TfIdf

from .texto import Tokenizador, oraciones

UMBRAL_SIMILITUD = 0.08   # similitud mínima del mejor fragmento
UMBRAL_COBERTURA = 0.6    # fracción mínima de términos de la pregunta presentes


def _presente(termino: str, presentes: set[str]) -> bool:
    """Coincidencia por raíz o por prefijo de 5 letras (agrupa ~ agrupamiento)."""
    if termino in presentes:
        return True
    return len(termino) >= 5 and any(p.startswith(termino[:5]) for p in presentes)


@dataclass
class Resultado:
    fragmento: Document
    puntaje: float

    @property
    def etiqueta(self) -> str:
        return self.fragmento.metadata["etiqueta"]


@dataclass
class Busqueda:
    resultados: list[Resultado]
    terminos: list[str]            # términos útiles de la pregunta
    encontrados: list[str]         # cuáles aparecen en los fragmentos
    hay_respaldo: bool             # ¿el documento contiene la respuesta?
    motivo: str = ""

    @property
    def cobertura(self) -> float:
        return len(self.encontrados) / len(self.terminos) if self.terminos else 0.0


class Indice:
    def __init__(self, fragmentos: list[Document]) -> None:
        if not fragmentos:
            raise ValueError("No hay texto para indexar.")
        self.fragmentos = fragmentos
        self.tok = Tokenizador()
        self.vectorizador = TfIdf(self.tok)
        self.matriz = self.vectorizador.ajustar_transformar(
            [f.page_content for f in fragmentos])

    def similitudes(self, texto: str) -> np.ndarray:
        q = self.vectorizador.transformar([texto])
        return (self.matriz @ q.T).ravel()

    def buscar(self, pregunta: str, k: int = 4) -> Busqueda:
        terminos = list(dict.fromkeys(self.tok(pregunta)))
        if not terminos:
            return Busqueda([], [], [], False,
                            "La pregunta no tiene palabras clave; intenta ser más específico.")
        sims = self.similitudes(pregunta)
        orden = np.argsort(-sims)[:k]
        resultados = [Resultado(self.fragmentos[i], float(sims[i]))
                      for i in orden if sims[i] > 0]
        texto_top = " ".join(r.fragmento.page_content for r in resultados)
        presentes = set(self.tok(texto_top))
        encontrados = [t for t in terminos if _presente(t, presentes)]
        mejor = resultados[0].puntaje if resultados else 0.0
        cobertura = len(encontrados) / len(terminos)
        respaldo = mejor >= UMBRAL_SIMILITUD and cobertura >= UMBRAL_COBERTURA
        motivo = "" if respaldo else (
            f"Similitud máxima {mejor:.2f} y {len(encontrados)}/{len(terminos)} "
            f"términos de la pregunta presentes en el documento.")
        # Solo se devuelven fragmentos razonablemente relacionados.
        resultados = [r for r in resultados if r.puntaje >= mejor * 0.35]
        return Busqueda(resultados, [self.tok.forma(t) for t in terminos],
                        [self.tok.forma(t) for t in encontrados], respaldo, motivo)

    def mejores_oraciones(self, consulta: str, fragmentos: list[Document],
                          n: int = 3) -> list[tuple[str, str]]:
        """Oraciones más parecidas a la consulta, con la etiqueta de su página."""
        candidatas = [(o, f.metadata["etiqueta"]) for f in fragmentos
                      for o in oraciones(f.page_content)]
        if not candidatas:
            return []
        q = self.vectorizador.transformar([consulta])
        m = self.vectorizador.transformar([c[0] for c in candidatas])
        sims = (m @ q.T).ravel()
        vistas, salida = set(), []
        for i in np.argsort(-sims):
            if sims[i] <= 0 or candidatas[i][0] in vistas:
                continue
            vistas.add(candidatas[i][0])
            salida.append(candidatas[i])
            if len(salida) == n:
                break
        return salida
