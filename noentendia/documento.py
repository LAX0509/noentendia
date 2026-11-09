"""Carga de documentos del estudiante y división en fragmentos citables.

Cada fragmento conserva de dónde salió (archivo + página o sección) para que
toda respuesta pueda señalar su origen exacto.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from .texto import limpiar


@dataclass
class Carga:
    documentos: list[Document]          # una entrada por página / sección
    paginas_sin_texto: list[str]        # p. ej. páginas escaneadas (sin OCR)


def _etiqueta(nombre: str, numero: int, es_pdf: bool) -> str:
    # Con varios archivos, el asistente antepone "Doc N" para no confundirlos.
    return f"p. {numero}" if es_pdf else f"sección {numero}"


def cargar_pdf(datos: bytes, nombre: str) -> Carga:
    lector = PdfReader(io.BytesIO(datos))
    docs, vacias = [], []
    for i, pagina in enumerate(lector.pages, start=1):
        texto = limpiar(pagina.extract_text() or "")
        if len(texto) < 20:
            vacias.append(f"{nombre} p. {i}")
            continue
        docs.append(Document(page_content=texto, metadata={
            "fuente": nombre, "pagina": i, "etiqueta": _etiqueta(nombre, i, True)}))
    return Carga(docs, vacias)


def cargar_texto(datos: bytes, nombre: str) -> Carga:
    """Apuntes en .txt/.md: no tienen páginas, se numeran por secciones."""
    texto = limpiar(datos.decode("utf-8", errors="replace"))
    # Secciones: encabezados markdown o bloques separados por líneas en blanco,
    # agrupados hasta ~1500 caracteres.
    bloques = re.split(r"\n(?=#{1,6}\s)|\n\s*\n", texto)
    secciones, actual = [], ""
    for b in bloques:
        b = b.strip()
        if not b:
            continue
        if actual and (len(actual) + len(b) > 1500 or b.startswith("#")):
            secciones.append(actual)
            actual = b
        else:
            actual = f"{actual}\n\n{b}" if actual else b
    if actual:
        secciones.append(actual)
    docs = [Document(page_content=s, metadata={
        "fuente": nombre, "pagina": i, "etiqueta": _etiqueta(nombre, i, False)})
        for i, s in enumerate(secciones, start=1)]
    return Carga(docs, [])


def cargar(datos: bytes, nombre: str) -> Carga:
    if nombre.lower().endswith(".pdf"):
        return cargar_pdf(datos, nombre)
    if nombre.lower().endswith((".txt", ".md")):
        return cargar_texto(datos, nombre)
    raise ValueError(f"Formato no soportado: {nombre} (usa PDF, TXT o MD)")


def fragmentar(documentos: list[Document], tamano: int = 500,
               solape: int = 100) -> list[Document]:
    """Divide cada página en fragmentos sin perder la página de origen."""
    divisor = RecursiveCharacterTextSplitter(
        chunk_size=tamano, chunk_overlap=solape,
        separators=["\n\n", "\n", ". ", "; ", ", ", " "])
    fragmentos = divisor.split_documents(documentos)
    for i, f in enumerate(fragmentos):
        f.metadata["id"] = i
    return fragmentos
