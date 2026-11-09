"""API de NoEntendIA (FastAPI). En Vercel corre como función serverless.

Es SIN ESTADO: el navegador extrae el texto del PDF (pdf.js) y lo envía en cada
petición; el servidor reconstruye el índice en milisegundos. Así no se guardan
documentos de los estudiantes en ningún servidor (privacidad por diseño).

Local:  .venv\\Scripts\\python -m uvicorn api.index:app --port 8000
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
load_dotenv(RAIZ / ".env")

from noentendia import NoEntendIA, crear_cliente  # noqa: E402
from noentendia.llm import leer_claves  # noqa: E402

MAX_CARACTERES = 1_500_000
app = FastAPI(title="NoEntendIA", docs_url=None, redoc_url=None)


class Pagina(BaseModel):
    doc: int = 1
    nombre: str = Field("documento", max_length=200)
    pagina: int
    texto: str
    tipo: str = "pdf"


class TemaRef(BaseModel):
    numero: int
    ids: list[int]
    palabras: list[str] = []
    titulo: str = ""


class Peticion(BaseModel):
    paginas: list[Pagina] = Field(..., max_length=2000)


class PeticionPregunta(Peticion):
    pregunta: str = Field(..., min_length=2, max_length=500)
    explicar: bool = False


class PeticionTema(Peticion):
    tema: TemaRef
    cantidad: int = Field(4, ge=1, le=8)


def _asistente(p: Peticion, con_llm: bool = True) -> NoEntendIA:
    total = sum(len(x.texto) for x in p.paginas)
    if total > MAX_CARACTERES:
        raise HTTPException(413, "El material es demasiado grande (máx. ~1,5 millones de "
                                 "caracteres). Divide el PDF en partes.")
    try:
        return NoEntendIA.desde_paginas([x.model_dump() for x in p.paginas],
                                        llm=crear_cliente() if con_llm else None)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


def _respuesta(r) -> dict:
    return {"texto": r.texto, "encontrado": r.encontrado, "modo": r.modo,
            "avisos": r.avisos,
            "fuentes": [{"etiqueta": f.metadata["etiqueta"], "fuente": f.metadata["fuente"],
                         "texto": f.page_content} for f in r.fuentes]}


@app.get("/api/estado")
def estado() -> dict:
    n = len(leer_claves())  # solo cuántas hay; las claves nunca salen del servidor
    return {"ia_generativa": n > 0, "claves": n,
            "modo": "IA generativa (Groq)" if n else "local (extractivo)"}


@app.post("/api/analizar")
def analizar(p: Peticion) -> dict:
    a = _asistente(p)
    ag = a.temas()
    return {
        "archivos": [{"doc": n, "nombre": nom, "paginas": c} for n, nom, c in a.archivos],
        "fragmentos": len(a.fragmentos),
        "k": ag.k, "silueta": ag.silueta, "segundos": ag.segundos,
        "evaluados": ag.evaluados, "modo": a.modo,
        "temas": [{"numero": t.numero, "nombre": t.nombre, "titulo": t.titulo,
                   "sugerencia": t.sugerencia,
                   "palabras": t.palabras, "etiquetas": t.etiquetas,
                   "ids": [f.metadata["id"] for f in t.fragmentos]} for t in ag.temas],
    }


@app.post("/api/preguntar")
def preguntar(p: PeticionPregunta) -> dict:
    return _respuesta(_asistente(p).preguntar(p.pregunta, explicar=p.explicar))


@app.post("/api/resumir")
def resumir(p: PeticionTema) -> dict:
    a = _asistente(p)
    t = p.tema
    return _respuesta(a.resumir(a.tema_desde(t.numero, t.ids, t.palabras, t.titulo)))


@app.post("/api/practicar")
def practicar(p: PeticionTema) -> dict:
    a = _asistente(p)
    t = p.tema
    preguntas, avisos = a.generar_preguntas(
        a.tema_desde(t.numero, t.ids, t.palabras, t.titulo), p.cantidad)
    return {"avisos": avisos, "modo": a.modo,
            "preguntas": [{"pregunta": q.pregunta, "respuesta": q.respuesta,
                           "cita": q.cita} for q in preguntas]}


# En local se sirve también la interfaz (en Vercel la sirve el CDN desde public/).
if not os.getenv("VERCEL"):
    PUBLICO = RAIZ / "public"

    @app.get("/")
    def inicio() -> FileResponse:
        return FileResponse(PUBLICO / "index.html")

    @app.get("/{ruta:path}")
    def estaticos(ruta: str) -> FileResponse:
        destino = (PUBLICO / ruta).resolve()
        if PUBLICO.resolve() not in destino.parents or not destino.is_file():
            raise HTTPException(404)
        return FileResponse(destino)
