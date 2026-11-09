"""NoEntendIA: asistente de estudio fundamentado en el documento del estudiante.

Principio central: toda explicación, resumen o pregunta debe poder rastrearse
hasta una página (o sección) del material subido. Si el documento no contiene
la respuesta, el asistente lo dice en lugar de inventarla.

Flujo de una pregunta (RAG con verificación):
  1. Recuperar fragmentos relevantes (TF-IDF + coseno).
  2. Compuerta de respaldo: si la similitud o la cobertura de términos es baja,
     responder "no está en el documento" SIN llamar al modelo.
  3. Generar la respuesta con el LLM usando solo esos fragmentos.
  4. Verificar las citas: las que no correspondan a fragmentos entregados se
     marcan como no verificadas; si no queda ninguna válida, se responde en
     modo extractivo (oraciones textuales del documento).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from langchain_core.documents import Document

from .documento import Carga, cargar, fragmentar
from .llm import ClienteLLM, SinLLM
from .recuperador import Indice
from .temas import Agrupamiento, Tema, agrupar
from .texto import Tokenizador, limpiar, quitar_tildes

NO_ESTA = "NO_ESTA_EN_DOCUMENTO"
MENSAJE_NO_ESTA = ("No encontré esa información en tu documento, así que no voy a "
                   "inventarla. Revisa si el tema está en otro material o reformula "
                   "la pregunta con las palabras que usa el texto.")
MARCA_EJEMPLO = "Ejemplo ilustrativo (no viene del documento)"

_ETIQUETA = re.compile(r"(?:Doc\s*\d+\s+)?(?:p\.|secci[oó]n)\s*\d+", re.I)
_CORCHETES = re.compile(r"\[([^\[\]]{1,80})\]")

REGLAS = f"""Eres NoEntendIA, un tutor de estudio para estudiantes universitarios.
Reglas obligatorias:
1. Usa ÚNICAMENTE la información de los FRAGMENTOS del documento del estudiante.
   No agregues datos, cifras, fechas ni definiciones que no estén en ellos.
2. Después de cada idea escribe su cita entre corchetes con la etiqueta exacta
   del fragmento, por ejemplo [p. 3] o [Doc 2 p. 5]. Nunca inventes etiquetas.
3. Si los fragmentos no contienen la respuesta, responde solo: {NO_ESTA}
4. Escribe en español claro y sencillo, en formato Markdown breve. No empieces
   con títulos como "Respuesta:" o "Resumen:" (la interfaz ya los muestra).
5. Si das una analogía o ejemplo cotidiano que NO está en el documento, ponlo
   aparte con el título "{MARCA_EJEMPLO}:" y no le pongas cita."""


@dataclass
class Respuesta:
    texto: str
    encontrado: bool
    fuentes: list[Document] = field(default_factory=list)
    modo: str = "local"
    avisos: list[str] = field(default_factory=list)


@dataclass
class Pregunta:
    pregunta: str
    respuesta: str
    cita: str
    tema: str = ""


def _bloque_fragmentos(fragmentos: list[Document], limite: int = 9000) -> str:
    partes, total = [], 0
    for f in fragmentos:
        p = f"[{f.metadata['etiqueta']}]\n{f.page_content}"
        if total + len(p) > limite:
            break
        partes.append(p)
        total += len(p)
    return "\n\n---\n\n".join(partes)


def _norm_etiqueta(e: str) -> str:
    e = quitar_tildes(re.sub(r"\s+", " ", e.strip().lower()))
    return re.sub(r"p\.\s*", "p. ", e)


def verificar_citas(texto: str, permitidas: list[str]) -> tuple[str, list[str], int]:
    """Marca citas inventadas. Devuelve (texto, avisos, n_citas_validas)."""
    validas = {_norm_etiqueta(p) for p in permitidas}
    avisos, n_validas = [], 0
    # Algunos modelos (gpt-oss) citan con corchetes de ancho completo: 【p. 4】 o ［p. 4］.
    texto = texto.translate(str.maketrans({"【": "[", "】": "]", "［": "[", "］": "]"}))

    def revisar(m: re.Match) -> str:
        nonlocal n_validas
        etiquetas = _ETIQUETA.findall(m.group(1))
        if not etiquetas:
            return m.group(0)
        buenas = [e for e in etiquetas if _norm_etiqueta(e) in validas]
        malas = [e for e in etiquetas if _norm_etiqueta(e) not in validas]
        n_validas += len(buenas)
        if malas:
            avisos.append("Se eliminó una cita que no corresponde a los fragmentos "
                          f"consultados: {', '.join(malas)}.")
        return f"[{', '.join(buenas)}]" if buenas else "[cita no verificada]"

    return _CORCHETES.sub(revisar, texto), avisos, n_validas


class NoEntendIA:
    def __init__(self, archivos: list[tuple[str, bytes]],
                 llm: ClienteLLM | None = None) -> None:
        self.llm = llm
        self.avisos_carga: list[str] = []
        paginas: list[Document] = []
        varios = len(archivos) > 1
        self.archivos = []
        for n, (nombre, datos) in enumerate(archivos, start=1):
            carga: Carga = cargar(datos, nombre)
            if carga.paginas_sin_texto:
                self.avisos_carga.append(
                    "Sin texto extraíble (¿escaneadas?): " + ", ".join(carga.paginas_sin_texto))
            for d in carga.documentos:
                d.metadata["doc"] = n
                if varios:
                    d.metadata["etiqueta"] = f"Doc {n} {d.metadata['etiqueta']}"
            paginas += carga.documentos
            self.archivos.append((n, nombre, len(carga.documentos)))
        self._preparar(paginas)

    def _preparar(self, paginas: list[Document]) -> None:
        if not paginas:
            raise ValueError("No se pudo extraer texto de los archivos. Si es un PDF "
                             "escaneado, pásalo primero por OCR.")
        self.paginas = paginas
        self.fragmentos = fragmentar(paginas)
        self.indice = Indice(self.fragmentos)
        self._temas: Agrupamiento | None = None

    @classmethod
    def desde_paginas(cls, paginas: list[dict], llm: ClienteLLM | None = None) -> "NoEntendIA":
        """Reconstruye el asistente a partir del texto ya extraído (p. ej. en el navegador).

        Cada página: {"doc": 1, "nombre": "x.pdf", "pagina": 3, "texto": "...",
        "tipo": "pdf" | "texto"}. La fragmentación es determinista, así que los ids
        de fragmento coinciden entre peticiones.
        """
        yo = cls.__new__(cls)
        yo.llm, yo.avisos_carga = llm, []
        varios = len({p["doc"] for p in paginas}) > 1
        docs, archivos = [], {}
        for p in paginas:
            texto = limpiar(str(p.get("texto", "")))
            if len(texto) < 20:
                continue
            n, num = int(p["doc"]), int(p["pagina"])
            etiqueta = f"p. {num}" if p.get("tipo", "pdf") == "pdf" else f"sección {num}"
            if varios:
                etiqueta = f"Doc {n} {etiqueta}"
            docs.append(Document(page_content=texto, metadata={
                "fuente": str(p.get("nombre", f"Doc {n}")), "pagina": num,
                "doc": n, "etiqueta": etiqueta}))
            archivos.setdefault(n, [str(p.get("nombre", "")), 0])[1] += 1
        yo.archivos = [(n, nom, c) for n, (nom, c) in sorted(archivos.items())]
        yo._preparar(docs)
        return yo

    def tema_desde(self, numero: int, ids: list[int], palabras: list[str],
                   titulo: str = "") -> Tema:
        """Reconstruye un tema a partir de los ids de sus fragmentos."""
        validos = [self.fragmentos[i] for i in ids if 0 <= i < len(self.fragmentos)]
        if not validos:
            raise ValueError("El tema no coincide con el documento cargado.")
        return Tema(numero, palabras, validos, titulo)

    def _paginas(self, fragmentos: list[Document]) -> list[Document]:
        """Páginas completas de esos fragmentos (evita oraciones cortadas por el solape)."""
        claves = list(dict.fromkeys((f.metadata["doc"], f.metadata["pagina"]) for f in fragmentos))
        por_clave = {(p.metadata["doc"], p.metadata["pagina"]): p for p in self.paginas}
        return [por_clave[c] for c in claves if c in por_clave]

    # ------------------------------------------------------------------ temas
    @property
    def modo(self) -> str:
        return "IA generativa (Groq)" if self.llm else "local (extractivo)"

    def temas(self) -> Agrupamiento:
        if self._temas is None:
            self._temas = agrupar(self.indice)
            self._titular_temas(self._temas.temas)
        return self._temas

    def _titular_temas(self, temas: list[Tema]) -> None:
        if not self.llm or len(temas) < 2:
            return
        listado = "\n".join(
            f"{t.numero}. palabras clave: {', '.join(t.palabras)} | inicio: "
            f"{t.fragmentos[0].page_content[:200]!r}" for t in temas)
        try:
            salida = self.llm(
                "Devuelves solo JSON válido.",
                "Para cada tema de unos apuntes, usando solo sus palabras clave y su "
                "inicio, escribe: un título corto (máximo 6 palabras, en español) y una "
                "pregunta breve que un estudiante haría y que ese texto sí responde. "
                'Responde como {"1": {"titulo": "...", "pregunta": "..."}, ...}.\n\n'
                + listado)
            datos = json.loads(_extraer_json(salida))
            for t in temas:
                d = datos.get(str(t.numero)) or {}
                if isinstance(d, str):
                    d = {"titulo": d}
                t.titulo = str(d.get("titulo", "")).strip()[:60]
                t.sugerencia = str(d.get("pregunta", "")).strip()[:120]
        except (SinLLM, ValueError, AttributeError):
            pass  # se conservan los nombres por palabras clave

    # -------------------------------------------------------------- preguntar
    def preguntar(self, pregunta: str, explicar: bool = False) -> Respuesta:
        busqueda = self.indice.buscar(pregunta, k=5)
        fuentes = [r.fragmento for r in busqueda.resultados]
        if not busqueda.hay_respaldo:
            return Respuesta(MENSAJE_NO_ESTA, False, fuentes[:2], "compuerta",
                             [busqueda.motivo] if busqueda.motivo else [])
        if self.llm:
            instruccion = (
                "Explica el concepto como a alguien que lo ve por primera vez: "
                "1) idea principal en una frase, 2) explicación paso a paso con "
                f"palabras sencillas, 3) un ejemplo (del documento si lo hay; si no, "
                f"usa la sección '{MARCA_EJEMPLO}')." if explicar else
                "Responde la pregunta de forma directa y breve.")
            try:
                texto = self.llm(REGLAS, f"{instruccion}\n\nPREGUNTA: {pregunta}\n\n"
                                         f"FRAGMENTOS:\n{_bloque_fragmentos(fuentes)}")
                return self._validar(texto, fuentes, pregunta)
            except SinLLM as e:
                r = self._extractiva(pregunta, fuentes)
                r.avisos.append(f"El modelo no respondió ({e}); se usó el modo local.")
                return r
        return self._extractiva(pregunta, fuentes)

    def _validar(self, texto: str, fuentes: list[Document], pregunta: str) -> Respuesta:
        if NO_ESTA in texto:
            return Respuesta(MENSAJE_NO_ESTA, False, fuentes[:2], "IA generativa",
                             ["El modelo no halló la respuesta en los fragmentos."])
        etiquetas = [f.metadata["etiqueta"] for f in fuentes]
        texto, avisos, n_validas = verificar_citas(texto, etiquetas)
        if n_validas == 0:
            r = self._extractiva(pregunta, fuentes)
            r.avisos = ["La respuesta generada no traía citas verificables; se "
                        "muestran las oraciones textuales del documento."]
            return r
        citadas = {_norm_etiqueta(e) for e in _ETIQUETA.findall(texto)}
        usadas = [f for f in fuentes if _norm_etiqueta(f.metadata["etiqueta"]) in citadas]
        return Respuesta(texto, True, usadas or fuentes, "IA generativa", avisos)

    def _extractiva(self, consulta: str, fuentes: list[Document]) -> Respuesta:
        oraciones = self.indice.mejores_oraciones(consulta, self._paginas(fuentes), n=3)
        if not oraciones:
            return Respuesta(MENSAJE_NO_ESTA, False, [], "local")
        texto = "Esto dice tu documento:\n\n" + "\n".join(
            f"- {o} [{e}]" for o, e in oraciones)
        return Respuesta(texto, True, fuentes, "local (extractivo)")

    # ---------------------------------------------------------------- resumir
    def resumir(self, tema: Tema) -> Respuesta:
        if self.llm:
            try:
                texto = self.llm(REGLAS, (
                    "Resume las ideas principales de este tema en 3 a 6 viñetas "
                    "cortas. Cada viñeta termina con su cita.\n\nTEMA: "
                    f"{tema.nombre}\n\nFRAGMENTOS:\n{_bloque_fragmentos(tema.fragmentos)}"))
                return self._validar(texto, tema.fragmentos, " ".join(tema.palabras))
            except SinLLM as e:
                r = self._resumen_local(tema)
                r.avisos.append(f"El modelo no respondió ({e}); resumen local.")
                return r
        return self._resumen_local(tema)

    def _resumen_local(self, tema: Tema) -> Respuesta:
        oraciones = self.indice.mejores_oraciones(" ".join(tema.palabras),
                                                  self._paginas(tema.fragmentos), n=4)
        texto = "\n".join(f"- {o} [{e}]" for o, e in oraciones)
        return Respuesta(texto, bool(oraciones), tema.fragmentos, "local (extractivo)")

    # ------------------------------------------------------------- practicar
    def generar_preguntas(self, tema: Tema, n: int = 4) -> tuple[list[Pregunta], list[str]]:
        avisos: list[str] = []
        if self.llm:
            try:
                salida = self.llm(REGLAS + "\nDevuelves solo JSON válido.", (
                    f"Crea {n} preguntas de práctica (comprensión, no memoria literal) "
                    "con su respuesta breve, basadas solo en los fragmentos. Formato: "
                    '[{"pregunta": "...", "respuesta": "...", "cita": "p. N"}]'
                    f"\n\nFRAGMENTOS:\n{_bloque_fragmentos(tema.fragmentos)}"))
                datos = json.loads(_extraer_json(salida))
                validas = {_norm_etiqueta(e): e for e in tema.etiquetas}
                preguntas = []
                for d in datos:
                    cita = _norm_etiqueta(str(d.get("cita", "")).strip("[] "))
                    if cita in validas and d.get("pregunta") and d.get("respuesta"):
                        preguntas.append(Pregunta(str(d["pregunta"]), str(d["respuesta"]),
                                                  validas[cita], tema.nombre))
                descartadas = len(datos) - len(preguntas)
                if descartadas:
                    avisos.append(f"Se descartaron {descartadas} preguntas con citas "
                                  "que no se pudieron verificar.")
                if preguntas:
                    return preguntas[:n], avisos
            except (SinLLM, ValueError, AttributeError, TypeError) as e:
                avisos.append(f"No se pudieron generar con IA ({type(e).__name__}); "
                              "se usan preguntas de completar.")
        return self._preguntas_locales(tema, n), avisos

    def _preguntas_locales(self, tema: Tema, n: int) -> list[Pregunta]:
        """Preguntas de completar: se oculta un término clave de una oración."""
        tok = Tokenizador()
        claves = set(self.indice.tok(" ".join(tema.palabras)))
        preguntas, usadas = [], set()
        for oracion, etiqueta in self.indice.mejores_oraciones(
                " ".join(tema.palabras), self._paginas(tema.fragmentos), n=n * 3):
            palabras = re.findall(r"[A-Za-zÁÉÍÓÚáéíóúÑñÜü]{4,}", oracion)
            candidatas = [p for p in palabras if tok(p) and tok(p)[0] in claves]
            if not candidatas:
                continue
            objetivo = max(candidatas, key=len)
            if objetivo.lower() in usadas:
                continue
            usadas.add(objetivo.lower())
            hueco = re.sub(rf"\b{re.escape(objetivo)}\b", "_____", oracion, count=1)
            preguntas.append(Pregunta(f"Completa: {hueco}", objetivo, etiqueta, tema.nombre))
            if len(preguntas) == n:
                break
        return preguntas


def _extraer_json(texto: str) -> str:
    m = re.search(r"(\[.*\]|\{.*\})", texto, re.S)
    if not m:
        raise ValueError("La respuesta no contiene JSON.")
    return m.group(1)
