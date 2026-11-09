"""Conexión con el modelo de lenguaje (la "G" de RAG) mediante LangChain.

Las claves de Groq viven SOLO en el servidor (variables de entorno o .env); el
navegador nunca las ve. Se admiten varias claves:

    GROQ_API_KEY_1, GROQ_API_KEY_2, GROQ_API_KEY_3   (o GROQ_API_KEYS=k1,k2,k3)

Si una clave se queda sin cuota (error 429) se "enfría" un rato y se usa la
siguiente; si todas fallan con el modelo grande, se prueba el modelo pequeño.
Si nada responde, NoEntendIA sigue en "modo local" (extractivo).
"""
from __future__ import annotations

import os
import re
import time

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

MODELOS_GROQ = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
ENFRIAMIENTO_S = 60          # pausa por defecto para una clave sin cuota
_enfriadas: dict[str, float] = {}   # "clave#modelo" -> instante en que vuelve a estar lista


class SinLLM(RuntimeError):
    pass


def leer_claves() -> list[str]:
    """Claves de Groq configuradas en el servidor, sin duplicados."""
    claves = [c for c in os.getenv("GROQ_API_KEYS", "").split(",")]
    claves.append(os.getenv("GROQ_API_KEY", ""))
    claves += [os.getenv(f"GROQ_API_KEY_{i}", "") for i in range(1, 10)]
    return list(dict.fromkeys(c.strip() for c in claves if c.strip()))


def _es_sin_cuota(e: Exception) -> bool:
    texto = f"{type(e).__name__} {e}".lower()
    return "ratelimit" in texto or "429" in texto or "rate limit" in texto


def _espera_sugerida(e: Exception) -> float:
    m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", str(e))
    if not m:
        return ENFRIAMIENTO_S
    return min(600.0, float(m.group(1) or 0) * 60 + float(m.group(2)) + 1)


class ClienteLLM:
    """Prueba (modelo, clave) en orden y salta las que estén sin cuota."""

    def __init__(self, modelos: list[BaseChatModel], nombres: list[str]) -> None:
        self.modelos, self.nombres = modelos, nombres
        self.ultimo_modelo = ""

    def __call__(self, sistema: str, usuario: str) -> str:
        errores, ahora = [], time.time()
        mensajes = [SystemMessage(sistema), HumanMessage(usuario)]
        for modelo, nombre in zip(self.modelos, self.nombres):
            if _enfriadas.get(nombre, 0) > ahora:
                errores.append(f"{nombre.split('#')[0]}: en pausa")
                continue
            try:
                r = modelo.invoke(mensajes)
                self.ultimo_modelo = nombre.split("#")[0]
                return str(r.content).strip()
            except Exception as e:  # cuota, red, modelo retirado...
                if _es_sin_cuota(e):
                    _enfriadas[nombre] = time.time() + _espera_sugerida(e)
                errores.append(f"{nombre.split('#')[0]}: {type(e).__name__}")
        raise SinLLM("; ".join(dict.fromkeys(errores)))


def crear_cliente(claves: list[str] | None = None) -> ClienteLLM | None:
    claves = claves if claves is not None else leer_claves()
    if not claves:
        return None
    from langchain_groq import ChatGroq

    modelos, nombres = [], []
    for m in MODELOS_GROQ:                     # primero el modelo grande con todas las claves
        for i, clave in enumerate(claves, start=1):
            modelos.append(ChatGroq(model=m, api_key=clave, temperature=0.2,
                                    timeout=40, max_retries=0))
            nombres.append(f"{m}#clave{i}")
    return ClienteLLM(modelos, nombres)
