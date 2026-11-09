"""Pruebas de NoEntendIA. Ejecutar con:  python -m pytest -q"""
import sys
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "ejemplos"))

from crear_pdf_ejemplo import crear  # noqa: E402
from noentendia import MENSAJE_NO_ESTA, NoEntendIA, verificar_citas  # noqa: E402
from noentendia.llm import ClienteLLM  # noqa: E402


@pytest.fixture(scope="module")
def pdf(tmp_path_factory) -> bytes:
    return crear(tmp_path_factory.mktemp("pdf") / "apuntes.pdf").read_bytes()


@pytest.fixture(scope="module")
def asistente(pdf) -> NoEntendIA:
    return NoEntendIA([("apuntes.pdf", pdf)])


def falso_llm(*respuestas: str) -> ClienteLLM:
    return ClienteLLM([FakeListChatModel(responses=list(respuestas))], ["falso"])


def test_carga_por_paginas(asistente):
    assert [d.metadata["pagina"] for d in asistente.paginas] == [1, 2, 3, 4, 5]
    assert all("etiqueta" in f.metadata for f in asistente.fragmentos)


@pytest.mark.parametrize("pregunta, pagina", [
    ("¿Qué es un motor de inferencia?", "p. 1"),
    ("¿Para qué sirve Protégé?", "p. 2"),
    ("¿Qué mide la matriz de confusión?", "p. 3"),
    ("¿Cómo funciona K-Means?", "p. 4"),
    ("¿Qué son las tortugas en NetLogo?", "p. 5"),
])
def test_cita_la_pagina_correcta(asistente, pregunta, pagina):
    r = asistente.preguntar(pregunta)
    assert r.encontrado
    assert f"[{pagina}]" in r.texto


@pytest.mark.parametrize("pregunta", [
    "¿Quién ganó el mundial de fútbol de 2022?",
    "¿Cuál es la receta de la bandeja paisa?",
    "Explícame la fotosíntesis de las plantas",
    "¿Quién inventó Prolog?",   # el tema sí está, pero ese dato no
])
def test_dice_que_no_esta(asistente, pregunta):
    r = asistente.preguntar(pregunta)
    assert not r.encontrado
    assert r.texto == MENSAJE_NO_ESTA


def test_temas_agrupa_y_cubre_todo(asistente):
    ag = asistente.temas()
    assert ag.k >= 2 and ag.silueta is not None
    assert sum(len(t.fragmentos) for t in ag.temas) == len(asistente.fragmentos)


def test_resumen_y_preguntas_locales_citan_paginas_reales(asistente):
    validas = {d.metadata["etiqueta"] for d in asistente.paginas}
    for tema in asistente.temas().temas:
        assert asistente.resumir(tema).encontrado
        preguntas, _ = asistente.generar_preguntas(tema, 2)
        assert preguntas
        assert all(p.cita in validas and "_____" in p.pregunta for p in preguntas)


def test_verificar_citas_elimina_inventadas():
    texto, avisos, n = verificar_citas("A [p. 1]. B [p. 9]. C [p. 1, p. 7].", ["p. 1"])
    assert texto == "A [p. 1]. B [cita no verificada]. C [p. 1]."
    assert n == 2 and len(avisos) == 2


def test_llm_con_cita_inventada_se_corrige(pdf):
    a = NoEntendIA([("apuntes.pdf", pdf)],
                   llm=falso_llm("K-Means agrupa datos [p. 4]. Lo inventó Turing [p. 99]."))
    r = a.preguntar("¿Cómo funciona K-Means?")
    assert r.encontrado and "[p. 4]" in r.texto and "p. 99" not in r.texto
    assert r.avisos


def test_llm_sin_citas_vuelve_a_modo_extractivo(pdf):
    a = NoEntendIA([("apuntes.pdf", pdf)], llm=falso_llm("K-Means es genial."))
    r = a.preguntar("¿Cómo funciona K-Means?")
    assert r.modo.startswith("local") and "[p. 4]" in r.texto


def test_llm_dice_no_esta(pdf):
    a = NoEntendIA([("apuntes.pdf", pdf)], llm=falso_llm("NO_ESTA_EN_DOCUMENTO"))
    assert not a.preguntar("¿Qué es el backtracking en Prolog?").encontrado


def test_compuerta_no_llama_al_modelo(pdf):
    class Espia(ClienteLLM):
        llamadas = 0

        def __call__(self, s, u):
            Espia.llamadas += 1
            return "x"

    a = NoEntendIA([("apuntes.pdf", pdf)], llm=Espia([], []))
    a.preguntar("¿Quién ganó el mundial de 2022?")
    assert Espia.llamadas == 0


def test_preguntas_ia_descartan_citas_falsas(pdf):
    a = NoEntendIA([("apuntes.pdf", pdf)])
    tema = a.temas().temas[0]
    cita = tema.etiquetas[0]
    a.llm = falso_llm(
        f'[{{"pregunta": "¿Q1?", "respuesta": "R1", "cita": "{cita}"}},'
        ' {"pregunta": "¿Q2?", "respuesta": "R2", "cita": "p. 42"}]')
    preguntas, avisos = a.generar_preguntas(tema, 4)
    assert [p.pregunta for p in preguntas] == ["¿Q1?"] and avisos


def test_varios_archivos_y_apuntes_txt(pdf):
    notas = "# Redes neuronales\n\nUna neurona artificial suma sus entradas ponderadas " \
            "y aplica una función de activación.".encode()
    a = NoEntendIA([("apuntes.pdf", pdf), ("notas.md", notas)])
    r = a.preguntar("¿Qué hace una neurona artificial?")
    assert r.encontrado and "[Doc 2 sección 1]" in r.texto


# ------------------------------------------------------- rotación de claves
def test_rota_a_la_siguiente_clave_si_una_se_queda_sin_cuota():
    from noentendia import llm as modulo

    class SinCuota(FakeListChatModel):
        def invoke(self, *a, **k):
            raise RuntimeError("Error code: 429 - rate limit. Please try again in 7.5s")

    modulo._enfriadas.clear()
    cliente = ClienteLLM([SinCuota(responses=["x"]), FakeListChatModel(responses=["ok"])],
                         ["m#clave1", "m#clave2"])
    assert cliente("s", "u") == "ok"
    assert modulo._enfriadas["m#clave1"] > 0          # la clave 1 queda en pausa
    assert cliente.ultimo_modelo == "m"


def test_leer_claves_sin_duplicados(monkeypatch):
    from noentendia.llm import leer_claves
    for i in range(1, 10):
        monkeypatch.delenv(f"GROQ_API_KEY_{i}", raising=False)
    monkeypatch.setenv("GROQ_API_KEYS", "a, b")
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("GROQ_API_KEY_1", "b")
    monkeypatch.setenv("GROQ_API_KEY_2", "c")
    assert leer_claves() == ["a", "b", "c"]


# --------------------------------------------------------------- API web
@pytest.fixture()
def cliente_api(monkeypatch):
    for v in ["GROQ_API_KEYS", "GROQ_API_KEY"] + [f"GROQ_API_KEY_{i}" for i in range(1, 10)]:
        monkeypatch.setenv(v, "")          # las pruebas de API corren en modo local
    from fastapi.testclient import TestClient
    from api.index import app
    return TestClient(app)


@pytest.fixture(scope="module")
def paginas(pdf):
    from pypdf import PdfReader
    import io
    return [{"doc": 1, "nombre": "apuntes.pdf", "pagina": i + 1, "texto": p.extract_text()}
            for i, p in enumerate(PdfReader(io.BytesIO(pdf)).pages)]


def test_api_flujo_completo(cliente_api, paginas):
    assert cliente_api.get("/api/estado").json()["ia_generativa"] is False
    a = cliente_api.post("/api/analizar", json={"paginas": paginas}).json()
    assert a["k"] >= 2 and a["temas"][0]["ids"]
    r = cliente_api.post("/api/preguntar", json={"paginas": paginas,
                                                 "pregunta": "¿Qué es el backtracking?"}).json()
    assert r["encontrado"] and "[p. 1]" in r["texto"] and r["fuentes"]
    r = cliente_api.post("/api/preguntar", json={"paginas": paginas,
                                                 "pregunta": "¿Quién inventó Prolog?"}).json()
    assert not r["encontrado"]
    tema = {k: a["temas"][0][k] for k in ("numero", "ids", "palabras", "titulo")}
    assert cliente_api.post("/api/resumir", json={"paginas": paginas, "tema": tema}).json()["encontrado"]
    q = cliente_api.post("/api/practicar", json={"paginas": paginas, "tema": tema}).json()
    assert q["preguntas"] and all(p["cita"].startswith("p. ") for p in q["preguntas"])


def test_api_no_expone_claves(cliente_api, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY_1", "gsk_secreta")
    texto = cliente_api.get("/api/estado").text
    assert "gsk_" not in texto and '"claves":1' in texto


def test_api_sirve_la_interfaz(cliente_api):
    assert "NoEntendIA" in cliente_api.get("/").text
    assert cliente_api.get("/../.env").status_code == 404


def test_acepta_citas_con_corchetes_de_ancho_completo():
    texto, avisos, n = verificar_citas("K-Means agrupa【p. 4】 y usa silueta［p. 4］.", ["p. 4"])
    assert texto == "K-Means agrupa[p. 4] y usa silueta[p. 4]." and n == 2 and not avisos
