# NoEntendIA

Asistente de estudio con inteligencia artificial que convierte documentos difíciles y
apuntes desordenados en material claro: **organiza por temas, resume, explica con
palabras sencillas y genera preguntas de práctica**. Todo lo que dice trae la **página o
sección** de donde sale, y si el documento no contiene la respuesta, **lo dice en lugar
de inventarla**.

**En línea:** https://noentendia.vercel.app
**Código:** https://github.com/LAX0509/noentendia (privado). Cada commit en `main` se
publica solo en Vercel.

Proyecto final — Análisis de IA. Técnica principal: **RAG (Retrieval-Augmented
Generation) con LangChain**, más **aprendizaje no supervisado (K-Means)** para los temas.

## Cómo ejecutarlo en local

Doble clic en `INICIAR_NOENTENDIA.bat`, o en una terminal:

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m uvicorn api.index:app --port 8000
```

Abre http://localhost:8000 y pulsa «Probar con apuntes de ejemplo» o sube tus PDF/TXT/MD.

**Claves de IA (ocultas):** copia `.env.example` como `.env` y pon hasta 3 claves de Groq
(`GROQ_API_KEY_1`, `_2`, `_3`). Viven solo en el servidor: el navegador nunca las ve
(`/api/estado` solo informa cuántas hay). Si una clave se queda sin cuota (error 429),
se pone en pausa el tiempo que indique Groq y se usa la siguiente; si todas fallan con el
modelo grande (`gpt-oss-120b`) se prueba el pequeño (`gpt-oss-20b`), y si nada responde,
la app sigue en **modo local extractivo**. Sin claves también funciona en modo local.

Pruebas: `.venv\Scripts\python -m pytest -q` (25 pruebas).

## Despliegue en Vercel

Proyecto `noentendia` (equipo nieto10). Está conectado al repositorio de GitHub: cada `git push` a
`main` despliega en producción y los demás ramales generan una vista previa. Las claves se configuran en Vercel →
noentendia → Settings → Environment Variables (`GROQ_API_KEY_1..3`), nunca en el código.
Publicar una versión nueva:

```bash
npx vercel deploy --prod
```

`.vercelignore` excluye `.env`, pruebas y entregables. Vercel detecta FastAPI en
`api/index.py` y sirve la interfaz desde `public/`.

## Arquitectura

```
Navegador (public/)                         Servidor sin estado (api/index.py, FastAPI)
 pdf.js extrae el texto por página ──JSON──► NoEntendIA.desde_paginas()
 (el PDF nunca sale del navegador)              │
                                                ▼
                         documento.py  ── RecursiveCharacterTextSplitter (LangChain):
                                          fragmentos que conservan su página
                                                ▼
                         recuperador.py ── TF-IDF + coseno (ml.py, numpy)
                                           └─ COMPUERTA: similitud < 0.08 o cobertura de
                                              términos < 60 % ⇒ "No está en tu documento"
                                              (no se llama al modelo)
                                                ▼
                         asistente.py  ── prompt con reglas + fragmentos etiquetados
                                           └─ LLM Groq vía LangChain (rotación de claves)
                                                ▼
                         verificar_citas ─ cita que no corresponde a un fragmento
                                           entregado ⇒ se elimina y se avisa; sin citas
                                           válidas ⇒ respuesta extractiva textual
temas.py ── K-Means (k-means++) sobre TF-IDF; K por coeficiente de silueta; mide el tiempo
```

| Archivo | Qué hace |
|---|---|
| `public/` | Interfaz: subir, temas, conversación, citas clicables, práctica |
| `api/index.py` | API: `/api/estado`, `/api/analizar`, `/api/preguntar`, `/api/resumir`, `/api/practicar` |
| `noentendia/ml.py` | TF-IDF, K-Means y silueta implementados con numpy |
| `noentendia/texto.py` | Normalización en español, stopwords, raíz ligera, oraciones |
| `noentendia/documento.py` | Carga PDF por página y apuntes por sección; fragmenta sin perder la página |
| `noentendia/recuperador.py` | Índice TF-IDF, búsqueda y compuerta «no está en el documento» |
| `noentendia/temas.py` | Agrupamiento K-Means + silueta + tiempo |
| `noentendia/llm.py` | Cliente Groq con LangChain, varias claves y respaldo entre modelos |
| `noentendia/asistente.py` | Preguntar, explicar, resumir, generar preguntas y verificar citas |
| `tests/` | 25 pruebas (citas, negativas, citas inventadas, rotación de claves, API) |

## Decisiones de diseño (para la sustentación)

- **Servidor sin estado y privacidad:** el PDF se lee en el navegador; al servidor solo
  viaja el texto y no se guarda. Reconstruir el índice en cada petición tarda milisegundos.
- **¿Por qué TF-IDF y no embeddings?** Corre sin GPU ni internet, es explicable y permite
  un umbral claro para negarse a responder. Límite: no entiende sinónimos.
- **¿Por qué implementar TF-IDF y K-Means con numpy?** scikit-learn + scipy superan el
  límite de tamaño de una función de Vercel; además cada paso queda visible.
- **Doble barrera contra alucinaciones:** la compuerta decide *antes* del LLM; la
  verificación de citas revisa *después* que cada cita exista entre los fragmentos.
- **Ejemplos inventados, marcados:** las analogías que no están en el documento van bajo
  «Ejemplo ilustrativo (no viene del documento)».
- **Limitaciones:** PDF escaneados necesitan OCR; fórmulas e imágenes no se leen; máximo
  ~600 páginas por carga; una cita correcta prueba que la página fue consultada, no que la
  paráfrasis sea perfecta (por eso se muestra el texto original).
