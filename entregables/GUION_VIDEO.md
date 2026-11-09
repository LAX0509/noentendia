# NoEntendIA · Guion del video-pitch (máx. 10 min)

Cada integrante explica una sección (requisito). Reemplaza «Integrante N» por los nombres.
Antes de grabar: abre https://noentendia.vercel.app (o local con `INICIAR_NOENTENDIA.bat`), pulsa
«Probar con apuntes de ejemplo» y ten lista la terminal con `.venv\Scripts\python -m pytest -q`.

---

## A. 0:00 – 1:30 · Contexto y problema social — *Integrante 1*

- **Gancho (película):** en *The Orville* y en *Her* la IA responde con seguridad total.
  En la vida real los chatbots también lo hacen… aunque estén inventando.
- **Problema:** los estudiantes reciben PDF largos y apuntes desordenados. Cuando no
  entienden, le preguntan a un chatbot general que puede **alucinar**: dar definiciones,
  fechas o autores que no están en su material, y el estudiante no tiene cómo verificar.
- **Por qué IA:** organizar, resumir y explicar cientos de páginas a la medida de cada
  estudiante no escala con tutores humanos; pero la IA solo sirve si es **verificable**.
- **Propuesta:** NoEntendIA, un tutor que solo usa el documento del estudiante, cita la
  página de cada idea y dice «no está en tu documento» en vez de inventar.

## B. 1:30 – 5:00 · Solución técnica y demo en vivo — *Integrantes 2 y 3*

**Técnica (Integrante 2, ~1:15):** RAG con LangChain + aprendizaje no supervisado.
1. *Carga:* pypdf lee el PDF página por página; el `RecursiveCharacterTextSplitter` de
   LangChain lo divide en fragmentos que **conservan su página**.
2. *Recuperación:* TF-IDF + similitud coseno, implementados por nosotros con numpy
   (`ml.py`). Elegimos TF-IDF por ser explicable, rápido y sin GPU.
3. *Compuerta anti-alucinación:* si la similitud es < 0,08 o la pregunta comparte menos
   del 60 % de sus términos con el documento → «no está», **sin llamar al modelo**.
4. *Generación:* el LLM (Groq, vía LangChain) recibe solo los fragmentos etiquetados y
   reglas estrictas: citar `[p. N]` o responder `NO_ESTA_EN_DOCUMENTO`.
5. *Verificación de citas:* el código revisa cada cita; las que no pertenecen a los
   fragmentos entregados se eliminan y se avisa. Sin citas válidas → respuesta textual.
6. *Temas:* K-Means (con k-means++, también implementado con numpy) sobre TF-IDF; K se elige con el coeficiente de silueta y se mide el
   tiempo (costo de máquina).

**Demo en pantalla (Integrante 3, ~2:15):**
1. Panel **Temas**: los 5 temas detectados; abrir «¿Cómo se organizaron? (K-Means)» para
   mostrar la silueta por K y el tiempo. En un tema → «Resumir» → tocar una cita `p. N`
   para abrir el texto original con el fragmento resaltado.
2. **Pregunta**: «¿Qué es el backtracking?» → respuesta con la cita `p. 1`.
3. Activar **Explícamelo fácil**: «¿Qué es una ontología?» → explicación + ejemplo marcado
   como «no viene del documento» (modo Groq).
4. **El momento clave:** «¿Quién inventó Prolog?» → el tema está, pero ese dato no →
   «No encontré esa información en tu documento». Comparar con un chatbot general.
5. **Practicar** en un tema: preguntas con «Ver respuesta» y su página.
6. Terminal: `pytest -q` → **25 pruebas pasan** (incluye citas inventadas que el sistema
   detecta).

## C. 5:00 – 8:30 · Análisis ético, legal y social — *Integrante 4 (y 1)*

**Principios en riesgo si se desplegara masivamente**
- **Privacidad:** los apuntes pueden traer datos personales (nombres, notas, casos
  clínicos). El PDF se lee en el navegador y el servidor no guarda nada, pero en modo
  IA generativa los fragmentos sí se envían a Groq (un tercero). *Mitigación:* modo
  local sin envío, no almacenar documentos y avisar al usuario.
- **Explicabilidad y transparencia:** cada respuesta muestra su fuente y el fragmento
  original; se informa si respondió el modelo o el modo local.
- **Alucinación residual:** una cita válida prueba que la página existe y fue consultada,
  no que la paráfrasis sea perfecta. Por eso mostramos el texto original para comparar.
- **Sesgo:** NoEntendIA hereda los sesgos del material subido (no los corrige) y TF-IDF
  favorece a quien pregunta con el vocabulario exacto del texto, lo que puede perjudicar a
  estudiantes con menos dominio del lenguaje técnico.
- **Autonomía y pensamiento crítico:** riesgo de sobredependencia (UNESCO lo señala para
  IA generativa en educación). Las preguntas de práctica buscan que el estudiante piense,
  no que copie.
- **Integridad académica:** podría usarse para resumir sin leer; su diseño apunta a
  estudiar, no a producir tareas.

**Marco normativo aplicable**
- **Colombia – Ley 1581 de 2012** (protección de datos personales) y Decreto 1377 de
  2013: autorización previa, finalidad, y tratamiento especial de datos de **niños y
  adolescentes** si lo usan colegios.
- **Colombia – CONPES 4144 de 2025** (Política Nacional de Inteligencia Artificial):
  desarrollo ético y confiable de la IA.
- **Derechos de autor – Ley 23 de 1982:** subir libros completos y enviarlos a un
  proveedor externo puede exceder el uso personal; el sistema no debe redistribuirlos.
- **UNESCO:** Recomendación sobre la Ética de la IA (2021) y Guía para la IA generativa
  en la educación y la investigación (2023): supervisión humana, transparencia, edad
  mínima y protección de datos.
- **Unión Europea – GDPR** (si hay usuarios en la UE) y **Reglamento de IA (2024/1689)**:
  los sistemas que *califican* o deciden admisiones en educación son de **alto riesgo**;
  NoEntendIA, como apoyo de estudio que no evalúa, quedaría fuera de esa categoría, pero sí
  le aplica el deber de transparencia (avisar que se habla con una IA).

**Responsabilidad y equidad**
- **Desarrolladores:** responden por el diseño (compuerta, verificación, avisos claros).
- **Institución que lo implanta:** por el uso, el tratamiento de datos y la supervisión.
- **Estudiante:** por contrastar con la fuente; la herramienta apoya, no reemplaza.
- **Brecha digital:** necesita dispositivo e idealmente internet; el modo local y el bajo
  costo (sin GPU) reducen la brecha, pero no la eliminan.

## D. 8:30 – 10:00 · Conclusiones y pregunta abierta — *Todos*

- **Aprendizajes:** una IA útil en educación no es la que más sabe, sino la que **sabe
  cuándo no sabe**; la verificación debe estar en el código, no solo en el prompt.
- **Lección técnica:** técnicas clásicas (TF-IDF, K-Means) siguen siendo valiosas por
  explicables y baratas, combinadas con un LLM.
- **Pregunta abierta para el debate:**
  > «Si una IA siempre te dice la página exacta de donde sacó la respuesta, ¿vas a abrir
  > esa página para comprobarlo… o la cita se convierte en una nueva forma de confiar
  > ciegamente?»
