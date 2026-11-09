/* NoEntendIA — interfaz web.
 * El PDF se lee AQUÍ (pdf.js); al servidor solo viaja el texto por página.
 * El servidor no guarda nada: cada petición lleva las páginas. */
"use strict";

pdfjsLib.GlobalWorkerOptions.workerSrc =
  "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";

const $ = (s) => document.querySelector(s);
const estado = { paginas: [], analisis: null, ocupado: false };
const MAX_PAGINAS = 600;

/* ------------------------------------------------------------ utilidades */
const esc = (t) => String(t).replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

async function api(ruta, cuerpo) {
  const r = await fetch(ruta, cuerpo ? {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(cuerpo),
  } : undefined);
  if (!r.ok) {
    let msg = `Error ${r.status}`;
    try { const d = await r.json(); msg = typeof d.detail === "string" ? d.detail : msg; } catch {}
    throw new Error(msg);
  }
  return r.json();
}

const ETIQ = "(?:Doc \\d+ )?(?:p\\.|secci[oó]n) ?\\d+";
const RE_CITA = new RegExp(`\\[(${ETIQ}(?:\\s*[,;]\\s*${ETIQ})*)\\]`, "g");

function conCitas(html) {
  return html.replace(RE_CITA, (_, grupo) =>
    grupo.split(/\s*[,;]\s*/).map((e) =>
      `<button type="button" class="cita" data-cita="${esc(e.trim())}">${esc(e.trim())}</button>`).join(" "));
}

function markdown(texto) {
  return conCitas(DOMPurify.sanitize(marked.parse(texto, { breaks: true })));
}

/* ------------------------------------------------------- leer archivos */
async function leerPDF(archivo, doc) {
  const pdf = await pdfjsLib.getDocument({ data: await archivo.arrayBuffer() }).promise;
  const paginas = [];
  for (let i = 1; i <= pdf.numPages; i++) {
    $("#cargando-txt").textContent = `Leyendo ${archivo.name}: página ${i} de ${pdf.numPages}…`;
    const c = await (await pdf.getPage(i)).getTextContent();
    const texto = c.items.map((it) => it.str + (it.hasEOL ? "\n" : "")).join("");
    paginas.push({ doc, nombre: archivo.name, pagina: i, texto, tipo: "pdf" });
  }
  return paginas;
}

async function leerTexto(archivo, doc) {
  // Apuntes sin páginas: se numeran por secciones (encabezados o bloques ~1500 car.).
  const texto = (await archivo.text()).replace(/\r/g, "");
  const bloques = texto.split(/\n(?=#{1,6}\s)|\n\s*\n/);
  const secciones = []; let actual = "";
  for (let b of bloques) {
    b = b.trim(); if (!b) continue;
    if (actual && (actual.length + b.length > 1500 || b.startsWith("#"))) { secciones.push(actual); actual = b; }
    else actual = actual ? `${actual}\n\n${b}` : b;
  }
  if (actual) secciones.push(actual);
  return secciones.map((s, i) => ({ doc, nombre: archivo.name, pagina: i + 1, texto: s, tipo: "texto" }));
}

async function cargarArchivos(archivos) {
  if (!archivos.length || estado.ocupado) return;
  mostrar("cargando");
  try {
    let paginas = [];
    for (const [i, a] of [...archivos].entries()) {
      const n = /\.pdf$/i.test(a.name) ? await leerPDF(a, i + 1)
        : /\.(txt|md)$/i.test(a.name) ? await leerTexto(a, i + 1) : null;
      if (!n) throw new Error(`«${a.name}» no es PDF, TXT ni MD.`);
      paginas = paginas.concat(n);
    }
    if (paginas.length > MAX_PAGINAS) throw new Error(`Son ${paginas.length} páginas; el máximo es ${MAX_PAGINAS}. Divide el documento.`);
    const conTexto = paginas.filter((p) => p.texto.trim().length >= 20);
    if (!conTexto.length) throw new Error("No encontré texto. Si es un PDF escaneado (fotos), primero pásalo por un OCR.");
    $("#cargando-txt").textContent = "Organizando tus temas…";
    estado.paginas = paginas;
    estado.analisis = await api("/api/analizar", { paginas });
    estado.sinTexto = paginas.length - conTexto.length;
    iniciarEstudio();
  } catch (e) {
    mostrar("inicio");
    alert(e.message || e);
  }
}

/* -------------------------------------------------------------- vistas */
function mostrar(cual) {
  $("#pantalla-inicio").hidden = cual !== "inicio";
  $("#cargando").hidden = cual !== "cargando";
  $("#pantalla-estudio").hidden = cual !== "estudio";
}

function iniciarEstudio() {
  const a = estado.analisis;
  $("#docs").innerHTML = a.archivos.map((d) =>
    `<div class="doc">📄 <span title="${esc(d.nombre)}">${esc(d.nombre)}</span><small>${d.paginas} pág.</small></div>`
  ).join("") + `<button class="btn-otro" id="btn-otro">＋ Cambiar de documento</button>`;
  $("#btn-otro").onclick = reiniciar;

  $("#temas").innerHTML = a.temas.map((t, i) => `
    <li class="tema" data-i="${i}">
      <div class="tema-cab"><span class="tema-num">${t.numero}</span>
        <div><div class="tema-nombre">${esc(t.nombre)}</div>
        <div class="tema-pags">${esc(t.etiquetas.join(", "))}</div></div></div>
      <div class="tema-acciones">
        <button data-accion="resumir">📝 Resumir</button>
        <button data-accion="practicar">❓ Practicar</button>
      </div>
    </li>`).join("");
  $("#toggle-txt").textContent = `Temas (${a.temas.length})`;

  const sil = a.silueta == null ? "—" : a.silueta.toFixed(2);
  $("#tecnico").innerHTML = `
    <p>Agrupé ${a.fragmentos} fragmentos con <b>K-Means</b> sobre vectores TF-IDF y elegí el número
    de temas con el <b>coeficiente de silueta</b> (de −1 a 1; más alto = temas mejor separados).</p>
    <table>${Object.entries(a.evaluados).map(([k, s]) =>
      `<tr><td>K = ${k}${+k === a.k ? " ✔" : ""}</td><td>${s.toFixed(3)}</td></tr>`).join("")}
    <tr><td>Silueta elegida</td><td>${sil}</td></tr>
    <tr><td>Tiempo de cómputo</td><td>${(a.segundos * 1000).toFixed(0)} ms</td></tr></table>`;

  const nPag = estado.paginas.length;
  const bienvenida = $("#tpl-bienvenida").content.cloneNode(true);
  bienvenida.querySelector("[data-paginas]").textContent =
    `${nPag} ${nPag === 1 ? "página" : "páginas"}${a.archivos.length > 1 ? ` de ${a.archivos.length} documentos` : ""}`;
  bienvenida.querySelector("[data-ntemas]").textContent =
    `${a.temas.length} ${a.temas.length === 1 ? "tema" : "temas"}`;
  $("#hilo").innerHTML = "";
  $("#hilo").append(bienvenida);
  if (estado.sinTexto) agregarBot(`<div class="avisos">⚠️ ${estado.sinTexto} página(s) sin texto (¿escaneadas?). Esas no las puedo leer.</div>`);

  // Preguntas sugeridas por la IA; sin IA, se propone explicar los temas.
  const sug = a.temas.slice(0, 4).map((t) => t.sugerencia || `Explícame: ${t.nombre}`);
  $("#sugerencias").innerHTML = sug.map((s) => `<button type="button">${esc(s)}</button>`).join("");
  mostrar("estudio");
  $("#pregunta").focus();
}

function reiniciar() {
  estado.paginas = []; estado.analisis = null;
  $("#archivo").value = "";
  mostrar("inicio");
}

/* --------------------------------------------------------------- hilo */
function agregar(html, clase) {
  const div = document.createElement("div");
  div.className = `msg ${clase}`;
  div.innerHTML = clase === "yo" ? `<div class="burbuja">${html}</div>`
    : `<div class="avatar">?</div><div class="burbuja">${html}</div>`;
  $("#hilo").append(div);
  div.scrollIntoView({ behavior: "smooth", block: "start" });
  return div;
}
const agregarBot = (html) => agregar(html, "bot");

function pensando() {
  return agregarBot(`<div class="escribiendo"><i></i><i></i><i></i></div>`);
}

function pintarRespuesta(nodo, r, titulo) {
  const burbuja = nodo.querySelector(".burbuja");
  burbuja.classList.toggle("noesta", !r.encontrado);
  const etiqueta = r.encontrado ? titulo : "No está en tu documento";
  const avisos = (r.avisos || []).map((a) => `⚠️ ${esc(a)}`).join("<br>");
  const fuentes = [...new Set(r.fuentes.map((f) => f.etiqueta))];
  burbuja.innerHTML = `<div class="etiqueta">${esc(etiqueta)}</div>${markdown(r.texto)}
    ${avisos ? `<div class="avisos">${avisos}</div>` : ""}
    <div class="pie">${r.encontrado ? "Fuentes:" : "Lo más cercano:"}
      ${fuentes.map((e) => `<button type="button" class="cita" data-cita="${esc(e)}">${esc(e)}</button>`).join(" ") || "—"}
      <span style="margin-left:auto">${esc(r.modo)}</span></div>`;
  nodo._fuentes = r.fuentes;
}

async function accion(fn, titulo) {
  if (estado.ocupado) return;
  estado.ocupado = true; $("#enviar").disabled = true;
  const nodo = pensando();
  try { await fn(nodo); }
  catch (e) {
    nodo.querySelector(".burbuja").innerHTML =
      `<div class="etiqueta" style="color:var(--ambar)">Algo falló</div>${esc(e.message || e)}. Intenta de nuevo.`;
  } finally { estado.ocupado = false; $("#enviar").disabled = false; }
}

function preguntar(texto) {
  texto = texto.trim();
  if (!texto) return;
  const facil = $("#facil").checked;
  agregar(esc(texto) + (facil ? " <small>· explicado fácil</small>" : ""), "yo");
  $("#pregunta").value = ""; ajustarAltura();
  accion(async (nodo) => {
    const r = await api("/api/preguntar", { paginas: estado.paginas, pregunta: texto, explicar: facil });
    pintarRespuesta(nodo, r, facil ? "Explicación" : "Respuesta");
  });
}

function refTema(t) {
  return { numero: t.numero, ids: t.ids, palabras: t.palabras, titulo: t.titulo || "" };
}

function resumir(t) {
  agregar(`Resúmeme el tema <b>${esc(t.nombre)}</b>`, "yo");
  accion(async (nodo) => {
    const r = await api("/api/resumir", { paginas: estado.paginas, tema: refTema(t) });
    pintarRespuesta(nodo, r, `Resumen · ${t.nombre}`);
  });
}

function practicar(t) {
  agregar(`Hazme preguntas de <b>${esc(t.nombre)}</b>`, "yo");
  accion(async (nodo) => {
    const r = await api("/api/practicar", { paginas: estado.paginas, tema: refTema(t), cantidad: 4 });
    const b = nodo.querySelector(".burbuja");
    if (!r.preguntas.length) {
      b.innerHTML = `<div class="etiqueta">Practicar</div>No pude crear preguntas verificables para este tema.`;
      return;
    }
    b.innerHTML = `<div class="etiqueta">Practicar · ${esc(t.nombre)}</div>
      <p>Intenta responder antes de ver la respuesta:</p>
      <div class="quiz">${r.preguntas.map((q, i) => `
        <div class="tarjeta">
          <div class="q">${i + 1}. ${esc(q.pregunta)}</div>
          <textarea rows="1" placeholder="Tu respuesta (opcional)"></textarea>
          <button type="button" class="ver">Ver respuesta</button>
          <div class="r" hidden>${markdown(q.respuesta)}
            <div style="margin-top:6px">Fuente: <button type="button" class="cita" data-cita="${esc(q.cita)}">${esc(q.cita)}</button></div></div>
        </div>`).join("")}</div>
      ${r.avisos.length ? `<div class="avisos">${r.avisos.map((a) => "⚠️ " + esc(a)).join("<br>")}</div>` : ""}
      <div class="pie"><span style="margin-left:auto">${esc(r.modo)}</span></div>`;
  });
}

/* ----------------------------------------------------- panel de fuente */
function abrirFuente(etiqueta, nodoMsg) {
  const fuentes = (nodoMsg && nodoMsg._fuentes || []).filter((f) => f.etiqueta === etiqueta);
  const pagina = buscarPagina(etiqueta);
  $("#fuente-titulo").textContent = `${etiqueta}${pagina ? " · " + pagina.nombre : ""}`;
  let html = "";
  if (pagina) {
    let texto = esc(pagina.texto);
    for (const f of fuentes) {   // resalta los fragmentos usados dentro de la página
      const trozo = esc(f.texto).split(/\s+/).slice(0, 8).join(" ");
      if (trozo.length > 15) texto = texto.replace(trozo, `<mark>${trozo}</mark>`);
    }
    html = `<div class="bloque">${texto}</div>`;
  } else {
    html = fuentes.map((f) => `<div class="bloque">${esc(f.texto)}</div>`).join("") || "No encontré ese texto.";
  }
  $("#fuente-cuerpo").innerHTML = html;
  $("#fuente").hidden = $("#velo").hidden = false;
  $("#fuente-cuerpo").querySelector("mark")?.scrollIntoView({ block: "center" });
}

function buscarPagina(etiqueta) {
  const m = etiqueta.match(/(?:Doc (\d+) )?(p\.|secci[oó]n) ?(\d+)/i);
  if (!m) return null;
  const doc = m[1] ? +m[1] : 1, num = +m[3];
  return estado.paginas.find((p) => p.doc === doc && p.pagina === num) || null;
}

function cerrarFuente() { $("#fuente").hidden = $("#velo").hidden = true; }

/* ------------------------------------------------------------- eventos */
function ajustarAltura() {
  const t = $("#pregunta"); t.style.height = "auto"; t.style.height = Math.min(t.scrollHeight, 140) + "px";
}

function eventos() {
  const zona = $("#zona");
  $("#archivo").onchange = (e) => cargarArchivos(e.target.files);
  zona.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") $("#archivo").click(); });
  ["dragenter", "dragover"].forEach((ev) => zona.addEventListener(ev, (e) => { e.preventDefault(); zona.classList.add("sobre"); }));
  ["dragleave", "drop"].forEach((ev) => zona.addEventListener(ev, (e) => { e.preventDefault(); zona.classList.remove("sobre"); }));
  zona.addEventListener("drop", (e) => cargarArchivos(e.dataTransfer.files));

  $("#btn-ejemplo").onclick = async () => {
    const r = await fetch("ejemplos/apuntes_ia.pdf");
    cargarArchivos([new File([await r.blob()], "apuntes_ia.pdf", { type: "application/pdf" })]);
  };
  $("#btn-inicio").onclick = () => { if (!estado.paginas.length || confirm("¿Volver al inicio y cerrar este documento?")) reiniciar(); };

  $("#form").onsubmit = (e) => { e.preventDefault(); preguntar($("#pregunta").value); };
  $("#pregunta").addEventListener("input", ajustarAltura);
  $("#pregunta").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); preguntar($("#pregunta").value); }
  });
  $("#sugerencias").onclick = (e) => { if (e.target.tagName === "BUTTON") preguntar(e.target.textContent); };

  $("#temas").onclick = (e) => {
    const b = e.target.closest("button[data-accion]"); if (!b) return;
    const li = b.closest(".tema"), t = estado.analisis.temas[+li.dataset.i];
    document.querySelectorAll(".tema").forEach((x) => x.classList.toggle("activo", x === li));
    $("#lateral").classList.remove("abierto");
    (b.dataset.accion === "resumir" ? resumir : practicar)(t);
  };
  $("#lateral-toggle").onclick = () => {
    const abierto = $("#lateral").classList.toggle("abierto");
    $("#lateral-toggle").setAttribute("aria-expanded", abierto);
  };

  $("#hilo").onclick = (e) => {
    const c = e.target.closest("button.cita");
    if (c) return abrirFuente(c.dataset.cita, c.closest(".msg"));
    const v = e.target.closest(".ver");
    if (v) { const r = v.nextElementSibling; r.hidden = !r.hidden; v.textContent = r.hidden ? "Ver respuesta" : "Ocultar respuesta"; }
  };
  $("#velo").onclick = cerrarFuente;
  $("#fuente-cerrar").onclick = cerrarFuente;
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") cerrarFuente(); });
}

async function cargarEstado() {
  try {
    const s = await api("/api/estado");
    $("#estado").classList.toggle("ia", s.ia_generativa);
    $("#estado-txt").textContent = s.ia_generativa ? "IA generativa activa" : "Modo local";
  } catch { $("#estado-txt").textContent = "Sin conexión con el servidor"; }
}

eventos();
cargarEstado();
