# Para el equipo: cómo seguir con NoEntendIA

## Lo que ya está hecho

- **App en línea con IA generativa:** https://noentendia.vercel.app (no hay que instalar
  nada para la demo ni para grabar el video).
- Código documentado (Python + LangChain + FastAPI; interfaz en `public/`), 25 pruebas.
- Guion del video y resumen ejecutivo en `entregables/`.

## Correrlo en su computador (opcional)

1. Necesitan Python 3.10 o superior.
2. Doble clic en `INICIAR_NOENTENDIA.bat` (la primera vez instala todo, tarda unos
   minutos) y se abre http://localhost:8000.
3. Pruebas: `.venv\Scripts\python -m pytest -q` → deben pasar 25.
4. **Claves de IA:** este zip NO trae claves (es a propósito). Sin claves la app funciona
   en *modo local* (responde con oraciones textuales del documento). Si quieren el modo
   IA en local, creen su propia clave gratis en https://console.groq.com/keys, copien
   `.env.example` como `.env` y péguenla en `GROQ_API_KEY_1`. Nunca suban ni compartan
   el `.env`.
5. Publicar cambios en la página en línea solo lo puede hacer el dueño del proyecto en
   Vercel (`npx vercel deploy --prod`). Si cambian código, envíenle la carpeta.

## Pendientes

- [ ] Escribir los nombres de los integrantes en `entregables/Resumen_Ejecutivo_NoEntendIA.docx`
      y exportarlo de nuevo a PDF (Archivo → Guardar como → PDF). Debe seguir en 1 página.
- [ ] Repartir las secciones del video (`entregables/GUION_VIDEO.md`): cada integrante
      debe explicar una parte en cámara. Máximo 10 minutos.
- [ ] Revisar las normas citadas en el análisis ético/legal antes de grabar.
- [ ] Ensayar la demo en https://noentendia.vercel.app: «Probar con apuntes de ejemplo» →
      temas → resumir → tocar una cita → «¿Quién inventó Prolog?» (debe decir que no está)
      → practicar → mostrar `pytest` en la terminal.
- [ ] Subir video, resumen y código (este zip) a la plataforma. Cierre: 9 de noviembre.

La arquitectura y las decisiones de diseño (útiles para la sustentación) están en `README.md`.
