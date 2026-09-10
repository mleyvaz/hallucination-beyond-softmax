# Prerregistro de los Experimentos 9 y 10 (A10, A11) — 10-sep-2026, ANTES de la primera inferencia

Motivo: bloqueante B1 de la tercera reseña adversarial (ChatGPT sobre v19, 10-sep-2026). El
Experimento 8 (A9) fija el contenido y varía **marcos completos**, de modo que sus diferencias
(0,73 y 0,93) son efectos de plantillas concretas y **no** una descomposición identificada en
«credibilidad» y «factividad». El revisor ofrece dos vías: (i) rebajar la redacción a efectos de
plantilla, o (ii) aportar el cruce fuente × verbo, la separación adjetivo/sustantivo y una
intervención sobre los textos originales y sus concatenaciones. Se elige (ii); la vía (i) se
aplica de todos modos a lo que estos experimentos no identifiquen.

Modelos, datos y regla son los ya publicados: `MODEL_A` (soporte) y `MODEL_B` (contradicción) de
`experiment_a2_conflicting_evidence.py`, los 20 ítems de `CONFLICT_ITEMS`, la compuerta léxica de
`experiment_a5_lexical_gate.py` (Jaccard ≥ θ) y la regla de margen `T+F>1 ∧ min(T,F) ≥ 0,15`.
Unidad de remuestreo: el ítem (20). Intervalos: bootstrap percentil, 5.000 remuestras, semilla 7.

---

## A10 — Cruce factorial fuente × verbo, con adjetivo y sustantivo separados

**Diseño.** Contenido fijo (los mismos 20 enunciados X y sus negaciones que usó A9), y marco
generado por el cruce completo de tres factores:

| Factor | Niveles |
|---|---|
| Sustantivo de la fuente | `document` (neutro), `pamphlet` (baja credibilidad) |
| Adjetivo | ninguno, `disputed` |
| Verbo | `states` (neutro), `claims` (no factivo), `confirms` (factivo) |

2 × 2 × 3 = 12 marcos × 2 polaridades × 20 ítems = **480 oraciones**, puntuadas por ambas cabezas
contra el enunciado desnudo X. Determinante fijo («The») en los doce marcos. Salida:
`validation_results_a10.csv`, `validation_summary_a10.txt`.

**Contrastes registrados** (todos pareados por ítem, con verbo fijo salvo donde se indique):

- **Q1 adjetivo, aislado:** `The pamphlet states X` − `The disputed pamphlet states X`, y
  `The document states X` − `The disputed document states X` (entailment de A).
- **Q2 sustantivo, aislado:** `The document states X` − `The pamphlet states X`.
- **Q3 interacción adjetivo × sustantivo:** diferencia de las dos diferencias de Q1.
- **Q4 verbo bajo fuente fija:** `confirms` − `claims` dentro de cada una de las cuatro fuentes; y
  la variación de ese contraste entre fuentes (interacción fuente × verbo).
- **Q5 señal de contradicción ciega a la polaridad:** para el contenido **de apoyo** X, número de
  ítems con contradicción de B ≥ 0,15 en cada una de las cuatro fuentes con el verbo `states`.

**Reglas de decisión, fijadas ahora:**

1. El artículo podrá atribuir un efecto **al adjetivo** solo si su contraste pareado tiene media
   ≥ 0,10 **bajo los dos sustantivos** y signo positivo en ≥ 15/20 ítems en ambos. En otro caso el
   efecto se reporta como propiedad de la frase de fuente completa.
2. Podrá describirse el efecto como **aditivo** (sin interacción) solo si |Q3| < 0,10 y su IC
   bootstrap contiene 0. En otro caso se reporta interacción y se retira toda descomposición en
   sumandos independientes.
3. Podrá hablarse de **factividad** como factor separado solo si el contraste `confirms` − `claims`
   conserva el signo y alcanza ≥ 0,10 **bajo las cuatro fuentes**.
4. La palabra «mecanismo» solo se usará para los factores que superen 1–3. Para los demás, el texto
   dirá «diferencia entre las formulaciones probadas».

**Límite declarado de antemano:** una intervención léxica identifica el efecto de una palabra en
estas plantillas; no identifica una representación interna de credibilidad ni distingue semántica
de respuesta aprendida a esos tokens. Eso se escribirá aunque los tres criterios se cumplan.

---

## A11 — Intervención sobre los estímulos originales y sus concatenaciones

**Diseño.** Sobre los 20 ítems originales, se sustituye **solo la frase nominal de la fuente** por
`The document`, dejando verbo y contenido verbatim (única excepción de concordancia: ítem 1,
`records state` → `document states`). Cuatro brazos:

| Brazo | Vano de apoyo | Vano refutador |
|---|---|---|
| `original` | verbatim | verbatim |
| `ref_neutral` | verbatim | fuente → `The document` |
| `sup_neutral` | fuente → `The document` | verbatim |
| `both_neutral` | fuente → `The document` | fuente → `The document` |

Medidas por brazo: T = entailment de A sobre (apoyo → enunciado); F = contradicción de B sobre
(refutador → enunciado); compuerta léxica; bandera de la regla de margen; y **protocolo holístico**
sobre la premisa concatenada (apoyo + « » + refutador), con ambas cabezas. La tabla de
sustituciones se publica íntegra en el CSV para que sea auditable. Salida:
`validation_results_a11.csv`, `validation_summary_a11.txt`.

**Predicciones registradas** (son exactamente lo que §5.4 afirma hoy como mecanismo):

- **P1.** Neutralizar la fuente refutadora eleva F por encima del margen en al menos **2 de los 3**
  ítems que el artículo atribuye al marcador (5, 11, 16).
- **P2.** La regla de margen pasa de 15/20 a **≥ 18/20** en el brazo `ref_neutral` o en `both_neutral`.
- **P3.** Los ítems 1 y 8, que el artículo atribuye a un vano de apoyo débilmente implicado, **no**
  se recuperan con `ref_neutral` y **sí** con `sup_neutral`.
- **P4.** El protocolo holístico sobre la concatenación neutralizada detecta **> 4/20** conflictos
  (la cifra publicada con los textos originales).

**Consecuencias registradas:**

- Si **P1 y P2** se cumplen, la afirmación causal de §5.4 sobre la descripción de la fuente en los
  estímulos originales queda respaldada por una intervención y se conserva, citando A11 y con el
  alcance «la descripción de la fuente», no «la palabra *disputed*», salvo que A10 aísle el adjetivo.
- Si **P1 o P2** fallan, se retira «the marker drove F below the margin» y todo lo que dependa de
  ella, y §5.4 pasa a la vía mínima del revisor.
- **P3** decide si se conserva la explicación de dos causas distintas para los cinco fallos.
- **P4** decide si el resultado holístico admite explicación por la descripción de la fuente. Si
  falla, se retira «two compounding reasons» y el holístico queda como resultado sin explicación
  identificada.
- El grado de contradicción de B sobre contenido de apoyo (Q5 de A10) decide si «the same
  phenomenon seen from two sides» se conserva o se rebaja a coincidencia observada.

**Límite declarado de antemano:** los cuatro brazos intervienen la descripción de la fuente, no el
verbo ni el contenido; una recuperación bajo `ref_neutral` identifica a la frase de fuente como
causa suficiente en estos textos, no a un componente léxico concreto ni a un mecanismo interno.
