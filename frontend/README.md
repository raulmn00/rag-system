# RAG System — Frontend

A React + TypeScript SPA that drives the
[RAG backend](../backend). Drop in `.md`/`.txt` files, ask
questions, watch the agent retrieve passages with relevance bars,
and — on demand — score the answer with LLM-judged faithfulness +
answer-relevancy gauges (computed by the backend's in-tree
Evaluator).

## Stack

- **React 19 + TypeScript 6** (Vite-scaffolded, `strict: true`)
- **No UI library** — every component and animation is written by
  hand in plain CSS. Single dark theme, teal accent, system fonts.
- **Native `fetch`** for the three calls; no SDK on top of it.

## Run

```bash
npm install
npm run dev               # :5173
```

The dev server expects the backend on `http://localhost:8000`. Set
`VITE_API_URL` to point at a different host (Cloud Run / Render /
etc.) — see below.

## Configuring the API URL

The base URL comes from `VITE_API_URL` with a localhost fallback:

```bash
cp .env.example .env
# edit .env:
# VITE_API_URL=https://your-backend.example.com
```

Vite reads env vars at build time. Restart `npm run dev` after
changing them.

## Build

```bash
npm run build             # tsc -b (typecheck whole tree) then vite build
npm run preview           # serve the production build locally
```

`npm run build` fails the whole build if the TypeScript pass fails —
no untyped code can ship. Current bundle is ~200 KB raw / ~64 KB
gzipped JS plus ~15 KB / ~3.5 KB gzipped CSS.

## Project layout

```
src/
├── api.ts                       Typed client for /ask, /upload, /evaluate
├── App.tsx                      State machine for the three lifecycles
├── App.css                      Layout shell (860px max, responsive)
├── index.css                    Design tokens (colors, radii, motion)
├── vite-env.d.ts                Types for VITE_API_URL
└── components/
    ├── Spinner.tsx              Reusable loading indicator (md / sm)
    ├── UploadArea.tsx           Drag-and-drop + click-to-select
    ├── QuestionForm.tsx         Textarea + Ctrl/Cmd+Enter submit
    ├── AnswerBox.tsx            Answer with preserved line breaks
    ├── SourceList.tsx           Sources with relevance bars
    └── MetricsPanel.tsx         Animated SVG gauges (Ragas)
```

Every component prop is explicitly typed; `any` is not used
anywhere. The metrics panel's state is a discriminated union
(`MetricsState`) rather than three parallel nullables — TypeScript
forces the parent to handle every variant.
