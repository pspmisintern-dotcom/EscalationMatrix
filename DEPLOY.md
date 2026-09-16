# Go-live: Vercel (frontend) + Render Docker (backend + Ollama)

Why this split: Vercel is serverless/static only — it cannot run Python,
FAISS, or Ollama. So the React UI lives on Vercel, and the FastAPI backend
plus Ollama (`qwen2.5:1.5b` + `nomic-embed-text`) share ONE Docker container
on Render. Repo root for both services is the `project/` folder.

## 0. Push to GitHub (repo root == `project/` contents)

```bash
cd "project"
git init
git add Dockerfile entrypoint.sh render.yaml vercel.json requirements.txt \
  backend data models frontend/src frontend/index.html frontend/package.json \
  frontend/vite.config.js frontend/.env.example .env.example README.md
git commit -m "Deploy: Vercel frontend + Render backend+Ollama"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```

Do NOT commit: `.env` (live OpenAI key), `.venv/`, `node_modules/`,
`frontend/dist/`, `*.log` — all covered by `.gitignore`.

## 1. Render — backend + Ollama (Docker)

1. Render dashboard -> New -> **Blueprint** -> select the repo
   (`render.yaml` sets Docker build, `plan: standard`, `/health` checks,
   and generous AI timeouts for shared CPUs).
   Alternative without Blueprint: New -> Web Service -> select repo ->
   Runtime **Docker**, Dockerfile Path `./Dockerfile`, Plan **Standard** (4 GB+).
2. Environment (dashboard -> service -> Environment):
   `OPENAI_API_KEY` — paste a **funded** key to keep the OpenAI service live on
   Render (chain is openai -> ollama -> heuristic; quota/offline just falls
   back to Ollama, which still works with the key unset/empty).
   `OPENAI_MODEL` (default `gpt-4o-mini`), `OPENAI_TIMEOUT` (default `30`),
   and `AI_PROVIDER` (default `auto`; `ollama` forces local-only, `off`
   disables AI) already have defaults from `render.yaml`/Dockerfile — override
   them in the dashboard only if you need to. Also set `FRONTEND_URL` and
   `OLLAMA_MODEL` to swap models later.
3. Deploy. First build takes **10–20 min** (torch + 1.3 GB of models).
4. Copy the service URL, e.g. `https://escalation-rag-backend.onrender.com`.
   Verify `GET <url>/health` returns `{"status":"ok"}`.

RAM reality: Render free = 512 MB; embeddings + the 1.1 GB model need ~2 GB.
On free, expect OOM restarts or the AI block timing out (the app still returns
instant historical answers). Standard (4 GB) is the minimum for reliable AI,
and paid instances don't sleep (free spins down after ~15 min idle).

## 2. Vercel — frontend

1. Vercel dashboard -> Add New -> Project -> Import the same repo.
2. Framework Preset: **Vite** (auto-detected; `vercel.json` pins the
   `frontend/` build). No Root Directory override needed.
3. Environment Variables: add `VITE_API_BASE` = your Render URL
   (no trailing slash), e.g. `https://escalation-rag-backend.onrender.com`.
4. Deploy. Open the Vercel URL and search:
   - exact historical cause -> instant answer;
   - novel cause -> instant grounded answer, AI enhancement streams in
     (cold/shared CPU can take up to ~2 min; `ai_skip` keeps the instant
     answer if the model is slower than `AI_TOTAL_TIMEOUT=120`).

## 3. Caveats (cloud vs local)

- `/save` appends to the CSV inside the container: **ephemeral** — edits vanish
  on redeploy/restart. For durable saves, attach a Render Persistent Disk
  mounted at `/app/data` (paid) or point `data_path` at external storage.
- FAISS `models/` artifacts are committed (450 KB) so boot is instant; if
  missing, the backend rebuilds the index automatically at startup.
- Backend CORS is locked to `FRONTEND_URL` (your Vercel URL) + local dev —
  set `FRONTEND_URL` on Render after the first Vercel deploy and redeploy.
- Local dev is unchanged: `VITE_API_BASE` unset -> `http://127.0.0.1:8000`.
