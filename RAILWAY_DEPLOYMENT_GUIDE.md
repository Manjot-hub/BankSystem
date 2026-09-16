# Railway Deployment Guide for the Banking Chatbot

This guide covers the production-ready environment checklist and the exact deployment steps for this repo without Docker.

## 1) Production-ready environment checklist

### Required service dependencies

- OpenAI API access
  - Purpose: LLM generation, tool-calling, and guardrails when enabled
  - Required values:
    - OPENAI_API_KEY
    - OPENAI_ORG_ID (only if your organization requires it)

- PostgreSQL database
  - Purpose: audit logs, user/session metadata, future production state
  - Required value:
    - DATABASE_URL

- Redis
  - Purpose: rate limiting, sessions, cache, future async job support
  - Required value:
    - REDIS_URL

- Qdrant (recommended for production)
  - Purpose: production vector store instead of local Chroma
  - Required values:
    - QDRANT_URL
    - QDRANT_API_KEY (if cloud-managed)
    - QDRANT_COLLECTION_NAME

- LangSmith (optional but recommended)
  - Purpose: tracing and observability
  - Required values:
    - LANGSMITH_API_KEY
    - LANGSMITH_PROJECT

### Required app secrets

- SECRET_KEY
  - Must be a long random secret; never commit it
- AUTH_TOKEN
  - Must be a secure API key for the protected /chat endpoint
- JWT_ALGORITHM
  - Default is HS256
- JWT_EXPIRY_HOURS
  - Default is 24

### Required runtime settings

These are already defined in [src/bank_chatbot/config/settings.py](src/bank_chatbot/config/settings.py):

- EMBEDDING_MODEL
- EMBEDDING_DIMENSION
- EMBEDDING_BATCH_SIZE
- LLM_MODEL_PRIMARY
- LLM_MODEL_FALLBACK
- LLM_TEMPERATURE
- LLM_MAX_TOKENS
- CHROMA_PERSIST_DIR
- CHROMA_COLLECTION_NAME
- API_HOST
- API_PORT
- API_WORKERS
- API_TIMEOUT
- RATE_LIMIT_REQUESTS_PER_MINUTE
- RATE_LIMIT_BURST
- FEATURE_RAG_ENABLED
- FEATURE_TOOLS_ENABLED
- FEATURE_MULTI_AGENT_ENABLED
- FEATURE_HUMAN_HANDOFF_ENABLED

### Deployment checklist before launch

- [ ] Real OpenAI API key configured
- [ ] Real PostgreSQL connection URL configured
- [ ] Real Redis URL configured
- [ ] Real secret key configured
- [ ] Real API auth token configured
- [ ] Production feature flags reviewed
- [ ] Logging level set for production
- [ ] Qdrant or Chroma choice finalized
- [ ] Health checks enabled on /health and /ready
- [ ] Rate limiting enabled
- [ ] Auth middleware enabled for /chat
- [ ] Debug/verbose output disabled in production
- [ ] No placeholder credentials remain in repo or env files
- [ ] Railway project created and GitHub repo connected
- [ ] App startup command defined

---

## 2) Where each key comes from

### OpenAI API key

Source:
- https://platform.openai.com/api-keys

Add to Railway as:
- OPENAI_API_KEY
- OPENAI_ORG_ID (if needed)

### PostgreSQL

Recommended providers:
- Railway Postgres add-on
- Neon
- Supabase Postgres
- AWS RDS
- Azure Database for PostgreSQL

Add to Railway as:
- DATABASE_URL

### Redis

Recommended providers:
- Railway Redis add-on
- Upstash Redis
- AWS ElastiCache
- Azure Cache for Redis

Add to Railway as:
- REDIS_URL

### Qdrant

Recommended providers:
- Qdrant Cloud
- self-hosted Qdrant on a VM or managed infrastructure

Add to Railway as:
- QDRANT_URL
- QDRANT_API_KEY
- QDRANT_COLLECTION_NAME

### LangSmith

Source:
- https://smith.langchain.com

Add to Railway as:
- LANGSMITH_API_KEY
- LANGSMITH_PROJECT

### Security secrets

Generate locally with:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Use the output for:
- SECRET_KEY
- AUTH_TOKEN

---

## 3) Railway deployment steps without Docker

### Step 1: Prepare the repo

Make sure the project root is ready:

```bash
cd C:\Users\Dell\Desktop\BankSystem
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Then test the app locally:

```bash
python -m pytest -q
```

Expected local health:
- tests pass
- app starts in local mode without fake external credentials

---

### Step 2: Create the Railway project

1. Go to https://railway.com
2. Sign in with GitHub
3. Click New Project
4. Choose Deploy from GitHub repo
5. Select this repository

---

### Step 3: Add required services

Add a PostgreSQL service in Railway.

- This creates a DATABASE_URL automatically for you

Add a Redis service in Railway.

- This creates a REDIS_URL automatically for you

If you use Qdrant Cloud, add your Qdrant connection values manually in Railway Variables.

---

### Step 4: Add environment variables in Railway

Open the Railway project, then go to Variables.

Set all required values:

```env
OPENAI_API_KEY=your_key_here
OPENAI_ORG_ID=
SECRET_KEY=your_long_random_secret
AUTH_TOKEN=your_secure_api_key
DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/dbname
REDIS_URL=redis://default:password@host:port
QDRANT_URL=http://host:6333
QDRANT_API_KEY=
QDRANT_COLLECTION_NAME=banking_knowledge
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=bank-chatbot
API_HOST=0.0.0.0
API_PORT=8000
FEATURE_RAG_ENABLED=true
FEATURE_TOOLS_ENABLED=false
FEATURE_MULTI_AGENT_ENABLED=false
FEATURE_HUMAN_HANDOFF_ENABLED=false
LOG_LEVEL=INFO
LOG_FORMAT=json
```

Important:
- Railway automatically injects PORT
- do not hard-code a fixed port in production unless required
- use `PORT` from Railway when launching the app

---

### Step 5: Set the start command

In Railway, set the Start Command to:

```bash
python -m uvicorn src.bank_chatbot.api.server:app --host 0.0.0.0 --port ${PORT}
```

This is the cleanest option for this repo because the app already exposes a FastAPI app in [src/bank_chatbot/api/server.py](src/bank_chatbot/api/server.py).

Alternative command if you want to keep the app bound to Railway's automatic port:

```bash
python -m uvicorn src.bank_chatbot.api.server:app --host 0.0.0.0 --port 8000
```

Use the first option when possible because Railway manages `PORT` automatically.

---

### Step 6: Deploy the app

From Railway UI:
- click Deploy
- wait for build and startup
- open the generated public domain

Or from local terminal:

```bash
railway login
railway link <project-id-or-name>
railway up
```

---

### Step 7: Validate deployment

Check these endpoints after deployment:

- /health
- /ready
- /metrics
- /guardrails/status
- /config

Example:

```bash
curl https://your-railway-app.up.railway.app/health
```

Then test the chat endpoint:

```bash
curl -X POST https://your-railway-app.up.railway.app/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_token" \
  -d '{"message":"What is the funds availability policy?"}'
```

---

## 4) Production-specific recommendations for this repo

### Use local fallback only for dev

Your project already supports local model fallback when no API key is present. That is useful for local development, but for production you should keep:

- OpenAI key enabled for better generation quality
- tools and multi-agent features only when fully tested

### Keep the app stateless where possible

- keep business logic in the app
- move session and audit state into Postgres/Redis
- prefer external services over local files for production persistence

### Secure the chat API

This app already has API key middleware in [src/bank_chatbot/api/server.py](src/bank_chatbot/api/server.py), which is good. In production, also enforce:

- strong auth token
- TLS termination at Railway
- no secrets in logs
- no raw PII in debug output

### Health and readiness

The repo already includes /health and /ready endpoints, which are the correct minimum checks for Railway.

---

## 5) Final production deployment checklist

- [ ] Railway project created
- [ ] GitHub repo linked
- [ ] PostgreSQL service added
- [ ] Redis service added
- [ ] Real OPENAI_API_KEY added
- [ ] Real SECRET_KEY added
- [ ] Real AUTH_TOKEN added
- [ ] DATABASE_URL set
- [ ] REDIS_URL set
- [ ] QDRANT_URL set if used
- [ ] LANGSMITH_API_KEY set if used
- [ ] Start command configured
- [ ] App exposes port from Railway env
- [ ] /health responds with 200
- [ ] /ready responds successfully
- [ ] /chat works with API key present
- [ ] Rate limiting returns 429 under load
- [ ] logs and errors are safe and structured

---

## 6) Best practical choice for this repo

For this project, the best non-Docker path is:

- Railway for hosting
- Railway Postgres for DB
- Railway Redis for sessions/rate limiting
- OpenAI for LLM generation
- Qdrant Cloud for production vector search
- optional LangSmith for tracing

This gives you an easy, interview-ready deployment story without depending on Docker Desktop.

---

## 7) Recommended order of operations

1. Deploy the app with the local fallback working
2. Add PostgreSQL and Redis
3. Add real OpenAI key
4. Enable full generation and guardrails
5. Enable tool features only after validation
6. Add Qdrant and LangSmith for production polish
7. Promote to staging and production environment names

This order reduces deployment risk and keeps the architecture understandable in interviews.
