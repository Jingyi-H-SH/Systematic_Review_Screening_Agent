# Systematic Review Screening

## multi-agent

```sh
cd multi-agent
python -m venv .venv
# Activate .venv, then:
pip install -r requirements.txt
# Set OPENAI_API_KEY in your local environment or .env file.
uvicorn app:app --host 127.0.0.1 --port 8000
```

## single-model

Node.js 22.13 or newer.

```sh
cd single-model
npm ci
npm run build
npm run db:init
npm run dev -- --hostname 127.0.0.1
```

Open `http://127.0.0.1:5173` and enter your own key in **API key & model**.

```sh
npm run typecheck
npm test
```

Both commands above run without model API calls. Local development is bound to loopback. A public deployment must supply a trusted authentication layer and its own D1/R2 bindings.
