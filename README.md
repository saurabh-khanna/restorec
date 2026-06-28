# Conversational Restaurant Recommender: Trust Study

> Built on [surveychat](https://github.com/surveychat/surveychat), an open-source platform for embedding LLM chatbots inside Qualtrics surveys. This repository specialises surveychat into a single fixed experiment. See [Relation to surveychat](#relation-to-surveychat) for what changed.

A conversational version of a 2 × 2 between-subjects experiment on consumer trust in conversational recommender systems (*"Understanding consumer trust in conversational recommender systems: the role of framing and consumption motivation"*, Conversational Study 2). Participants have a live, free-flowing chat with a restaurant recommender rather than viewing a static screenshot, then paste the transcript back into Qualtrics.

## Design

Two factors, fully crossed into four arms:

- **Message framing**, manipulated in the system prompt and the bot's opening line:
  - **Expert**: recommendations attributed to food critics & nutritionists
  - **Bandwagon**: recommendations attributed to user ratings & reviews
- **Consumption motivation**, manipulated in a scenario banner above the chat (the bot is blind to this):
  - **Utilitarian**: "affordable, quick, healthy, and filling meals"
  - **Hedonic**: "tasty food and cozy, relaxing atmosphere"

Each arm is reached with its own passcode:

| Passcode | Framing | Motivation |
|---|---|---|
| `AMBER` | Expert | Utilitarian |
| `CORAL` | Expert | Hedonic |
| `OLIVE` | Bandwagon | Utilitarian |
| `SLATE` | Bandwagon | Hedonic |

Held constant across every arm: the bot recommends the same three restaurants (**Sirocco's Table**, **The Organic Boho**, **Shiso Fine**) and never names any other. Only the recommendation *source* (the framing) differs systematically between arms. The bot may add plausible per-conversation detail as long as it stays internally consistent.

## Setup

Requires Python 3.10+ and an API key for your LLM provider.

```bash
git clone <this-repo-url>
cd restorec
pip install -r requirements.txt
cp .env.example .env          # then add your key: OPENAI_API_KEY=...
streamlit run app.py
```

Open <http://localhost:8501>. By default the app calls the University of Amsterdam LLM proxy (`API_BASE_URL` in `app.py`); for local testing without proxy access, point `API_BASE_URL` at `https://api.openai.com/v1` and use a personal key.

## Configuration

All study settings live in the **RESEARCHER CONFIGURATION** block near the top of [`app.py`](app.py): the four conditions and their passcodes, prompts, opening lines, and scenario banners, plus:

| Setting | What it does |
|---|---|
| `MODEL` | Model used for every arm (model is not a factor). Pin a dated snapshot before data collection if your proxy supports it. |
| `MIN_TURNS_BEFORE_RECS` | The bot won't present its three recommendations before the participant's Nth message. |
| `RECOMMEND_BY_TURN` | The bot is told to present the recommendations by this turn at the latest. |
| `MAX_EXCHANGES` | Soft cap on participant messages; on reaching it the bot wraps up and the input is disabled. |
| `END_CHAT_BUTTON_BELOW` | Layout toggle for where the End-chat button sits. |

The **End chat** button only appears once the bot has actually presented all three recommendations; after that, the bot gently steers the conversation toward a close.

## Deployment

The app must be served over HTTPS to embed in Qualtrics.

| Option | How |
|---|---|
| **Streamlit Cloud** (easy to test) | Push to GitHub → [share.streamlit.io](https://share.streamlit.io) → add `OPENAI_API_KEY` under Advanced settings → Secrets → Deploy |
| **Docker** | `docker compose up --build` (reads your `.env`) |
| **Any cloud server** | `streamlit run app.py --server.port 80 --server.headless true` |

## Qualtrics integration

1. **Survey Flow → Randomizer** with four evenly-presented branches.
2. In each branch, set an embedded-data field to that arm's passcode (e.g. `chat_code = AMBER`) and add a **Text / Graphic** block that shows the code and embeds the app:

   ```html
   <p>Your code is: <strong>${e://Field/chat_code}</strong></p>
   <p>Enter it in the recommender below to begin. When you're done, click <strong>End chat</strong> and copy your transcript.</p>
   <div style="border:1px solid #d4d4d4;border-radius:8px;overflow:hidden;margin:16px 0;">
     <iframe src="https://your.app.url/" width="100%" height="700" style="display:block;border:none;" allow="clipboard-write"></iframe>
   </div>
   ```

3. Add a **Text Entry** question right after for the participant to paste the transcript.
4. Recover condition assignment from the embedded-data passcode at analysis time, not from the transcript (which excludes condition info).

## Transcript format

On **End chat**, the participant copies a JSON transcript:

```json
{
  "messages": [
    {"role": "participant", "content": "Hello!",    "timestamp": "2026-03-06T14:22:01+00:00"},
    {"role": "assistant",   "content": "Hi there!", "timestamp": "2026-03-06T14:22:03+00:00"}
  ]
}
```

Condition name and model are deliberately excluded so the participant can't infer their arm.

Parse it for analysis:

```python
import json, pandas as pd
df = pd.DataFrame(json.loads(transcript_string)["messages"])
```

```r
df <- as.data.frame(jsonlite::fromJSON(transcript_string)$messages)
```

Worked templates: [Python/pandas notebook](analysis/python_pandas_qualtrics_json.ipynb), [R/tidyverse Rmd](analysis/r_tidyverse_qualtrics_json.Rmd). The passcode → condition map and manipulation-fidelity checks are documented at the bottom of [`app.py`](app.py).

## Relation to surveychat

This repo is a fork of [surveychat](https://github.com/surveychat/surveychat) specialised into one experiment. The chat engine (session handling, passcode routing, streaming chat, transcript export) is surveychat's. The study-specific additions are a per-condition scenario banner, turn-aware pacing of the recommendation, a soft turn cap, a recommendation-gated End-chat button, and a single-container layout. Search for `STUDY 2 CHANGE` in `app.py` to find them.

## Troubleshooting

| Symptom | Fix |
|---|---|
| "Code not recognised" | Passcode doesn't match one in `app.py` (`AMBER` / `CORAL` / `OLIVE` / `SLATE`). Not case-sensitive. |
| "OPENAI_API_KEY not found" | Ensure `.env` contains `OPENAI_API_KEY=sk-...` with no spaces around `=`. |
| Chat returns an error | Check `API_BASE_URL` and that your key is valid for that endpoint. |
| Port 8501 already in use | `pkill -f "streamlit run"`, or start on another port with `--server.port 8502`. |

## License

[AGPL-3.0](LICENSE), inherited from surveychat.
