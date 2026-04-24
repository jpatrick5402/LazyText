# LazyText

## Installation
```
git clone https://github.com/jpatrick5402/LazyText
cd LazyText/
python -m venv .venv
source .venv/bin/activate.xxx
source .env # You'll need to create your own API keys
pip install -r requirements.txt
python trainAndSave.py
uvicorn api_server:app --reload --host 0.0.0.0 --port 8000

curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"text":"I think I\'m sick"}' | jq
curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"text":"I miss you!"}' | jq
curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"text":"Ok"}' | jq
```
