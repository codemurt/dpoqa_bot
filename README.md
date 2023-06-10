# dpoqa_bot
## Deployment

```
git clone https://github.com/codemurt/dpoqa_bot.git
cd dpoqa_bot
pip install -r requirements.txt
```

```
ngrok http 8000
```

```
export TOKEN=<YOUR_TELEGRAM_BOT_TOKEN>
```

```
gunicorn -k uvicorn.workers.UvicornWorker main:app --timeout 600
```
