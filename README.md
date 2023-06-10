# dpoqa_bot
## Deployment

```
git clone https://github.com/codemurt/dpoqa_bot.git
cd dpoqa_bot
```

Set up virtual enviroment(Linux):
```
python3 -m venv <virtual-environment-name>
source <virtual-environment-name>/bin/activate
```

Windows:
```
pip install virtualenv
virtualenv <virtual-environment-name>
source <virtual-environment-name>/Scripts/activate
```

```
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
