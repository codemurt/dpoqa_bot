from fastapi import FastAPI
import time
import logging
import os

from aiogram import Bot, Dispatcher, types

from transformers import AutoTokenizer, AutoModel

import faiss
import torch
import numpy as np
import pickle

TOKEN = os.getenv('TOKEN')

WEBHOOK_PATH = f"/bot/{TOKEN}"
WEBHOOK_URL = "https://dpoqa-bot.onrender.com" + WEBHOOK_PATH

logging.basicConfig(filemode='a', level=logging.DEBUG)
bot = Bot(token=TOKEN)
dp = Dispatcher(bot=bot)

app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")
model = AutoModel.from_pretrained("cointegrated/rubert-tiny")

with open('questions.pkl', 'rb') as f:
    questions = pickle.load(f)

with open('answers.pkl', 'rb') as f:
    answers = pickle.load(f)

with open('question_embs.pkl', 'rb') as f:
    question_embeddings = pickle.load(f)

question_embeddings = torch.tensor(question_embeddings)
d = question_embeddings.size(1)
index = faiss.IndexFlatIP(d)
index.add(question_embeddings.numpy())

def semantic_search(query, index, question_embeddings, k=5):
    inputs = tokenizer(query, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    query_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    D, I = index.search(query_embedding.reshape(1, -1), k)
    results = []
    for i, score in zip(I[0], D[0]):
        results.append((score, questions[i], answers[i]))
    return results

@app.on_event("startup")
async def on_startup():
    webhook_info = await bot.get_webhook_info()
    if webhook_info.url != WEBHOOK_URL:
        await bot.set_webhook(
            url=WEBHOOK_URL
        )

@dp.message_handler(commands=['start'])
async def start_handler(message: types.Message):
    user_id = message.from_user.id
    user_full_name = message.from_user.full_name
    logging.info(f'start: {user_id} {user_full_name} {time.asctime()}. Message: {message}')
    await message.reply(f"Привет, {user_full_name}! Чтобы воспользоваться ботом просто напиши свой вопрос. И бот постарается ответить на него.")

@dp.message_handler()
async def main_handler(message: types.Message):
    try:
        user_id = message.from_user.id
        user_full_name = message.from_user.full_name
        logging.info(f'Main: {user_id} {user_full_name} {time.asctime()}. Message: {message}')
        results = semantic_search(str(message), index, question_embeddings, k=1)
        await message.reply(results[0][2])
        
    except:
        logging.info(f'Main: {user_id} {user_full_name} {time.asctime()}. Message: {message}. LAST ERROR. AHTUNG! AAAAAA')
        await message.reply("Что-то пошло не так...")    

@app.post(WEBHOOK_PATH)
async def bot_webhook(update: dict):
    telegram_update = types.Update(**update)
    Dispatcher.set_current(dp)
    Bot.set_current(bot)
    await dp.process_update(telegram_update)

@app.on_event("shutdown")
async def on_shutdown():
    bot_session = await bot.get_session()
    await bot_session.close()

@app.get("/")
def main_web_handler():
    return "Everything ok!"