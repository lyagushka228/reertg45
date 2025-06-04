import os
import re
import logging
import sqlite3
import base64
import mimetypes
from collections import deque
import nest_asyncio
nest_asyncio.apply()
import asyncio
import requests
import time
import random
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
import openai
from PyPDF2 import PdfReader
from docx import Document
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
from telegram import Update, Bot
from telegram.constants import ParseMode
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from urllib.parse import urlparse, urlunparse, quote, unquote
from dotenv import load_dotenv
load_dotenv()

# ===================== Конфигурация =====================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY не установлен в переменных окружения!")
openai.api_key = OPENAI_API_KEY

ADMIN_IDS = [int(x.strip()) for x in os.getenv("ADMIN_IDS", "84544725").split(",")]
GPT_MODEL = "o4-mini-2025-04-16"
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, "users.db")
MAX_CONTEXT_LENGTH = 20
TELEGRAM_MAX_MESSAGE_LENGTH = 4096
MAX_FILE_SIZE = 10_000_000
MAX_TEXT_LENGTH = 5000
MAX_URL_LENGTH = 2000

user_context = {}
web_cache = {}
CACHE_EXPIRATION_SECONDS = 3600
gpt_queue = asyncio.Queue()

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ===================== Очистка памяти =====================
def trim_user_context_after_file(user_id, keep_last_n=2):
    """
    После загрузки файла или картинки оставляет в user_context только последние N сообщений.
    """
    if user_id in user_context:
        msgs = list(user_context[user_id])
        user_context[user_id] = deque(msgs[-keep_last_n:], maxlen=MAX_CONTEXT_LENGTH)


# ===================== URL нормализация =====================
def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    netloc = parsed.netloc.encode('idna').decode('ascii')
    path = quote(unquote(parsed.path), safe="/")
    params = quote(unquote(parsed.params), safe=";=")
    query = quote(unquote(parsed.query), safe="=&?")
    fragment = quote(unquote(parsed.fragment), safe="")
    return urlunparse((parsed.scheme, netloc, path, params, query, fragment))

# ===================== Markdown эскейпинг =====================
def custom_escape_markdown(text: str) -> str:
    escape_chars = r'\*_['
    for ch in escape_chars:
        text = text.replace(ch, '\\' + ch)
    return text

def split_text(text: str, max_length: int = TELEGRAM_MAX_MESSAGE_LENGTH) -> list:
    parts = []
    while len(text) > max_length:
        split_index = text.rfind('\n', 0, max_length)
        if split_index == -1:
            split_index = max_length
        parts.append(text[:split_index])
        text = text[split_index:]
    if text:
        parts.append(text)
    return parts

async def safe_reply_text(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str):
    try:
        await update.message.reply_text(
            text,
            parse_mode=ParseMode.MARKDOWN,
            disable_web_page_preview=True
        )
    except Exception as ex:
        logger.exception("Ошибка отправки с Markdown, пробую plain text")
        try:
            await update.message.reply_text(
                text,
                parse_mode=None,
                disable_web_page_preview=True
            )
        except Exception as ex2:
            logger.exception("Ошибка отправки plain text")

async def send_long_message(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str):
    parts = split_text(text)
    for part in parts:
        await safe_reply_text(update, context, part)

# ===================== Playwright =====================
async def async_fetch_with_playwright(url: str) -> str:
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, timeout=30000)
            content = await page.content()
            await browser.close()
            soup = BeautifulSoup(content, "html.parser")
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text(separator="\n")
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            cleaned_text = "\n".join(lines)
            return cleaned_text
    except Exception as e:
        logger.exception(f"Ошибка в async_fetch_with_playwright для {url}: {e}")
        raise
# ===================== Preplexity =====================

async def search_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not user or not is_user_authorized(user.id):
        await update.message.reply_text(
            custom_escape_markdown("Извините, у вас нет доступа к этому боту."),
            parse_mode=ParseMode.MARKDOWN
        )
        return
    # Если есть аргументы - сразу выполняем поиск
    if context.args:
        prompt = " ".join(context.args).strip()
        await update.message.reply_text("Выполняю поиск Perplexity…", parse_mode=None)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, search_perplexity, prompt)
        result = format_perplexity_markdown_for_telegram(result)
        if len(result) > TELEGRAM_MAX_MESSAGE_LENGTH:
            result = result[:TELEGRAM_MAX_MESSAGE_LENGTH-100] + "\n\n[Ответ обрезан]"
        await send_long_message(update, context, result)
        return
    # Нет аргументов — ждём следующее сообщение от пользователя
    user_context.setdefault(user.id, deque(maxlen=MAX_CONTEXT_LENGTH))
    user_context[user.id].append({"role": "system", "content": "/search_no_args"})
    await update.message.reply_text(
        custom_escape_markdown("Пришлите текст или голосовое сообщение для поиска."),
        parse_mode=ParseMode.MARKDOWN
    )




def format_perplexity_markdown_for_telegram(text: str) -> str:
    # 1. Убираем все # и --- (заголовки и разделители)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^---$', '', text, flags=re.MULTILINE)
    
    # 2. Markdown-ссылки вида [текст](url) оставить (Telegram их поддерживает)
    # 3. Ссылки типа [1] заменить на просто 1.
    text = re.sub(r'\[(\d+)\]', r'\1.', text)
    
    # 4. Убираем тройные пустые строки
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 5. Лишние пробелы после \n
    text = re.sub(r'\n +', '\n', text)
    
    # 6. Финальное экранирование спецсимволов, кроме тех, что в markdown
    def escape_non_markdown_symbols(match):
        s = match.group(0)
        if s.startswith('**') or s.startswith('_') or s.startswith('['):
            return s
        return '\\' + s
    # Только те символы, которые не стоят в начале markdown-элементов
    text = re.sub(r'(?<!\*)\*(?!\*)', r'\*', text)
    text = re.sub(r'(?<!_)_(?!_)', r'\_', text)
    
    # 7. Удаляем пустые строки в начале/конце
    text = text.strip()
    return text



async def handle_search(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not user or not is_user_authorized(user.id):
        await update.message.reply_text(
            custom_escape_markdown("Извините, у вас нет доступа к этому боту."),
            parse_mode=ParseMode.MARKDOWN
        )
        return
    prompt = None
    # Текст из команды: /search <текст>
    if context.args:
        prompt = " ".join(context.args).strip()
    # Если голос — обработаем ниже
    elif update.message.voice:
        voice = update.message.voice
        if voice.file_size and voice.file_size > MAX_FILE_SIZE:
            await update.message.reply_text("Голосовое слишком большое.", parse_mode=None)
            return
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
        file = await context.bot.get_file(voice.file_id)
        file_path = os.path.join(BASE_DIR, f"voice_search_{user.id}.ogg")
        await file.download_to_drive(custom_path=file_path)
        with open(file_path, "rb") as audio_file:
            transcript = await asyncio.to_thread(
                openai.Audio.transcribe, "whisper-1", audio_file
            )
            prompt = transcript["text"].strip()
        os.remove(file_path)
    else:
        await update.message.reply_text(
            custom_escape_markdown("Пришлите текст для поиска после команды /search"),
            parse_mode=ParseMode.MARKDOWN
        )
        return

    await update.message.reply_text("Выполняю поиск Perplexity…", parse_mode=None)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, search_perplexity, prompt)
    result = format_perplexity_markdown_for_telegram(result)
    if len(result) > TELEGRAM_MAX_MESSAGE_LENGTH:
       result = result[:TELEGRAM_MAX_MESSAGE_LENGTH-100] + "\n\n[Ответ обрезан]"
    await send_long_message(update, context, result)



async def fetch_url_content(url: str) -> str:
    url = normalize_url(url)
    now = time.time()
    if url in web_cache:
        content, timestamp = web_cache[url]
        if now - timestamp < CACHE_EXPIRATION_SECONDS:
            logger.info(f"Возвращаю кэшированное содержимое для {url}")
            return content
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, как Gecko) "
            "Chrome/92.0.4515.159 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5"
    }
    for attempt in range(3):
        try:
            response = await asyncio.to_thread(requests.get, url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                for script in soup(["script", "style"]):
                    script.decompose()
                text = soup.get_text(separator="\n")
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                cleaned_text = "\n".join(lines)
                if len(cleaned_text) < 50:
                    logger.warning(f"Слишком мало текста извлечено с {url}, fallback с Playwright")
                    raise Exception("Мало содержимого")
                logger.info(f"Длина извлечённого текста с {url}: {len(cleaned_text)} символов")
                web_cache[url] = (cleaned_text, now)
                return cleaned_text
            elif response.status_code in (403, 429):
                logger.error(f"Ошибка загрузки страницы {url}: статус {response.status_code}")
                await asyncio.sleep(random.uniform(2, 5))
            else:
                logger.error(f"Неожиданный статус {response.status_code} для {url}")
                await asyncio.sleep(random.uniform(2, 5))
        except Exception as e:
            logger.exception(f"Ошибка при загрузке URL {url} на попытке {attempt+1}: {e}")
            await asyncio.sleep(random.uniform(2, 5))
    try:
        logger.info(f"Fallback: попытка получить содержимое {url} с помощью async_playwright")
        text = await async_fetch_with_playwright(url)
        logger.info(f"Длина извлечённого текста с async_playwright: {len(text)} символов")
        web_cache[url] = (text, now)
        return text
    except Exception as e:
        logger.exception(f"Fallback с использованием async_playwright не удался для {url}: {e}")
        return ""

def search_perplexity(prompt: str, max_tokens: int = 2000) -> str:
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "sonar",
        "messages": [
            {"role": "system", "content": "Be precise and concise. Respond in Russian if the query is in Russian."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "web_search_options": {
            "search_context_size": "medium"
        }
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        result = data['choices'][0]['message']['content'].strip()
        citations = data.get('citations', [])
        if citations:
            cite_list = "\n".join([f"[{i+1}] {c}" for i, c in enumerate(citations[:5])])
            result += f"\n\nИсточники:\n{cite_list}"
        return result
    except Exception as ex:
        logger.exception("Ошибка Perplexity API")
        return f"Ошибка при поиске через Perplexity: {ex}"

# ===================== Обработка файлов =====================
def process_pdf_local(file_path: str) -> str:
    try:
        with open(file_path, "rb") as f:
            reader = PdfReader(f)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
        if not text.strip():
            return "Из файла не удалось извлечь текст. Возможно, документ является сканом."
        return text
    except Exception as e:
        logger.exception("Ошибка при чтении PDF")
        return f"Ошибка при чтении PDF: {e}"

def process_docx_local(file_path: str) -> str:
    try:
        doc = Document(file_path)
        text = "\n".join(para.text for para in doc.paragraphs)
        if not text.strip():
            return "Из файла не удалось извлечь текст."
        return text
    except Exception as e:
        logger.exception("Ошибка при чтении DOCX")
        return f"Ошибка при чтении DOCX: {e}"

def process_excel_local(file_path: str) -> str:
    try:
        dfs = pd.read_excel(file_path, sheet_name=None)
        texts = []
        for sheet_name, df in dfs.items():
            texts.append(f"=== Лист: {sheet_name} ===")
            texts.append(df.head(15).to_csv(index=False))
        return "\n".join(texts)
    except Exception as e:
        logger.exception("Ошибка при чтении Excel")
        return f"Ошибка при чтении Excel: {e}"

# ===================== База данных =====================
def init_db():
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS authorized_users (
            id INTEGER PRIMARY KEY,
            allowed INTEGER NOT NULL DEFAULT 0
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS reminders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            remind_time TEXT NOT NULL,
            text TEXT NOT NULL,
            done INTEGER NOT NULL DEFAULT 0
        )
    ''')
    conn.commit()
    c.execute("SELECT COUNT(*) FROM authorized_users")
    count = c.fetchone()[0]
    if count == 0:
        logger.warning("В базе данных нет авторизованных пользователей!")
    conn.close()

def is_user_authorized(user_id: int) -> bool:
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()
    c.execute("SELECT allowed FROM authorized_users WHERE id = ?", (user_id,))
    row = c.fetchone()
    conn.close()
    logger.info(f"Проверка доступа user_id={user_id}: row={row}")
    return bool(row and row[0] == 1)

def add_authorized_user(user_id: int):
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO authorized_users (id, allowed) VALUES (?, ?)", (user_id, 1))
    conn.commit()
    conn.close()
    logger.info(f"Пользователь {user_id} добавлен с доступом.")

def remove_authorized_user(user_id: int):
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM authorized_users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()
    logger.info(f"Пользователь {user_id} удалён из базы.")

# ===================== GPT =====================
async def process_gpt_request(user_id: int, user_message: str, update: Update, context: ContextTypes.DEFAULT_TYPE):
    if user_id not in user_context:
        user_context[user_id] = deque(maxlen=MAX_CONTEXT_LENGTH)
    messages = user_context[user_id]
    messages.append({"role": "user", "content": user_message})
    try:
        response = await openai.ChatCompletion.acreate(
            model=GPT_MODEL,
            messages=list(messages)
        )
        reply = response["choices"][0]["message"]["content"].strip()
        messages.append({"role": "assistant", "content": reply})
        await send_long_message(update, context, reply)
    except Exception as e:
        logger.exception("Ошибка при вызове OpenAI API")
        await safe_reply_text(update, context, f"Ошибка при обращении к GPT: {e}")

async def call_gpt_api(user_id: int, user_message: str, update: Update, context: ContextTypes.DEFAULT_TYPE):
    await gpt_queue.put((user_id, user_message, update, context))

async def call_gpt_api_multimodal(user_id: int, content: list) -> str:
    """Отправляет мультимодальный запрос к GPT.

    Ранее функция была синхронной и вызывала API напрямую, из-за чего
    блокировался событийный цикл asyncio. Теперь запрос выполняется
    асинхронно через ``openai.ChatCompletion.acreate``.
    """

    # Для мультимодального запроса не сохраняем историю, отправляем только
    # текущее сообщение пользователя (подпись + изображение).
    messages = [{"role": "user", "content": content}]

    try:
        response = await openai.ChatCompletion.acreate(
            model=GPT_MODEL,
            messages=messages,
        )
        reply = response["choices"][0]["message"]["content"].strip()
        return reply
    except Exception as e:
        logger.exception("Ошибка при вызове OpenAI API (мультимодальный)")
        return f"Ошибка при обращении к GPT: {e}"


async def process_gpt_queue():
    while True:
        user_id, user_message, update, context = await gpt_queue.get()
        try:
            await process_gpt_request(user_id, user_message, update, context)
        except Exception as e:
            logger.exception("Ошибка обработки запроса из очереди")
        finally:
            gpt_queue.task_done()

# ===================== Напоминания =====================
async def parse_reminder_text(input_text: str) -> dict:
    current_time = datetime.now(ZoneInfo("Europe/Moscow")).strftime("%Y-%m-%d %H:%M:%S")
    system_prompt = (
        f"Current Moscow time: {current_time}\n"
        "You are a highly intelligent, structured assistant. Your job is to extract, systematize and format a reminder from the user's input, which may be a long voice message or a free-form list of items or tasks.\n"
        "Requirements:\n"
        "1. Analyze the input and, if the user gives a list or several tasks/items, identify semantic duplicates and merge them into one item (use concise wording).\n"
        "2. Remove filler, redundant, or repeated information. If the same item is mentioned multiple times in different words, leave only one clear version.\n"
        "3. For checklists, group related items and sort them logically (by category, time, or relevance, if applicable).\n"
        "4. If there is a general summary or context, include it before the list.\n"
        "5. Always return ONLY valid JSON with two keys:\n"
        "   - \"datetime\": the time for the reminder in ISO 8601 format: YYYY-MM-DD HH:MM:SS (interpret relative expressions like \"tomorrow\", \"in 5 days\" or \"every Friday at 9:00\" using the current Moscow time above). "
        "IMPORTANT: If the user DID NOT clearly specify a date, time, or a relative term ('завтра', 'через 2 дня', '16:00', '22 мая', etc), ALWAYS return an empty string \"\" for 'datetime' and DO NOT invent or guess any date/time!\n"
        "   - \"reminder_text\": a concise, well-formatted reminder message in Russian. If input was a list, produce an organized checklist, not just a plain string.\n"
        "Example output for a checklist:\n"
        "{\n"
        "  \"datetime\": \"2025-05-18 15:00:00\",\n"
        "  \"reminder_text\": \"Позвонить врачу, купить продукты (молоко, хлеб, яйца), подготовить документы для отчёта.\"\n"
        "}\n"
        "If there is NO explicit date or time, output:\n"
        "{\n"
        "  \"datetime\": \"\",\n"
        "  \"reminder_text\": \"Проверить апдейты\"\n"
        "}\n"
        "Do NOT include your reasoning, comments, or anything other than correct JSON. If the information entered by the user is ambiguous, clarify it by generating the most likely structured version. If a list is implied, output the list with a line break.\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input_text}
    ]
    try:
        resp = openai.ChatCompletion.create(model=GPT_MODEL, messages=messages)
        content = resp["choices"][0]["message"]["content"].strip()
        import json
        parsed = json.loads(content)
        # Проверка: если нет даты — вернём специальный флаг
        if not parsed.get("datetime"):
            return None  # Или {}
        # Можно добавить другие проверки (на слишком короткий текст и т.п.)
        return parsed
    except Exception as e:
        logger.exception("Ошибка при парсинге напоминания")
        return None


def save_reminder(user_id: int, remind_time_str: str, reminder_text: str):
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO reminders (user_id, remind_time, text, done) VALUES (?, ?, ?, 0)",
        (user_id, remind_time_str, reminder_text)
    )
    conn.commit()
    conn.close()
    logger.info(f"Reminder saved for user {user_id}: {remind_time_str}, {reminder_text}")

async def check_reminders(bot: Bot):
    while True:
        try:
            now_dt = datetime.now()
            conn = sqlite3.connect(DATABASE_PATH)
            c = conn.cursor()
            c.execute(
                "SELECT id, user_id, remind_time, text FROM reminders "
                "WHERE done=0 AND datetime(remind_time) <= datetime(?)",
                (now_dt.strftime("%Y-%m-%d %H:%M:%S"),)
            )
            rows = c.fetchall()
            for row in rows:
                reminder_id, user_id, _, remind_text = row
                try:
                    await bot.send_message(
                        chat_id=user_id,
                        text=custom_escape_markdown(f"🔔Напоминание:\n{remind_text}"),
                        parse_mode=ParseMode.MARKDOWN
                    )
                except Exception as e:
                    logger.exception(f"Не удалось отправить напоминание user_id={user_id}: {e}")
                c.execute("UPDATE reminders SET done=1 WHERE id=?", (reminder_id,))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.exception(f"Ошибка в check_reminders: {e}")
        await asyncio.sleep(60)

# ===================== Ошибки =====================
async def global_error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Произошло исключение:", exc_info=context.error)
    if update and hasattr(update, "message") and update.message:
        try:
            await update.message.reply_text("Произошла внутренняя ошибка. Попробуйте позже.", parse_mode=ParseMode.MARKDOWN)
        except Exception:
            pass

# ===================== Команды =====================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not user:
        return
    logger.info(f"Пользователь: id={user.id}")
    if not is_user_authorized(user.id):
        await update.message.reply_text(
            custom_escape_markdown("Извините, у вас нет доступа к этому боту."),
            parse_mode=ParseMode.MARKDOWN
        )
        return
    user_context.setdefault(user.id, deque(maxlen=MAX_CONTEXT_LENGTH))
    await update.message.reply_text(
        custom_escape_markdown("Добро пожаловать! Отправьте сообщение, голос, фото, документ или ссылку – я обработаю его и отвечу с помощью GPT‑модели.\n\n"
                                 "Доступные команды:\n"
                                 "/reset – сбросить контекст\n"
                                 "/reminder – создать напоминание\n"
                                 "/search <запрос> — Быстрый поиск информации через Perplexity \n"
                                 "/myid – показать ваш ID\n"),
        parse_mode=ParseMode.MARKDOWN
    )

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not user:
        return
    if not is_user_authorized(user.id):
        await update.message.reply_text(
            custom_escape_markdown("Извините, у вас нет доступа к этому боту."),
            parse_mode=ParseMode.MARKDOWN
        )
        return
    user_context[user.id] = deque(maxlen=MAX_CONTEXT_LENGTH)
    await update.message.reply_text(
        custom_escape_markdown("Контекст диалога сброшен."),
        parse_mode=ParseMode.MARKDOWN
    )

async def myid(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if user:
        await update.message.reply_text(
            custom_escape_markdown(f"Ваш user_id: {user.id}"),
            parse_mode=ParseMode.MARKDOWN
        )

async def adduser(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not user or user.id not in ADMIN_IDS:
        await update.message.reply_text(
            custom_escape_markdown("У вас нет прав для выполнения этой команды."),
            parse_mode=ParseMode.MARKDOWN
        )
        return
    if not context.args:
        await update.message.reply_text(
            custom_escape_markdown("Используйте: /adduser <user_id>"),
            parse_mode=ParseMode.MARKDOWN
        )
        return
    try:
        target_id = int(context.args[0])
        add_authorized_user(target_id)
        await update.message.reply_text(
            custom_escape_markdown(f"Пользователь {target_id} добавлен и получил доступ."),
            parse_mode=ParseMode.MARKDOWN
        )
    except Exception as e:
        await update.message.reply_text(
            custom_escape_markdown(f"Ошибка: {e}"),
            parse_mode=ParseMode.MARKDOWN
        )

async def removeuser(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not user or user.id not in ADMIN_IDS:
        await update.message.reply_text(
            custom_escape_markdown("У вас нет прав для выполнения этой команды."),
            parse_mode=ParseMode.MARKDOWN
        )
        return
    if not context.args:
        await update.message.reply_text(
            custom_escape_markdown("Используйте: /removeuser <user_id>"),
            parse_mode=ParseMode.MARKDOWN
        )
        return
    try:
        target_id = int(context.args[0])
        remove_authorized_user(target_id)
        await update.message.reply_text(
            custom_escape_markdown(f"Доступ пользователя {target_id} отозван."),
            parse_mode=ParseMode.MARKDOWN
        )
    except Exception as e:
        await update.message.reply_text(
            custom_escape_markdown(f"Ошибка: {e}"),
            parse_mode=ParseMode.MARKDOWN
        )

async def reminder(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not user or not is_user_authorized(user.id):
        await update.message.reply_text(
            custom_escape_markdown("Извините, у вас нет доступа к этому боту."),
            parse_mode=ParseMode.MARKDOWN
        )
        return
    if context.args:
        raw_text = " ".join(context.args)
        parsed = await parse_reminder_text(raw_text)
        if not parsed or not parsed.get("datetime"):
            await update.message.reply_text(
                "Не удалось распознать дату или время напоминания. Пожалуйста, повторите запрос и обязательно укажите срок (например: *завтра в 16:00*, *через 3 дня* или *22 мая в 9:30*).",
                parse_mode=ParseMode.MARKDOWN
            )
            return

        dt_str = parsed["datetime"]
        reminder_txt = parsed["reminder_text"]
        save_reminder(user.id, dt_str, reminder_txt)
        await update.message.reply_text(
            custom_escape_markdown(f"Напоминание сохранено на {dt_str}: {reminder_txt}"),
            parse_mode=ParseMode.MARKDOWN
        )
    else:
        user_context.setdefault(user.id, deque(maxlen=MAX_CONTEXT_LENGTH))
        user_context[user.id].append({"role": "system", "content": "/reminder_no_args"})
        await update.message.reply_text(
            custom_escape_markdown("Хорошо, пришлите текст или голосовое сообщение с описанием и датой напоминания."),
            parse_mode=ParseMode.MARKDOWN
        )

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not user or not update.message:
        return
    if not is_user_authorized(user.id):
        await update.message.reply_text(
            custom_escape_markdown("Извините, у вас нет доступа к этому боту."),
            parse_mode=ParseMode.MARKDOWN
        )
        return
    user_text = update.message.text.strip()
    if len(user_text) > MAX_TEXT_LENGTH:
        await update.message.reply_text(
            custom_escape_markdown("Слишком длинное сообщение. Пожалуйста, сократите."),
            parse_mode=ParseMode.MARKDOWN
        )
        return

    # ==== SEARCH QUEUE LOGIC ====
    if user.id in user_context and len(user_context[user.id]) > 0:
        last_msg = user_context[user.id][-1].get("content", "")
        if last_msg.startswith("/search_no_args"):
            prompt = user_text
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, search_perplexity, prompt)
            result = format_perplexity_markdown_for_telegram(result)
            if len(result) > TELEGRAM_MAX_MESSAGE_LENGTH:
                result = result[:TELEGRAM_MAX_MESSAGE_LENGTH-100] + "\n\n[Ответ обрезан]"
            await send_long_message(update, context, result)
            user_context[user.id].append({"role": "system", "content": "Search done."})
            return

    # ==== REMINDER LOGIC ====
    if user.id in user_context and len(user_context[user.id]) > 0:
        last_msg = user_context[user.id][-1].get("content", "")
        if last_msg.startswith("/reminder_no_args"):
            parsed = await parse_reminder_text(user_text)
            if not parsed or not parsed.get("datetime"):
                await update.message.reply_text(
                    "Не удалось распознать срок для напоминания. Пожалуйста, повторите запрос и обязательно укажите дату или время (например: *завтра в 16:00*, *через 3 дня* или *20 мая в 9:30*).",
                    parse_mode=ParseMode.MARKDOWN
                )
                return

            dt_str = parsed["datetime"]
            reminder_txt = parsed["reminder_text"]
            save_reminder(user.id, dt_str, reminder_txt)
            await update.message.reply_text(
                custom_escape_markdown(f"Напоминание сохранено на {dt_str}:\n{reminder_txt}"),
                parse_mode=ParseMode.MARKDOWN
            )
            user_context[user.id].append({"role": "system", "content": "Reminder set."})
            return

    logger.info(f"Пользователь {user.id}: {user_text}")
    if re.match(r"^https?://", user_text):
        if len(user_text) > MAX_URL_LENGTH:
            await update.message.reply_text(
                custom_escape_markdown("Ссылка слишком длинная, отказываюсь обрабатывать."),
                parse_mode=ParseMode.MARKDOWN
            )
            return
        await update.message.reply_text(
            custom_escape_markdown("Ваш запрос поставлен в очередь на обработку..."),
            parse_mode=ParseMode.MARKDOWN
        )
        await call_gpt_api(user.id, user_text, update, context)
            
        page_text = await fetch_url_content(user_text)
        if page_text:
            user_context.setdefault(user.id, deque(maxlen=MAX_CONTEXT_LENGTH))
            user_context[user.id].append({"role": "user", "content": page_text})
            logger.info(f"Контекст для {user.id} обновлён, длина: {len(user_context[user.id])}")
            await update.message.reply_text(
                custom_escape_markdown("Содержимое страницы добавлено в контекст. Теперь задавайте вопросы по нему."),
                parse_mode=ParseMode.MARKDOWN
            )
        else:
            await update.message.reply_text(
                custom_escape_markdown("Не удалось извлечь содержимое страницы."),
                parse_mode=ParseMode.MARKDOWN
            )
        return
    await update.message.reply_text(
        custom_escape_markdown("Ваш запрос поставлен в очередь на обработку..."),
        parse_mode=ParseMode.MARKDOWN
    )
    await call_gpt_api(user.id, user_text, update, context)

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not user or not update.message:
        return
    if not is_user_authorized(user.id):
        await update.message.reply_text(
            custom_escape_markdown("Извините, у вас нет доступа к этому боту."),
            parse_mode=ParseMode.MARKDOWN
        )
        return
    voice = update.message.voice
    if not voice:
        return
    if voice.file_size and voice.file_size > MAX_FILE_SIZE:
        await update.message.reply_text(
            custom_escape_markdown("Файл голосового сообщения слишком большой."),
            parse_mode=ParseMode.MARKDOWN
        )
        return
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    logger.info(f"Получено голосовое сообщение от пользователя {user.id}")
    try:
        file = await context.bot.get_file(voice.file_id)
        file_path = os.path.join(BASE_DIR, f"voice_{user.id}.ogg")
        await file.download_to_drive(custom_path=file_path)
        with open(file_path, "rb") as audio_file:
            transcript = await asyncio.to_thread(
                openai.Audio.transcribe, "whisper-1", audio_file
            )
            user_text = transcript["text"].strip()
        os.remove(file_path)
        if len(user_text) > MAX_TEXT_LENGTH:
            await update.message.reply_text(
                custom_escape_markdown("Слишком длинный распознанный текст."),
                parse_mode=ParseMode.MARKDOWN
            )
            return
        await update.message.reply_text(
            custom_escape_markdown(f"Распознан текст: {user_text}"),
            parse_mode=ParseMode.MARKDOWN
        )
        logger.info(f"Распознан текст: {user_text}")
        if user.id in user_context and len(user_context[user.id]) > 0:
            last_msg = user_context[user.id][-1].get("content", "")


        # ==== SEARCH QUEUE LOGIC ====
        if user.id in user_context and len(user_context[user.id]) > 0:
            last_msg = user_context[user.id][-1].get("content", "")
            if last_msg.startswith("/search_no_args"):
                prompt = user_text
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, search_perplexity, prompt)
                result = format_perplexity_markdown_for_telegram(result)
                if len(result) > TELEGRAM_MAX_MESSAGE_LENGTH:
                    result = result[:TELEGRAM_MAX_MESSAGE_LENGTH-100] + "\n\n[Ответ обрезан]"
                await send_long_message(update, context, result)
                user_context[user.id].append({"role": "system", "content": "Search done."})
                return

        if user.id in user_context and len(user_context[user.id]) > 0:
            last_msg = user_context[user.id][-1].get("content", "")
            if last_msg.startswith("/reminder_no_args"):
                parsed = await parse_reminder_text(user_text)
                if not parsed or "datetime" not in parsed or "reminder_text" not in parsed:
                    await update.message.reply_text(
                        custom_escape_markdown("Не удалось понять дату/время или текст напоминания. Повторите запрос."),
                        parse_mode=ParseMode.MARKDOWN
                    )
                    return
                dt_str = parsed["datetime"]
                reminder_txt = parsed["reminder_text"]
                save_reminder(user.id, dt_str, reminder_txt)
                await update.message.reply_text(
                    custom_escape_markdown(f"Напоминание сохранено на {dt_str}: {reminder_txt}"),
                    parse_mode=ParseMode.MARKDOWN
                )
                user_context[user.id].append({"role": "system", "content": "Reminder set."})
                return
        await update.message.reply_text(
            custom_escape_markdown("Ваш запрос (голос -> текст) поставлен в очередь."),
            parse_mode=ParseMode.MARKDOWN
        )
        await call_gpt_api(user.id, user_text, update, context)
    except Exception as e:
        logger.exception("Ошибка при обработке голосового сообщения")
        await update.message.reply_text(
            custom_escape_markdown(f"Ошибка при обработке голосового сообщения: {e}"),
            parse_mode=ParseMode.MARKDOWN
        )

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not user or not update.message:
        return
    if not is_user_authorized(user.id):
        await update.message.reply_text(
            custom_escape_markdown("Извините, у вас нет доступа к этому боту."),
            parse_mode=ParseMode.MARKDOWN
        )
        return
    doc = update.message.document
    if not doc:
        return
    if doc.file_size and doc.file_size > MAX_FILE_SIZE:
        await update.message.reply_text(
            custom_escape_markdown("Файл слишком большой, максимальный размер 10MB."),
            parse_mode=ParseMode.MARKDOWN
        )
        return
    mime_type = doc.mime_type or ""
    file_name = doc.file_name or "document"
    ext = os.path.splitext(file_name)[1].lower()
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    logger.info(f"Получен документ {file_name} от пользователя {user.id}")
    try:
        file = await context.bot.get_file(doc.file_id)
        local_path = os.path.join(BASE_DIR, f"doc_{user.id}{ext}")
        await file.download_to_drive(custom_path=local_path)
        extracted_text = ""
        if ext == ".pdf":
            logger.info("Обрабатываем PDF документ локально.")
            extracted_text = process_pdf_local(local_path)
        elif ext in [".doc", ".docx"]:
            logger.info("Обрабатываем DOC/DOCX документ.")
            extracted_text = process_docx_local(local_path)
        elif ext in [".xls", ".xlsx"]:
            logger.info("Обрабатываем Excel документ.")
            extracted_text = process_excel_local(local_path)
        elif mime_type.startswith("image/"):
            logger.info("Обрабатываем присланный файл как изображение.")
            with open(local_path, "rb") as f:
                file_bytes = f.read()
            os.remove(local_path)
            encoded_data = base64.b64encode(file_bytes).decode("utf-8")
            guessed_mime, _ = mimetypes.guess_type(local_path)
            if not guessed_mime:
                guessed_mime = "image/jpeg"
            image_data_url = f"data:{guessed_mime};base64,{encoded_data}"
            content = []
            if update.message.caption:
                content.append({"type": "text", "text": update.message.caption})
            content.append({"type": "image_url", "image_url": {"url": image_data_url}})
            reply = await call_gpt_api_multimodal(user.id, content)
            await send_long_message(update, context, reply)
            return
        else:
            await update.message.reply_text(
                custom_escape_markdown("Поддерживаются только PDF, DOC/DOCX, XLS/XLSX или изображения."),
                parse_mode=ParseMode.MARKDOWN
            )
            os.remove(local_path)
            return
        os.remove(local_path)
        if extracted_text and extracted_text.strip():
            user_context.setdefault(user.id, deque(maxlen=MAX_CONTEXT_LENGTH))
            user_context[user.id].append({"role": "user", "content": extracted_text})
            # <--- ВАЖНО! Обрезаем контекст до последних 2 сообщений
            trim_user_context_after_file(user.id, keep_last_n=2)
            await update.message.reply_text(
                custom_escape_markdown("Файл обработан, содержимое добавлено в контекст! Теперь задавайте вопросы по документу."),
                parse_mode=ParseMode.MARKDOWN
            )
        else:
            await update.message.reply_text(
                custom_escape_markdown("Из файла не удалось извлечь текст."),
                parse_mode=ParseMode.MARKDOWN
            )
    except Exception as e:
        logger.exception("Ошибка при обработке документа")
        await update.message.reply_text(
            custom_escape_markdown(f"Ошибка при обработке документа: {e}"),
            parse_mode=ParseMode.MARKDOWN
        )

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not user or not update.message:
        return
    if not is_user_authorized(user.id):
        await update.message.reply_text(
            custom_escape_markdown("Извините, у вас нет доступа к этому боту."),
            parse_mode=ParseMode.MARKDOWN
        )
        return
    photo = update.message.photo[-1]
    if photo.file_size and photo.file_size > MAX_FILE_SIZE:
        await update.message.reply_text(
            custom_escape_markdown("Фото слишком большое, максимальный размер 10MB."),
            parse_mode=ParseMode.MARKDOWN
        )
        return
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    logger.info(f"Получено фото от пользователя {user.id}")
    try:
        file = await context.bot.get_file(photo.file_id)
        file_path = os.path.join(BASE_DIR, f"photo_{user.id}.jpg")
        await file.download_to_drive(custom_path=file_path)
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        os.remove(file_path)
        encoded_data = base64.b64encode(file_bytes).decode("utf-8")
        guessed_mime, _ = mimetypes.guess_type(file_path)
        if not guessed_mime:
            guessed_mime = "image/jpeg"
        image_data_url = f"data:{guessed_mime};base64,{encoded_data}"
        content = []
        if update.message.caption:
            content.append({"type": "text", "text": update.message.caption})
        content.append({"type": "image_url", "image_url": {"url": image_data_url}})
        # --- Вот здесь вызывается НОВЫЙ call_gpt_api_multимodal:
        reply = await call_gpt_api_multimodal(user.id, content)
        await send_long_message(update, context, reply)
    except Exception as e:
        logger.exception("Ошибка при обработке фото")
        await update.message.reply_text(
            custom_escape_markdown(f"Ошибка при обработке фото: {e}"),
            parse_mode=ParseMode.MARKDOWN
        )


async def main():
    init_db()
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(CommandHandler("myid", myid))
    app.add_handler(CommandHandler("adduser", adduser))
    app.add_handler(CommandHandler("removeuser", removeuser))
    app.add_handler(CommandHandler("reminder", reminder))
    app.add_handler(CommandHandler("search", search_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_error_handler(global_error_handler)
    asyncio.create_task(process_gpt_queue())
    asyncio.create_task(check_reminders(app.bot))
    logger.info("Бот запущен.")
    await app.run_polling()

if __name__ == '__main__':
    asyncio.run(main())
