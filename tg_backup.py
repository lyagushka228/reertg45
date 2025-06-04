import os
import re
import logging
import sqlite3
import base64
import mimetypes
from collections import deque
import nest_asyncio
nest_asyncio.apply()
import requests
import time
import random
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import logging


import asyncio
import requests  # для HTTP-запросов
from bs4 import BeautifulSoup  # для парсинга HTML

import openai
from PyPDF2 import PdfReader  # новый интерфейс для чтения PDF
from docx import Document    # для работы с DOCX
import pandas as pd

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# === Функция экранирования для MarkdownV2 ===
def custom_escape_markdown_v2(text: str) -> str:
    """
    Экранирует символы, зарезервированные в MarkdownV2, если они не экранированы.
    Зарезервированные символы: _ * [ ] ( ) ~ ` > # + - = | { } . !
    """
    pattern = r'(?<!\\)([_*\[\]\(\)~`>#+\-=|{}\.!])'
    return re.sub(pattern, r'\\\1', text)

# ===================== Конфигурация =====================
# Задайте токены и ключи (например, через переменные окружения)
# TELEGRAM_BOT_TOKEN = os.environ.get("5625888369:AAFzEi7HqV4mgJTXMHh8_8NnegC9TAjy_7I")  # токен телеграм-бота
TELEGRAM_BOT_TOKEN = "5625888369:AAFzEi7HqV4mgJTXMHh8_8NnegC9TAjy_7I"
OPENAI_API_KEY = "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"     # API-ключ OpenAI
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY не установлен в переменных окружения!")
openai.api_key = OPENAI_API_KEY
# Выбранная модель GPT (укажите нужный snapshot)
GPT_MODEL = "o3-mini-2025-01-31"
# Пути и лимиты
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, "users.db")
MAX_CONTEXT_LENGTH = 20              # Максимальное число сообщений в истории диалога
TELEGRAM_MAX_MESSAGE_LENGTH = 4096   # Лимит символов для одного сообщения Telegram

# Глобальное хранилище контекста: user_id -> deque сообщений
user_context = {}

# Модели: текстовая и мультимодальная
CURRENT_MODEL = "o3-mini-2025-01-31"            # Текстовая модель (только текст, но "умнее")
MULTIMODAL_MODEL = "gpt-4o-mini-2024-07-18"         # Мультимодальная (поддерживает файлы и картинки)


ADMIN_ID = 84544725
# ========================================================

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ===================== Функция разбиения текста на части =====================
def split_text(text: str, max_length: int = TELEGRAM_MAX_MESSAGE_LENGTH) -> list:
    parts = []
    while len(text) > max_length:
        split_index = text.rfind('\n', 0, max_length)
        if split_index == -1:
            split_index = max_length
        backslash_count = 0
        i = split_index - 1
        while i >= 0 and text[i] == '\\':
            backslash_count += 1
            i -= 1
        if backslash_count % 2 == 1 and split_index < len(text):
            split_index += 1
        parts.append(text[:split_index])
        text = text[split_index:]
    if text:
        parts.append(text)
    return parts

# ===================== Безопасная отправка сообщения =====================
async def safe_reply_text(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str):
    try:
        await update.message.reply_text(
            text,
            parse_mode=ParseMode.MARKDOWN_V2,
            disable_web_page_preview=True
        )
    except Exception as ex:
        logger.exception("Ошибка отправки с MarkdownV2, отправляем без форматирования")
        await update.message.reply_text(text, disable_web_page_preview=True)

# ===================== Отправка длинного сообщения =====================
async def send_long_message(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str):
    # Если текст начинается с блока кода, отправляем как есть; иначе — экранируем
    if text.lstrip().startswith("```"):
        parts = split_text(text)
    else:
        parts = split_text(custom_escape_markdown_v2(text))
    for part in parts:
        await safe_reply_text(update, context, part)

# ===================== Функция для извлечения содержимого веб-страницы =====================
def fetch_with_playwright(url: str) -> str:
    """Попытка получить содержимое страницы с помощью Playwright (головного браузера)."""
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            # Задаём дополнительный timeout (например, 30 секунд)
            page.goto(url, timeout=30000)
            content = page.content()
            browser.close()
            soup = BeautifulSoup(content, "html.parser")
            # Удаляем скрипты и стили
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text(separator="\n")
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            cleaned_text = "\n".join(lines)
            return cleaned_text
    except Exception as e:
        logger.exception(f"Ошибка в fetch_with_playwright для {url}: {e}")
        raise

def fetch_url_content(url: str) -> str:
    """
    Пытается получить содержимое веб-страницы.
    Сначала через requests с корректными заголовками и повторными попытками.
    При неудаче – переходит на получение через Playwright.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/92.0.4515.159 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5"
    }
    
    # Пытаемся выполнить запрос через requests до 3 раз
    for attempt in range(3):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                # Удаляем теги script и style
                for script in soup(["script", "style"]):
                    script.decompose()
                text = soup.get_text(separator="\n")
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                cleaned_text = "\n".join(lines)
                # Если извлечённого текста недостаточно, пробуем fallback
                if len(cleaned_text) < 50:
                    logger.warning(f"Слишком мало текста извлечено с {url}, fallback с Playwright")
                    raise Exception("Мало содержимого")
                logger.info(f"Длина извлечённого текста с {url}: {len(cleaned_text)} символов")
                return cleaned_text
            elif response.status_code in (403, 429):
                logger.error(f"Ошибка загрузки страницы {url}: статус {response.status_code}")
                time.sleep(random.uniform(2, 5))
            else:
                logger.error(f"Неожиданный статус {response.status_code} для {url}")
                time.sleep(random.uniform(2, 5))
        except Exception as e:
            logger.exception(f"Ошибка при загрузке URL {url} на попытке {attempt+1}: {e}")
            time.sleep(random.uniform(2, 5))
    
    # Если через requests не получилось, используем Playwright как резервный вариант
    try:
        logger.info(f"Fallback: попытка получить содержимое {url} с помощью Playwright")
        text = fetch_with_playwright(url)
        logger.info(f"Длина извлечённого текста с Playwright: {len(text)} символов")
        return text
    except Exception as e:
        logger.exception(f"Fallback с использованием Playwright не удался для {url}: {e}")
        return ""


# ===================== Извлечение текста из файлов =====================
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
        df = pd.read_excel(file_path)
        return df.to_csv(index=False)
    except Exception as e:
        logger.exception("Ошибка при чтении Excel")
        return f"Ошибка при чтении Excel: {e}"

# ===================== Работа с базой данных =====================
def init_db():
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS authorized_users (
            id INTEGER PRIMARY KEY,
            allowed INTEGER NOT NULL DEFAULT 0
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

# ===================== GPT API =====================
def call_gpt_api(user_id: int, user_message: str) -> str:
    if user_id not in user_context:
        user_context[user_id] = deque(maxlen=MAX_CONTEXT_LENGTH)
    messages = user_context[user_id]
    messages.append({"role": "user", "content": user_message})
    try:
        response = openai.ChatCompletion.create(
            model=CURRENT_MODEL,
            messages=list(messages)
        )
        reply = response["choices"][0]["message"]["content"].strip()
        messages.append({"role": "assistant", "content": reply})
        return reply
    except Exception as e:
        logger.exception("Ошибка при вызове OpenAI API")
        return f"Ошибка при обращении к GPT: {e}"

def call_gpt_api_multimodal(user_id: int, content: list) -> str:
    if user_id not in user_context:
        user_context[user_id] = deque(maxlen=MAX_CONTEXT_LENGTH)
    messages = user_context[user_id]
    messages.append({"role": "user", "content": content})
    try:
        response = openai.ChatCompletion.create(
            model=CURRENT_MODEL,
            messages=list(messages)
        )
        reply = response["choices"][0]["message"]["content"].strip()
        messages.append({"role": "assistant", "content": reply})
        return reply
    except Exception as e:
        logger.exception("Ошибка при вызове OpenAI API (мультимодальный)")
        return f"Ошибка при обращении к GPT: {e}"

# ===================== Команды =====================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not user:
        return
    logger.info(f"Пользователь: id={user.id}")
    if not is_user_authorized(user.id):
        await update.message.reply_text(
            custom_escape_markdown_v2("Извините, у вас нет доступа к этому боту."),
            parse_mode=ParseMode.MARKDOWN_V2
        )
        return
    if user.id not in user_context:
        user_context[user.id] = deque(maxlen=MAX_CONTEXT_LENGTH)
    await update.message.reply_text(
        custom_escape_markdown_v2("Добро пожаловать! Отправьте сообщение, голос, фото, документ или ссылку – я обработаю его и отвечу с помощью GPT‑модели."),
        parse_mode=ParseMode.MARKDOWN_V2
    )

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not user:
        return
    if not is_user_authorized(user.id):
        await update.message.reply_text(
            custom_escape_markdown_v2("Извините, у вас нет доступа к этому боту."),
            parse_mode=ParseMode.MARKDOWN_V2
        )
        return
    user_context[user.id] = deque(maxlen=MAX_CONTEXT_LENGTH)
    await update.message.reply_text(
        custom_escape_markdown_v2("Контекст диалога сброшен"),
        parse_mode=ParseMode.MARKDOWN_V2
    )

async def myid(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if user:
        await update.message.reply_text(
            custom_escape_markdown_v2(f"Ваш user_id: {user.id}"),
            parse_mode=ParseMode.MARKDOWN_V2
        )

async def switch_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global CURRENT_MODEL
    user_id = update.effective_user.id
    if CURRENT_MODEL == MULTIMODAL_MODEL:
        multi_present = False
        if user_id in user_context:
            for msg in user_context[user_id]:
                if not isinstance(msg.get("content"), str):
                    multi_present = True
                    break
        if multi_present:
            await update.message.reply_text(
                custom_escape_markdown_v2("В диалоге уже присутствуют мультимодальные данные. Для переключения на текстовую модель используйте /reset"),
                parse_mode=ParseMode.MARKDOWN_V2
            )
            return
        else:
            CURRENT_MODEL = "o3-mini-2025-01-31"
            await update.message.reply_text(
                custom_escape_markdown_v2("Переключено на модель o3-mini-2025-01-31 (только текст, но умнее)"),
                parse_mode=ParseMode.MARKDOWN_V2
            )
    else:
        CURRENT_MODEL = MULTIMODAL_MODEL
        await update.message.reply_text(
            custom_escape_markdown_v2("Переключено на модель gpt-4o-mini-2024-07-18 (поддерживает файлы, картинки, PDF, DOCX, Excel)"),
            parse_mode=ParseMode.MARKDOWN_V2
        )

async def adduser(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not user or user.id != ADMIN_ID:
        await update.message.reply_text(
            custom_escape_markdown_v2("У вас нет прав для выполнения этой команды."),
            parse_mode=ParseMode.MARKDOWN_V2
        )
        return
    if not context.args:
        await update.message.reply_text(
            custom_escape_markdown_v2("Используйте: /adduser <user_id>"),
            parse_mode=ParseMode.MARKDOWN_V2
        )
        return
    try:
        target_id = int(context.args[0])
        add_authorized_user(target_id)
        await update.message.reply_text(
            custom_escape_markdown_v2(f"Пользователь {target_id} добавлен и получил доступ"),
            parse_mode=ParseMode.MARKDOWN_V2
        )
    except Exception as e:
        await update.message.reply_text(
            custom_escape_markdown_v2(f"Ошибка: {e}"),
            parse_mode=ParseMode.MARKDOWN_V2
        )

async def removeuser(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not user or user.id != ADMIN_ID:
        await update.message.reply_text(
            custom_escape_markdown_v2("У вас нет прав для выполнения этой команды."),
            parse_mode=ParseMode.MARKDOWN_V2
        )
        return
    if not context.args:
        await update.message.reply_text(
            custom_escape_markdown_v2("Используйте: /removeuser <user_id>"),
            parse_mode=ParseMode.MARKDOWN_V2
        )
        return
    try:
        target_id = int(context.args[0])
        remove_authorized_user(target_id)
        await update.message.reply_text(
            custom_escape_markdown_v2(f"Доступ пользователя {target_id} отозван"),
            parse_mode=ParseMode.MARKDOWN_V2
        )
    except Exception as e:
        await update.message.reply_text(
            custom_escape_markdown_v2(f"Ошибка: {e}"),
            parse_mode=ParseMode.MARKDOWN_V2
        )

# ===================== Обработка текстовых сообщений =====================
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not user or not update.message:
        return
    if not is_user_authorized(user.id):
        await update.message.reply_text(
            custom_escape_markdown_v2("Извините, у вас нет доступа к этому боту."),
            parse_mode=ParseMode.MARKDOWN_V2
        )
        return
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    user_text = update.message.text.strip()
    logger.info(f"Пользователь {user.id}: {user_text}")

    # Если сообщение выглядит как ссылка, переходим по ней и добавляем содержимое в контекст
    if re.match(r"^https?://", user_text):
        page_text = fetch_url_content(user_text)
        if page_text:
            if user.id not in user_context:
                user_context[user.id] = deque(maxlen=MAX_CONTEXT_LENGTH)
            user_context[user.id].append({"role": "user", "content": page_text})
            logger.info(f"Контекст для {user.id} обновлён, длина: {len(user_context[user.id])}")
            await update.message.reply_text(
                custom_escape_markdown_v2("Содержимое страницы добавлено в контекст. Теперь задавайте вопросы по нему."),
                parse_mode=ParseMode.MARKDOWN_V2
            )
        else:
            await update.message.reply_text(
                custom_escape_markdown_v2("Не удалось извлечь содержимое страницы."),
                parse_mode=ParseMode.MARKDOWN_V2
            )
        return

    # Обычная обработка текстовых сообщений
    reply = call_gpt_api(user.id, user_text)
    await send_long_message(update, context, reply)

# ===================== Обработка голосовых сообщений =====================
async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not user or not update.message:
        return
    if not is_user_authorized(user.id):
        await update.message.reply_text(
            custom_escape_markdown_v2("Извините, у вас нет доступа к этому боту."),
            parse_mode=ParseMode.MARKDOWN_V2
        )
        return
    voice = update.message.voice
    if not voice:
        return
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    logger.info(f"Получено голосовое сообщение от пользователя {user.id}")
    try:
        file = await context.bot.get_file(voice.file_id)
        file_path = os.path.join(BASE_DIR, f"voice_{user.id}.ogg")
        await file.download_to_drive(custom_path=file_path)
        with open(file_path, "rb") as audio_file:
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
            user_text = transcript["text"].strip()
        os.remove(file_path)
        await update.message.reply_text(
            custom_escape_markdown_v2(f"Распознан текст: {user_text}"),
            parse_mode=ParseMode.MARKDOWN_V2
        )
        logger.info(f"Распознан текст: {user_text}")
        reply = call_gpt_api(user.id, user_text)
        await send_long_message(update, context, reply)
    except Exception as e:
        logger.exception("Ошибка при обработке голосового сообщения")
        await update.message.reply_text(
            custom_escape_markdown_v2(f"Ошибка при обработке голосового сообщения: {e}"),
            parse_mode=ParseMode.MARKDOWN_V2
        )

# ===================== Обработка документов и файлов =====================
async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Обрабатывает файлы: PDF, DOC/DOCX, XLS/XLSX и изображения.
    Если файл поддерживается, его текстовое содержимое извлекается и добавляется в контекст.
    """
    user = update.effective_user
    if not user or not update.message:
        return
    if not is_user_authorized(user.id):
        await update.message.reply_text(
            custom_escape_markdown_v2("Извините, у вас нет доступа к этому боту."),
            parse_mode=ParseMode.MARKDOWN_V2
        )
        return

    doc = update.message.document
    if not doc:
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
            reply = call_gpt_api_multimodal(user.id, content)
            await send_long_message(update, context, reply)
            return
        else:
            await update.message.reply_text(
                custom_escape_markdown_v2("Поддерживаются только PDF, DOC/DOCX, XLS/XLSX или изображения."),
                parse_mode=ParseMode.MARKDOWN_V2
            )
            os.remove(local_path)
            return

        os.remove(local_path)
        if extracted_text and extracted_text.strip():
            # Добавляем извлечённый текст в контекст
            user_context.setdefault(user.id, deque(maxlen=MAX_CONTEXT_LENGTH))
            user_context[user.id].append({"role": "user", "content": extracted_text})
            await update.message.reply_text(
                custom_escape_markdown_v2("Файл обработан и его содержимое добавлено в контекст! Теперь задавайте вопросы по документу."),
                parse_mode=ParseMode.MARKDOWN_V2
            )
        else:
            await update.message.reply_text(
                custom_escape_markdown_v2("Из файла не удалось извлечь текст."),
                parse_mode=ParseMode.MARKDOWN_V2
            )
    except Exception as e:
        logger.exception("Ошибка при обработке документа")
        await update.message.reply_text(
            custom_escape_markdown_v2(f"Ошибка при обработке документа: {e}"),
            parse_mode=ParseMode.MARKDOWN_V2
        )

# ===================== Обработка фотографий =====================
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not user or not update.message:
        return
    if not is_user_authorized(user.id):
        await update.message.reply_text(
            custom_escape_markdown_v2("Извините, у вас нет доступа к этому боту."),
            parse_mode=ParseMode.MARKDOWN_V2
        )
        return
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    logger.info(f"Получено фото от пользователя {user.id}")
    try:
        photo = update.message.photo[-1]
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
        reply = call_gpt_api_multimodal(user.id, content)
        await send_long_message(update, context, reply)
    except Exception as e:
        logger.exception("Ошибка при обработке фото")
        await update.message.reply_text(
            custom_escape_markdown_v2(f"Ошибка при обработке фото: {e}"),
            parse_mode=ParseMode.MARKDOWN_V2
        )

# ===================== Основная функция =====================
async def main():
    init_db()
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(CommandHandler("myid", myid))
    app.add_handler(CommandHandler("adduser", adduser))
    app.add_handler(CommandHandler("removeuser", removeuser))
    app.add_handler(CommandHandler("switch", switch_model))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))

    logger.info("Бот запущен.")
    await app.run_polling()

if __name__ == '__main__':
    asyncio.run(main())