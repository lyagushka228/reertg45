FROM python:3.11-slim

# Создаём рабочую директорию
WORKDIR /app

# Копируем файлы зависимостей отдельно для кэширования
COPY requirements.txt .

# Устанавливаем системные зависимости для lxml и playwright
RUN apt-get update && \
    apt-get install -y gcc libxml2-dev libxslt1-dev libsqlite3-dev \
    curl unzip && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    python -m playwright install --with-deps chromium && \
    rm -rf /var/lib/apt/lists/*

# Копируем все исходники, включая .env, .py, users.db, logs
COPY . .

# Переменная окружения для Python
ENV PYTHONUNBUFFERED=1

# Открываем порт для бота (если надо, иначе строку можно убрать)
EXPOSE 8080

# Запуск бота (замени на tg_backup.py, если нужно)
CMD ["python", "tg.py"]
