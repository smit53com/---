# -*- coding: utf-8 -*-
import logging
import re
import math
import datetime as dt
import json
import os
from io import BytesIO
from typing import Dict, Any, List, Tuple, Optional

import pytz
from timezonefinder import TimezoneFinder
from geopy.geocoders import Nominatim

import swisseph as swe
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    filters, ConversationHandler, ContextTypes, CallbackQueryHandler
)
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv
import sqlite3
from contextlib import contextmanager

# ================== НАСТРОЙКИ ==================
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not TELEGRAM_TOKEN or not OPENAI_API_KEY:
    raise ValueError("❌ Не установлены переменные окружения TELEGRAM_TOKEN или OPENAI_API_KEY!")

client = OpenAI(api_key=OPENAI_API_KEY)
geolocator = Nominatim(user_agent="astro_bot_pro_v2")
tfinder = TimezoneFinder()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('astro_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Состояния диалога
BIRTH_DATA, CURRENT_PLACE, GENDER, FAMILY_STATUS = range(4)

# База данных
DB_NAME = "astro_users.db"

@contextmanager
def get_db():
    """Контекстный менеджер для безопасной работы с БД."""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    """Инициализация базы данных."""
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                profile TEXT NOT NULL,
                daily_off INTEGER DEFAULT 0,
                last_daily_sent TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
    logger.info("✅ База данных инициализирована")

def save_profile(user_id: int, profile: Dict[str, Any]):
    """Сохранение профиля пользователя в БД."""
    profile_json = json.dumps(profile, default=str)
    with get_db() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO users (user_id, profile, daily_off, last_daily_sent)
            VALUES (?, ?, ?, ?)
        """, (user_id, profile_json, profile.get('daily_off', 0), profile.get('last_daily_sent')))
        conn.commit()

def load_profile(user_id: int) -> Optional[Dict[str, Any]]:
    """Загрузка профиля пользователя из БД."""
    with get_db() as conn:
        row = conn.execute("SELECT profile FROM users WHERE user_id = ?", (user_id,)).fetchone()
        if row:
            profile = json.loads(row['profile'])
            # Восстановление datetime объектов
            if 'birth_time_utc' in profile:
                profile['birth_time_utc'] = dt.datetime.fromisoformat(profile['birth_time_utc'])
            return profile
    return None

def delete_profile(user_id: int):
    """Удаление профиля пользователя."""
    with get_db() as conn:
        conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
        conn.commit()

def get_all_users() -> List[Dict[str, Any]]:
    """Получение всех пользователей для рассылки."""
    with get_db() as conn:
        rows = conn.execute("""
            SELECT user_id, profile, daily_off, last_daily_sent 
            FROM users WHERE daily_off = 0
        """).fetchall()
        users = []
        for row in rows:
            profile = json.loads(row['profile'])
            profile['user_id'] = row['user_id']
            profile['last_daily_sent'] = row['last_daily_sent']
            if 'birth_time_utc' in profile:
                profile['birth_time_utc'] = dt.datetime.fromisoformat(profile['birth_time_utc'])
            users.append(profile)
        return users

def update_last_daily_sent(user_id: int, date_str: str):
    """Обновление даты последней отправки."""
    with get_db() as conn:
        conn.execute("UPDATE users SET last_daily_sent = ? WHERE user_id = ?", (date_str, user_id))
        conn.commit()

# ----------------- UI элементы -----------------
def kb_topics() -> InlineKeyboardMarkup:
    """Генерирует интерактивные кнопки."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("💼 Карьера", callback_data="career")],
        [InlineKeyboardButton("❤️ Любовь", callback_data="love")],
        [InlineKeyboardButton("🩺 Здоровье", callback_data="health")],
        [InlineKeyboardButton("🌱 Личностный рост", callback_data="growth")],
        [InlineKeyboardButton("♻️ Сменить данные", callback_data="change")],
        [InlineKeyboardButton("🚫 Отписаться", callback_data="stop_daily")],
    ])

# ----------------- Парсинг ввода -----------------
def try_parse_input(text: str) -> Tuple[dt.datetime, str]:
    """
    Парсит ввод: дата, время, место.
    Поддерживает: 'ДД.ММ.ГГГГ ЧЧ:ММ Город, Страна' или 'YYYY-MM-DD HH:MM Город, Страна'.
    """
    text = text.strip()
    # Формат ДД.ММ.ГГГГ
    m = re.match(r"^(\d{2}\.\d{2}\.\d{4})\s+(\d{2}:\d{2})\s+(.+)$", text)
    if m:
        date_str, time_str, place = m.groups()
        naive = dt.datetime.strptime(f"{date_str} {time_str}", "%d.%m.%Y %H:%M")
        return naive, place.strip()

    # Формат YYYY-MM-DD
    m2 = re.match(r"^(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2})\s+(.+)$", text)
    if m2:
        date_str, time_str, place = m2.groups()
        naive = dt.datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
        return naive, place.strip()

    raise ValueError("Неверный формат. Используйте: 'ДД.ММ.ГГГГ ЧЧ:ММ Город, Страна'")

def geocode(place: str) -> Tuple[float, float, str]:
    """Определяет геокоординаты и часовой пояс."""
    try:
        loc = geolocator.geocode(place, timeout=10)
        if not loc:
            raise ValueError("Город не найден")
        lat, lon = float(loc.latitude), float(loc.longitude)
        tz = tfinder.timezone_at(lat=lat, lng=lon) or "UTC"
        return lat, lon, tz
    except Exception as e:
        logger.error(f"Ошибка геокодирования '{place}': {e}")
        raise ValueError("Не удалось найти место. Попробуйте другой формат: 'Город, Страна'")

def local_naive_to_utc(naive: dt.datetime, tz_name: str) -> dt.datetime:
    """Переводит локальное время в UTC."""
    tz = pytz.timezone(tz_name)
    local_dt = tz.localize(naive)
    return local_dt.astimezone(pytz.UTC)

# ----------------- Астрологические расчёты -----------------
PLANETS = [
    ("Sun", swe.SUN), ("Moon", swe.MOON), ("Mercury", swe.MERCURY),
    ("Venus", swe.VENUS), ("Mars", swe.MARS), ("Jupiter", swe.JUPITER),
    ("Saturn", swe.SATURN), ("Uranus", swe.URANUS), ("Neptune", swe.NEPTUNE),
    ("Pluto", swe.PLUTO),
]

ASPECTS = [
    ("Conjunction", 0), ("Sextile", 60), ("Square", 90),
    ("Trine", 120), ("Quincunx", 150), ("Opposition", 180),
]
ORB = 2.0

def normalize_angle(a: float) -> float:
    return a % 360.0

def angle_diff(a: float, b: float) -> float:
    d = abs((a - b) % 360.0)
    return d if d <= 180.0 else 360.0 - d

def julian_day_utc(utc_dt: dt.datetime) -> float:
    return swe.julday(utc_dt.year, utc_dt.month, utc_dt.day,
                      utc_dt.hour + utc_dt.minute/60.0 + utc_dt.second/3600.0)

def calc_positions(utc_dt: dt.datetime) -> Dict[str, float]:
    """Расчет позиций планет."""
    jd = julian_day_utc(utc_dt)
    positions = {}
    for name, code in PLANETS:
        try:
            lonlatdist, _ = swe.calc_ut(jd, code, swe.FLG_SWIEPH | swe.FLG_SPEED)
            positions[name] = normalize_angle(lonlatdist[0])
        except Exception as e:
            logger.error(f"Ошибка расчета {name}: {e}")
            positions[name] = 0.0
    return positions

def calc_houses(utc_dt: dt.datetime, lat: float, lon: float) -> Tuple[List[float], float]:
    """Расчет домов (Placidus) и Asc."""
    try:
        jd = julian_day_utc(utc_dt)
        houses, ascmc = swe.houses(jd, lat, lon, b'P')
        return list(houses[1:13]), ascmc[0]
    except Exception as e:
        logger.error(f"Ошибка расчета домов: {e}")
        return [0.0]*12, 0.0

def calc_aspects(positions: Dict[str, float]) -> List[Dict[str, Any]]:
    """Расчет аспектов в карте."""
    lst = []
    names = list(positions.keys())
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            p1, p2 = names[i], names[j]
            ang = angle_diff(positions[p1], positions[p2])
            for aname, exact in ASPECTS:
                if abs(ang - exact) <= ORB:
                    lst.append({
                        "p1": p1, "p2": p2, "aspect": aname,
                        "angle": round(ang, 2), "delta": round(ang - exact, 2)
                    })
    return lst

def calc_transit_aspects(transit_pos: Dict[str, float], natal_pos: Dict[str, float]) -> List[Dict[str, Any]]:
    """Транзитные аспекты."""
    lst = []
    for t_name, t_lon in transit_pos.items():
        for n_name, n_lon in natal_pos.items():
            ang = angle_diff(t_lon, n_lon)
            for aname, exact in ASPECTS:
                if abs(ang - exact) <= ORB:
                    lst.append({
                        "transit_p": t_name, "natal_p": n_name, "aspect": aname,
                        "angle": round(ang, 2), "delta": round(ang - exact, 2)
                    })
    return lst

# ----------------- Визуализация -----------------
ZODIAC = ["Овен","Телец","Близнецы","Рак","Лев","Дева","Весы","Скорпион","Стрелец","Козерог","Водолей","Рыбы"]
PLANET_COLORS = {
    "Sun": "#ffcc00", "Moon": "#c0c0c0", "Mercury": "#4f83ff",
    "Venus": "#5ac18e", "Mars": "#ff5c5c", "Jupiter": "#a060ff",
    "Saturn": "#8c6239", "Uranus": "#2aa9ff", "Neptune": "#3eb5c7", "Pluto": "#d47fff"
}
ASPECT_COLORS = {
    "Conjunction": "#ffffff", "Sextile": "#66ccff", "Square": "#ff6666",
    "Trine": "#66ff66", "Quincunx": "#ffd166", "Opposition": "#ff99ff",
}

def draw_chart(positions: Dict[str, float], houses: List[float], 
               aspects: List[Dict[str, Any]], asc: float) -> BytesIO:
    """Рисует натальную карту."""
    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, projection='polar')
    ax.set_theta_direction(1)
    ax.set_theta_offset(math.pi)
    ax.set_yticks([])
    
    rising_sign_idx = int(asc / 30)
    zodiac_labels = ZODIAC[rising_sign_idx:] + ZODIAC[:rising_sign_idx]
    ax.set_xticks([math.radians(i*30) for i in range(12)])
    ax.set_xticklabels(zodiac_labels, fontsize=11)

    # Линии домов
    for cusp in houses:
        rel_theta = math.radians((cusp - asc + 360) % 360)
        ax.plot([rel_theta, rel_theta], [0.0, 1.0], linewidth=1, alpha=0.4, color="#888888")

    # Планеты с улучшенным размещением
    planet_points = {}
    sector_planets = {i: [] for i in range(12)}
    
    for name, lon in positions.items():
        rel_lon = (lon - asc + 360) % 360
        sector = int(rel_lon / 30)
        sector_planets[sector].append(name)
    
    for name, lon in positions.items():
        rel_lon = (lon - asc + 360) % 360
        sector = int(rel_lon / 30)
        sector_list = sector_planets[sector]
        idx = sector_list.index(name)
        
        # Улучшенное смещение для избежания наложений
        if len(sector_list) > 1:
            offset = (idx - len(sector_list)/2) * 0.08
        else:
            offset = 0
        
        theta = math.radians(rel_lon)
        r = 0.85 + offset
        planet_points[name] = (theta, r)
        ax.plot(theta, r, 'o', markersize=12, color=PLANET_COLORS.get(name, "#ffffff"))
        ax.text(theta, r + 0.05, name, fontsize=9, ha='center', 
                va='center', color=PLANET_COLORS.get(name, "#ffffff"), weight='bold')

    # Аспекты
    for a in aspects:
        p1, p2 = a["p1"], a["p2"]
        if p1 in planet_points and p2 in planet_points:
            th1, r1 = planet_points[p1]
            th2, r2 = planet_points[p2]
            ax.plot([th1, th2], [r1*0.9, r2*0.9], linewidth=1.2, alpha=0.7, 
                   color=ASPECT_COLORS.get(a["aspect"], "#ffffff"))

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=150, facecolor='#1a1a2e')
    buf.seek(0)
    plt.close(fig)
    return buf

# ----------------- GPT анализ с retry -----------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def gpt_analyze(question: str, profile: Dict[str, Any], current_data: Dict[str, Any] = None) -> str:
    """Генерирует анализ с помощью OpenAI GPT с повторными попытками."""
    positions = profile.get("positions", {})
    aspects = profile.get("aspects", [])
    birth_place = profile.get("birth_place")
    birth_tz = profile.get("birth_tz")
    birth_time_local = profile.get("birth_time_local_str")
    gender = profile.get("gender")
    status = profile.get("status")

    summary_aspects = "; ".join([f"{a['p1']}-{a['p2']} {a['aspect']} ({a['angle']}°)" 
                                  for a in aspects[:10]])

    # Получаем текущую дату
    now_tz = pytz.timezone(profile.get("now_tz", "UTC"))
    today = dt.datetime.now(now_tz)
    today_str = today.strftime("%d.%m.%Y (%A)")  # Например: 09.11.2025 (Saturday)
    
    # Русские названия дней недели
    days_ru = {
        'Monday': 'понедельник', 'Tuesday': 'вторник', 'Wednesday': 'среда',
        'Thursday': 'четверг', 'Friday': 'пятница', 'Saturday': 'суббота', 'Sunday': 'воскресенье'
    }
    day_name = today.strftime("%A")
    day_name_ru = days_ru.get(day_name, day_name)
    current_date_formatted = f"{today.strftime('%d.%m.%Y')} ({day_name_ru})"

    prompt = f"""Ты — профессиональный астролог. 

ТЕКУЩАЯ ДАТА: {current_date_formatted}
ВАЖНО: Сегодня именно {current_date_formatted}. Не используй другие даты!

Данные пользователя:

Пол: {gender}
Семейное положение: {status}
Место рождения: {birth_place} ({birth_tz})
Время: {birth_time_local}
Планеты: {positions}
Аспекты: {summary_aspects}
"""

    if current_data:
        current_positions = current_data.get("current_positions", {})
        transit_aspects = current_data.get("transit_aspects", [])
        summary_transit = "; ".join([f"{a['transit_p']}-{a['natal_p']} {a['aspect']}" 
                                     for a in transit_aspects[:8]])
        prompt += f"""
Текущие транзиты:
Место: {profile.get('now_place')}
Позиции: {current_positions}
Аспекты: {summary_transit}
"""

    prompt += f'\nОтветь на: "{question}"\nГовори по-русски, структурно, без воды.'

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Ты профессиональный астролог. Пиши ясно и конкретно."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Ошибка GPT: {e}")
        raise  # Retry сработает

# ----------------- Диалоговые хендлеры -----------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Начало диалога."""
    user_id = update.effective_user.id
    existing = load_profile(user_id)
    
    if existing:
        await update.message.reply_text(
            "У вас уже есть профиль! Используйте /change для изменения данных, "
            "или задавайте вопросы прямо сейчас.",
            reply_markup=kb_topics()
        )
        return ConversationHandler.END
    
    await update.message.reply_text(
        "👋 Привет! Отправьте данные рождения:\n\n"
        "`ДД.ММ.ГГГГ ЧЧ:ММ Город, Страна`\n\n"
        "Пример: `18.08.1995 14:30 Москва, Россия`",
        parse_mode="Markdown"
    )
    return BIRTH_DATA

async def get_birth_data(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка данных рождения."""
    text = update.message.text
    try:
        naive_dt, birth_place = try_parse_input(text)
        birth_lat, birth_lon, birth_tz = geocode(birth_place)
        birth_utc = local_naive_to_utc(naive_dt, birth_tz)

        context.user_data.clear()
        context.user_data.update({
            "birth_time_utc": birth_utc,
            "birth_time_local_str": f"{naive_dt.strftime('%d.%m.%Y %H:%M')} ({birth_tz})",
            "birth_place": birth_place,
            "birth_lat": birth_lat,
            "birth_lon": birth_lon,
            "birth_tz": birth_tz,
        })

        await update.message.reply_text(
            "📍 Укажите текущее место (Город, Страна):"
        )
        return CURRENT_PLACE
    except Exception as e:
        await update.message.reply_text(
            f"⚠ {e}\nПопробуйте: `18.08.1995 14:30 Москва, Россия`",
            parse_mode="Markdown"
        )
        return BIRTH_DATA

async def get_current_place(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка текущего места."""
    try:
        place_now = update.message.text.strip()
        now_lat, now_lon, now_tz = geocode(place_now)
        context.user_data.update({
            "now_place": place_now,
            "now_lat": now_lat,
            "now_lon": now_lon,
            "now_tz": now_tz,
        })
        await update.message.reply_text("Ваш пол (м/ж):")
        return GENDER
    except Exception as e:
        await update.message.reply_text(f"⚠ {e}\nУкажите: `Город, Страна`")
        return CURRENT_PLACE

async def get_gender(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка пола."""
    gender = update.message.text.strip().lower()
    if gender not in ("м", "ж"):
        await update.message.reply_text("⚠ Укажите: 'м' или 'ж'")
        return GENDER
    context.user_data["gender"] = gender
    await update.message.reply_text("Семейное положение:")
    return FAMILY_STATUS

async def finalize_and_send(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Финализация профиля и отправка карты."""
    status = update.message.text.strip()
    context.user_data["status"] = status

    # Расчёты
    b_utc = context.user_data["birth_time_utc"]
    blat, blon = context.user_data["birth_lat"], context.user_data["birth_lon"]

    await update.message.reply_text("⏳ Рассчитываю вашу карту...")

    try:
        positions = calc_positions(b_utc)
        houses, asc = calc_houses(b_utc, blat, blon)
        aspects = calc_aspects(positions)

        context.user_data["positions"] = positions
        context.user_data["aspects"] = aspects
        context.user_data["houses"] = houses
        context.user_data["asc"] = asc
        context.user_data["daily_off"] = False

        user_id = update.message.from_user.id
        save_profile(user_id, dict(context.user_data))

        # Визуализация
        img = draw_chart(positions, houses, aspects, asc)

        intro = (
            f"✨ Натальная карта готова!\n"
            f"📅 {context.user_data['birth_time_local_str']}\n"
            f"📍 {context.user_data['birth_place']}\n"
            f"🌍 Сейчас: {context.user_data.get('now_place')}\n\n"
            f"Выберите тему или задайте вопрос:"
        )

        await update.message.reply_photo(photo=img, caption=intro, reply_markup=kb_topics())

        # Полный анализ
        profile = load_profile(user_id)
        full_text = gpt_analyze("Дай полный анализ натальной карты.", profile)
        await update.message.reply_text(full_text, reply_markup=kb_topics())

    except Exception as e:
        logger.error(f"Ошибка финализации: {e}")
        await update.message.reply_text(f"⚠ Ошибка расчета: {e}\nПопробуйте /start")

    return ConversationHandler.END

# --------- Callback обработка ---------
async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка нажатий кнопок."""
    query = update.callback_query
    await query.answer()
    uid = query.from_user.id

    if query.data == "change":
        delete_profile(uid)
        context.user_data.clear()
        await query.message.reply_text(
            "♻️ Данные сброшены. Отправьте новые:\n`ДД.ММ.ГГГГ ЧЧ:ММ Город, Страна`",
            parse_mode="Markdown"
        )
        return

    if query.data == "stop_daily":
        profile = load_profile(uid)
        if profile:
            profile['daily_off'] = True
            save_profile(uid, profile)
        await query.message.reply_text("🚫 Рассылка отключена. Включить: /start")
        return

    # Тематические вопросы
    THEME_MAP = {
        "career": "Карьера и профессиональная реализация",
        "love": "Любовь и отношения",
        "health": "Здоровье и благополучие",
        "growth": "Личностный рост и развитие",
    }
    
    if query.data in THEME_MAP:
        theme = THEME_MAP[query.data]
        profile = load_profile(uid)
        
        if not profile:
            await query.message.reply_text("Сначала: /start")
            return
        
        await query.message.reply_text(f"⏳ Анализирую: {theme}...")
        
        try:
            answer = gpt_analyze(f"Подробно про: {theme}", profile)
            await query.message.reply_text(f"🔮 {theme}\n\n{answer}", reply_markup=kb_topics())
        except Exception as e:
            logger.error(f"Ошибка анализа темы: {e}")
            await query.message.reply_text(
                "⚠ Не удалось получить анализ. Попробуйте позже.",
                reply_markup=kb_topics()
            )

# --------- Свободные вопросы ---------
async def free_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка текстовых вопросов."""
    # Проверяем, что не в процессе настройки
    if context.user_data.get('in_conversation'):
        return
        
    uid = update.message.from_user.id
    profile = load_profile(uid)
    
    if not profile:
        await update.message.reply_text(
            "Сначала настройте профиль: /start",
            parse_mode="Markdown"
        )
        return
    
    q = update.message.text.strip()
    await update.message.reply_text("⏳ Думаю...")
    
    # Проверка на прогнозные запросы
    current_data = None
    if any(word in q.lower() for word in ["прогноз", "сегодня", "сейчас", "транзит"]):
        try:
            now_utc = dt.datetime.now(pytz.UTC)
            current_positions = calc_positions(now_utc)
            transit_aspects = calc_transit_aspects(current_positions, profile["positions"])
            current_houses, _ = calc_houses(now_utc, profile["now_lat"], profile["now_lon"])
            current_data = {
                "current_positions": current_positions,
                "transit_aspects": transit_aspects,
                "current_houses": current_houses,
            }
        except Exception as e:
            logger.error(f"Ошибка расчета транзитов: {e}")
    
    try:
        answer = gpt_analyze(q, profile, current_data)
        await update.message.reply_text(answer, reply_markup=kb_topics())
    except Exception as e:
        logger.error(f"Ошибка ответа на вопрос: {e}")
        await update.message.reply_text(
            "⚠ Не удалось обработать вопрос. Попробуйте переформулировать.",
            reply_markup=kb_topics()
        )

# --------- Команды ---------
async def cmd_change(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Смена данных."""
    uid = update.message.from_user.id
    delete_profile(uid)
    context.user_data.clear()
    await update.message.reply_text(
        "♻️ Данные сброшены. Отправьте новые:\n`ДД.ММ.ГГГГ ЧЧ:ММ Город, Страна`",
        parse_mode="Markdown"
    )

async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Отключение рассылки."""
    uid = update.message.from_user.id
    profile = load_profile(uid)
    if profile:
        profile['daily_off'] = True
        save_profile(uid, profile)
        await update.message.reply_text("🚫 Рассылка отключена.")
    else:
        await update.message.reply_text("У вас нет активного профиля.")

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Справка по командам."""
    help_text = """
📚 Доступные команды:

/start - Создать профиль
/change - Сменить данные
/stop - Отключить рассылку
/help - Эта справка

💡 Вы можете задавать любые вопросы текстом, и я отвечу с учётом вашей натальной карты!
    """
    await update.message.reply_text(help_text)

# --------- Ежедневная рассылка (исправленная) ---------
async def daily_job(app):
    """Отправка ежедневных прогнозов."""
    logger.info("🔄 Запуск ежедневной рассылки...")
    now_utc = dt.datetime.now(pytz.UTC)
    users = get_all_users()
    
    logger.info(f"📊 Найдено пользователей для проверки: {len(users)}")
    
    sent_count = 0
    for profile in users:
        try:
            uid = profile['user_id']
            tz_name = profile.get("now_tz", "UTC")
            
            # Конвертируем в локальное время пользователя
            user_tz = pytz.timezone(tz_name)
            local_now = now_utc.astimezone(user_tz)
            today_str = local_now.strftime("%Y-%m-%d")
            
            logger.info(f"👤 Пользователь {uid}: локальное время {local_now.strftime('%H:%M')}, часовой пояс {tz_name}")
            
            # ИСПРАВЛЕНО: проверяем 9:00-9:59
            if 9 <= local_now.hour < 10:
                last_sent = profile.get('last_daily_sent')
                
                if last_sent == today_str:
                    logger.info(f"⏭️ Пользователь {uid}: уже получил прогноз сегодня ({last_sent})")
                    continue
                
                logger.info(f"📤 Отправка прогноза пользователю {uid}...")
                
                # Расчет транзитов
                current_positions = calc_positions(now_utc)
                transit_aspects = calc_transit_aspects(current_positions, profile["positions"])
                current_houses, _ = calc_houses(now_utc, profile["now_lat"], profile["now_lon"])
                
                current_data = {
                    "current_positions": current_positions,
                    "transit_aspects": transit_aspects,
                    "current_houses": current_houses,
                }
                
                # Генерация прогноза
                text = gpt_analyze(
                    "Дай краткий астрологический прогноз на сегодня (3-4 абзаца). ВАЖНО: НЕ указывай конкретную дату в тексте, используй только слово 'Сегодня'.", 
                    profile, 
                    current_data
                )
                
                # Отправка
                await app.bot.send_message(
                    chat_id=uid,
                    text=f"🌞 Прогноз на {local_now.strftime('%d.%m.%Y')}:\n\n{text}",
                    reply_markup=kb_topics()
                )
                
                # Обновляем дату последней отправки
                update_last_daily_sent(uid, today_str)
                sent_count += 1
                logger.info(f"✅ Прогноз отправлен пользователю {uid}")
            else:
                logger.debug(f"⏰ Пользователь {uid}: не время для отправки (сейчас {local_now.hour}:00)")
                
        except Exception as e:
            logger.error(f"❌ Ошибка отправки прогноза пользователю {profile.get('user_id')}: {e}", exc_info=True)
    
    logger.info(f"✅ Рассылка завершена. Отправлено: {sent_count} из {len(users)}")

# ----------------- MAIN (исправленный) -----------------
def main():
    """Запуск бота."""
    logger.info("🚀 Инициализация бота...")
    
    # Инициализация БД
    init_db()
    
    # Создание приложения
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Conversation handler
    conv = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            BIRTH_DATA: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_birth_data)],
            CURRENT_PLACE: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_current_place)],
            GENDER: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_gender)],
            FAMILY_STATUS: [MessageHandler(filters.TEXT & ~filters.COMMAND, finalize_and_send)],
        },
        fallbacks=[CommandHandler("start", start)],
        per_user=True,
        per_chat=True,
    )
    
    # Добавление обработчиков
    app.add_handler(conv)
    app.add_handler(CallbackQueryHandler(on_callback))
    app.add_handler(CommandHandler("change", cmd_change))
    app.add_handler(CommandHandler("stop", cmd_stop))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, free_question))
    
    # Обработчик ошибок
    app.add_error_handler(error_handler)
    
    # Планировщик для ежедневных прогнозов
    scheduler = AsyncIOScheduler(timezone="UTC")
    
    # ИСПРАВЛЕНО: проверяем каждые 15 минут с 6:00 до 12:00 UTC
    # Это покрывает 9:00 во всех часовых поясах (от UTC-3 до UTC+12)
    for hour in range(6, 13):
        scheduler.add_job(
            daily_job,
            CronTrigger(hour=hour, minute="0,15,30,45", timezone="UTC"),
            args=[app],
            id=f"daily_job_{hour}",
            replace_existing=True
        )
    
    scheduler.start()
    logger.info("⏰ Планировщик запущен (проверка с 6:00 до 12:00 UTC каждые 15 минут)")
    
    # Запуск polling
    logger.info("✅ Бот запущен и готов к работе!")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("👋 Бот остановлен пользователем")
    except Exception as e:
        logger.error(f"💥 Критическая ошибка: {e}", exc_info=True)