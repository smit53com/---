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
    raise ValueError("Не установлены переменные окружения TELEGRAM_TOKEN или OPENAI_API_KEY!")

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
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_db():
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
    logger.info("База данных инициализирована")

def save_profile(user_id: int, profile: Dict[str, Any]):
    profile_json = json.dumps(profile, default=str)
    with get_db() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO users (user_id, profile, daily_off, last_daily_sent)
            VALUES (?, ?, ?, ?)
        """, (user_id, profile_json, profile.get('daily_off', 0), profile.get('last_daily_sent')))
        conn.commit()

def load_profile(user_id: int) -> Optional[Dict[str, Any]]:
    with get_db() as conn:
        row = conn.execute("SELECT profile FROM users WHERE user_id = ?", (user_id,)).fetchone()
        if row:
            profile = json.loads(row['profile'])
            if 'birth_time_utc' in profile:
                profile['birth_time_utc'] = dt.datetime.fromisoformat(profile['birth_time_utc'])
            return profile
    return None

def delete_profile(user_id: int):
    with get_db() as conn:
        conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
        conn.commit()

def get_all_users() -> List[Dict[str, Any]]:
    with get_db() as conn:
        rows = conn.execute("""
            SELECT user_id, profile, daily_off, last_daily_sent 
            FROM users WHERE daily_off = 0
        """).fetchall()
        users = []
        for row in rows:
            profile = json.loads(row['profile'])
            profile['user_id'] = row['user_id']
            profile['daily_off'] = row['daily_off']  # Добавлено!
            profile['last_daily_sent'] = row['last_daily_sent']
            if 'birth_time_utc' in profile:
                profile['birth_time_utc'] = dt.datetime.fromisoformat(profile['birth_time_utc'])
            users.append(profile)
        return users

def update_last_daily_sent(user_id: int, date_str: str):
    with get_db() as conn:
        conn.execute("UPDATE users SET last_daily_sent = ? WHERE user_id = ?", (date_str, user_id))
        conn.commit()

# ----------------- UI элементы -----------------
def kb_topics() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Карьера", callback_data="career")],
        [InlineKeyboardButton("Любовь", callback_data="love")],
        [InlineKeyboardButton("Здоровье", callback_data="health")],
        [InlineKeyboardButton("Личностный рост", callback_data="growth")],
        [InlineKeyboardButton("Сменить данные", callback_data="change")],
        [InlineKeyboardButton("Отписаться", callback_data="stop_daily")],
    ])

# ----------------- Парсинг ввода -----------------
def try_parse_input(text: str) -> Tuple[dt.datetime, str]:
    text = text.strip()
    m = re.match(r"^(\d{2}\.\d{2}\.\d{4})\s+(\d{2}:\d{2})\s+(.+)$", text)
    if m:
        date_str, time_str, place = m.groups()
        naive = dt.datetime.strptime(f"{date_str} {time_str}", "%d.%m.%Y %H:%M")
        return naive, place.strip()

    m2 = re.match(r"^(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2})\s+(.+)$", text)
    if m2:
        date_str, time_str, place = m2.groups()
        naive = dt.datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
        return naive, place.strip()

    raise ValueError("Неверный формат. Используйте: 'ДД.ММ.ГГГГ ЧЧ:ММ Город, Страна'")

def geocode(place: str) -> Tuple[float, float, str]:
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

def calc_houses(utc_dt: dt.datetime