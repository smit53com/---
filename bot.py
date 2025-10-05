import os
import pytz
from datetime import datetime
from dotenv import load_dotenv
from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ConversationHandler, ContextTypes, filters

from astro.ephemeris import planet_longitudes, aspect
from astro.chart import render_wheel_png
from app.storage import load_store, save_store, User, hash_email

load_dotenv()
BOT_TOKEN = os.getenv("TELEGRAM_TOKEN")
geolocator = Nominatim(user_agent="astro_bot")
tf = TimezoneFinder()

EMAIL, BIRTH_DATE, BIRTH_TIME, PLACE = range(4)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Я астрологический ассистент. Введите ваш e-mail:")
    return EMAIL

async def get_email(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['email'] = update.message.text.strip()
    await update.message.reply_text("Введите дату рождения (ГГГГ-ММ-ДД):")
    return BIRTH_DATE

async def get_birth_date(update, context):
    context.user_data['birth_date'] = update.message.text.strip()
    await update.message.reply_text("Введите время рождения (ЧЧ:ММ):")
    return BIRTH_TIME

async def get_birth_time(update, context):
    context.user_data['birth_time'] = update.message.text.strip()
    await update.message.reply_text("Введите место рождения (Город, Страна):")
    return PLACE

async def get_place(update, context):
    place = update.message.text.strip()
    location = geolocator.geocode(place)
    if not location:
        await update.message.reply_text("❌ Не удалось найти этот город. Попробуйте снова.")
        return PLACE
    lat, lon = location.latitude, location.longitude
    tz_name = tf.timezone_at(lng=lon, lat=lat)
    if not tz_name:
        await update.message.reply_text("❌ Не удалось определить часовой пояс.")
        return PLACE
    context.user_data['birth_place'] = place
    context.user_data['latitude'] = lat
    context.user_data['longitude'] = lon
    context.user_data['timezone'] = tz_name

    store = load_store()
    u = User(**context.user_data)
    store[hash_email(u.email)] = u
    save_store(store)
    await update.message.reply_text(f"✅ Данные сохранены! Ваш часовой пояс: {tz_name}. Используйте /today для гороскопа.")
    return ConversationHandler.END

async def today(update: Update, context: ContextTypes.DEFAULT_TYPE):
    store = load_store()
    user = next(iter(store.values()), None)
    if not user:
        await update.message.reply_text("Сначала зарегистрируйтесь: /start")
        return
    tz = pytz.timezone(user.timezone)
    natal_dt = tz.localize(datetime.strptime(f"{user.birth_date} {user.birth_time}", "%Y-%m-%d %H:%M"))
    today_dt = datetime.now(tz)
    natal_longs = planet_longitudes(natal_dt)
    transit_longs = planet_longitudes(today_dt)

    aspects_list = []
    for t_name, t_lon in transit_longs.items():
        for n_name, n_lon in natal_longs.items():
            asp = aspect(t_lon, n_lon)
            if asp:
                aspects_list.append(f"{t_name} {asp[0]} {n_name} (орб {asp[1]:.1f}°)")
    if not aspects_list:
        aspects_list.append("Сегодня без значимых аспектов.")

    img_bytes = render_wheel_png(transit_longs)
    await update.message.reply_photo(photo=img_bytes, caption="Ваш гороскоп на сегодня:\n" + "\n".join(aspects_list[:10]))

async def unsubscribe(update, context):
    store = load_store()
    store.clear()
    save_store(store)
    await update.message.reply_text("❌ Данные удалены, подписка отменена.")

async def help_cmd(update, context):
    await update.message.reply_text("/start — регистрация\n/today — гороскоп на сегодня\n/unsubscribe — отписка")

def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    conv = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            EMAIL: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_email)],
            BIRTH_DATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_birth_date)],
            BIRTH_TIME: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_birth_time)],
            PLACE: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_place)],
        },
        fallbacks=[],
    )
    app.add_handler(conv)
    app.add_handler(CommandHandler("today", today))
    app.add_handler(CommandHandler("unsubscribe", unsubscribe))
    app.add_handler(CommandHandler("help", help_cmd))
    app.run_polling()

if __name__ == "__main__":
    main()
