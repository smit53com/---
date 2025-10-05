import swisseph as swe
import datetime
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io

# Настройки эфемерид
swe.set_ephe_path('.')

PLANETS = {
    "☉": swe.SUN,
    "☽": swe.MOON,
    "☿": swe.MERCURY,
    "♀": swe.VENUS,
    "♂": swe.MARS,
    "♃": swe.JUPITER,
    "♄": swe.SATURN,
    "♅": swe.URANUS,
    "♆": swe.NEPTUNE,
    "♇": swe.PLUTO,
}

ASPECTS = {
    "Соединение": 0,
    "Оппозиция": 180,
    "Трин": 120,
    "Квадрат": 90,
    "Секстиль": 60,
    "Квиконс": 150,
}

def calculate_positions(date_time, lat, lon):
    jd = swe.julday(date_time.year, date_time.month, date_time.day,
                    date_time.hour + date_time.minute/60.0)

    positions = {}
    for symbol, planet in PLANETS.items():
        lon, lat_, dist = swe.calc_ut(jd, planet)[0:3]
        positions[symbol] = lon
    return positions

def calculate_aspects(positions, orb=2):
    aspects_found = []
    planets = list(positions.items())
    for i in range(len(planets)):
        for j in range(i+1, len(planets)):
            p1, lon1 = planets[i]
            p2, lon2 = planets[j]
            diff = abs(lon1 - lon2) % 360
            for name, angle in ASPECTS.items():
                if abs(diff - angle) <= orb or abs(360 - diff - angle) <= orb:
                    aspects_found.append((p1, p2, name))
    return aspects_found

def draw_chart(positions, aspects):
    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_aspect("equal")
    ax.axis("off")

    # Круг зодиака
    circle = plt.Circle((0,0), 0.9, fill=False)
    ax.add_artist(circle)

    # Дома 12
    for i in range(12):
        angle = math.radians(i*30)
        ax.plot([0, 0.9*math.cos(angle)], [0, 0.9*math.sin(angle)], color="black", lw=0.5)

    # Планеты
    for symbol, lon in positions.items():
        angle = math.radians(lon)
        x, y = 0.8*math.cos(angle), 0.8*math.sin(angle)
        ax.text(x, y, symbol, fontsize=14, ha="center", va="center")

    # Аспекты
    for p1, p2, aspect in aspects:
        lon1, lon2 = positions[p1], positions[p2]
        x1, y1 = 0.8*math.cos(math.radians(lon1)), 0.8*math.sin(math.radians(lon1))
        x2, y2 = 0.8*math.cos(math.radians(lon2)), 0.8*math.sin(math.radians(lon2))
        ax.plot([x1, x2], [y1, y2], color="red", lw=0.7)

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    return buf
