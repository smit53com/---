from skyfield.api import load
from datetime import datetime

def planet_longitudes(dt):
    planets = load('de421.bsp')
    earth = planets['earth']
    ts = load.timescale()
    t = ts.utc(dt.year, dt.month, dt.day, dt.hour, dt.minute)
    planet_names = ['sun', 'moon', 'mercury', 'venus', 'mars', 'jupiter', 'saturn']
    longs = {}
    for name in planet_names:
        body = planets[name]
        astrometric = earth.at(t).observe(body).apparent().ecliptic_latlon()
        longs[name.capitalize()] = float(astrometric[1].degrees)
    return longs

def aspect(lon1, lon2, orb=6):
    aspects = {0: 'соединение', 60: 'секстиль', 90: 'квадрат', 120: 'тригон', 180: 'оппозиция'}
    diff = abs(lon1 - lon2) % 360
    if diff > 180:
        diff = 360 - diff
    for angle, name in aspects.items():
        if abs(diff - angle) <= orb:
            return name, diff - angle
    return None
