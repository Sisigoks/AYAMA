"""Solar position from latitude, longitude and a UTC timestamp.

Fallback for imagery that carries no SUN_AZIMUTH / SUN_ELEVATION tag but does
carry GPS coordinates and an acquisition time (typical of drone JPGs).

NOAA solar position equations, accurate to roughly 0.1 deg over 1950-2050,
which is far tighter than the shadow-length error budget needs.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional


def julian_day(dt: datetime) -> float:
    dt = dt.astimezone(timezone.utc)
    y, m = dt.year, dt.month
    d = dt.day + (dt.hour + dt.minute / 60 + dt.second / 3600) / 24
    if m <= 2:
        y -= 1
        m += 12
    a = y // 100
    b = 2 - a + a // 4
    return int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + b - 1524.5


def solar_position(lat_deg: float, lon_deg: float, when: datetime) -> tuple[float, float]:
    """Return (azimuth_deg clockwise from north, elevation_deg above horizon)."""
    jd = julian_day(when)
    t = (jd - 2451545.0) / 36525.0

    # Geometric mean longitude and anomaly of the sun.
    l0 = (280.46646 + t * (36000.76983 + t * 0.0003032)) % 360.0
    m = 357.52911 + t * (35999.05029 - 0.0001537 * t)
    mr = math.radians(m)
    c = (
        math.sin(mr) * (1.914602 - t * (0.004817 + 0.000014 * t))
        + math.sin(2 * mr) * (0.019993 - 0.000101 * t)
        + math.sin(3 * mr) * 0.000289
    )
    true_long = l0 + c
    omega = 125.04 - 1934.136 * t
    app_long = true_long - 0.00569 - 0.00478 * math.sin(math.radians(omega))

    # Obliquity of the ecliptic.
    e0 = 23.0 + (26.0 + (21.448 - t * (46.815 + t * (0.00059 - t * 0.001813))) / 60.0) / 60.0
    eps = e0 + 0.00256 * math.cos(math.radians(omega))
    epsr, allr = math.radians(eps), math.radians(app_long)

    decl = math.asin(math.sin(epsr) * math.sin(allr))

    # Equation of time, in minutes.
    y = math.tan(epsr / 2) ** 2
    e_ecc = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)
    l0r = math.radians(l0)
    eot = 4 * math.degrees(
        y * math.sin(2 * l0r)
        - 2 * e_ecc * math.sin(mr)
        + 4 * e_ecc * y * math.sin(mr) * math.cos(2 * l0r)
        - 0.5 * y * y * math.sin(4 * l0r)
        - 1.25 * e_ecc * e_ecc * math.sin(2 * mr)
    )

    utc = when.astimezone(timezone.utc)
    minutes = utc.hour * 60 + utc.minute + utc.second / 60.0
    true_solar_time = (minutes + eot + 4 * lon_deg) % 1440.0
    hour_angle = math.radians(true_solar_time / 4.0 - 180.0)

    latr = math.radians(lat_deg)
    zenith = math.acos(
        max(-1.0, min(1.0, math.sin(latr) * math.sin(decl)
                      + math.cos(latr) * math.cos(decl) * math.cos(hour_angle)))
    )
    elev = 90.0 - math.degrees(zenith)

    # Azimuth, clockwise from north (NOAA spreadsheet form).
    sz = math.sin(zenith)
    if abs(sz) < 1e-9:
        az = 180.0
    else:
        cos_az = (math.sin(latr) * math.cos(zenith) - math.sin(decl)) / (math.cos(latr) * sz)
        acos_az = math.degrees(math.acos(max(-1.0, min(1.0, cos_az))))
        az = (acos_az + 180.0) % 360.0 if hour_angle > 0 else (540.0 - acos_az) % 360.0
    return az % 360.0, elev


def parse_utc(stamp: Optional[str]) -> Optional[datetime]:
    """Accept the handful of timestamp spellings that show up in EXIF and TIFF tags."""
    if not stamp:
        return None
    s = str(stamp).strip().replace("Z", "+00:00")
    for fmt in (
        "%Y:%m:%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%d",
    ):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.replace(tzinfo=dt.tzinfo or timezone.utc)
        except ValueError:
            continue
    try:
        dt = datetime.fromisoformat(s)
        return dt.replace(tzinfo=dt.tzinfo or timezone.utc)
    except ValueError:
        return None
