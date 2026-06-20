---
name: weather
description: Get current conditions and a multi-day forecast for any location, via the free open-meteo.com API. Use when the user asks about weather, temperature, or forecasts.
---

# Weather

Fetch weather with the `http_request` tool against open-meteo.com (no API key).

## 1. Geocode the place name → latitude/longitude

```
http_request(url="https://geocoding-api.open-meteo.com/v1/search?name=<PLACE>&count=1&language=en&format=json")
```

The API rejects commas — for "Paris, France" send just "Paris". Read
`results[0].latitude`, `results[0].longitude`, and `results[0].name`. If
`results` is empty, the place wasn't found.

## 2. Fetch the forecast

```
http_request(url="https://api.open-meteo.com/v1/forecast?latitude=<LAT>&longitude=<LON>&current=temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m&daily=weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,sunrise,sunset&forecast_days=5&temperature_unit=fahrenheit&wind_speed_unit=mph&precipitation_unit=inch&timezone=auto")
```

Drop the `*_unit` params for metric (°C, km/h, mm).

## 3. Report

Summarise current conditions and the daily highs/lows. `weather_code` is a
WMO code: 0 clear, 1–3 increasingly cloudy, 45/48 fog, 51–67 drizzle/rain,
71–77 snow, 80–82 rain showers, 95–99 thunderstorm.
