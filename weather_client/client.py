import requests
from pydantic import BaseModel, Field
from typing import List, Optional

class DailyForecast(BaseModel):
    dt: int
    sunrise: int
    sunset: int
    moonrise: Optional[int] = None
    moonset: Optional[int] = None
    moon_phase: float

class WeatherResponse(BaseModel):
    lat: float
    lon: float
    timezone: str
    timezone_offset: int
    data: List[DailyForecast] = Field(default_factory=list)

class OpenWeatherClient:
    def __init__(self, api_key: str, base_url: str = "https://api.openweathermap.org/data/2.5/onecall?lat={lat}&lon={lon}&appid={API key}"):
        self.api_key = api_key
        self.base_url = base_url

    def get_current(self, lat: float, lon: float) -> WeatherResponse:
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.api_key,
            "units": "metric",
            "lang": "en",
            "exclude": "current,minutely,hourly,alerts"
        }
        resp = requests.get(self.base_url, params=params, timeout=10)
        resp.raise_for_status()
        return WeatherResponse(**resp.json())

if __name__ == "__main__":
    client = OpenWeatherClient("key")
    weather = client.get_current(50, 30)
    
    print(f"Прогноз для {weather.lat}, {weather.lon}")
    print(f"Первый день: восход {weather.data[0].sunrise}, закат {weather.data[0].sunset}")
