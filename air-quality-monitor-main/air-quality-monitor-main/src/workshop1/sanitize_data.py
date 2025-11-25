import json
import os
from datetime import datetime, timedelta

INPUT_PATH = "sensor_data.json"
OUTPUT_DIR = "data"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "sensor_data_clean.json")

def main():
    #data if no exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    #load raw data
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    #timestamp base
    base_time = datetime(2025, 1, 1, 0, 0, 0)

    for i, rec in enumerate(data):
        ts = rec.get("timestamp")
        if not isinstance(ts, str) or "TypeError" in ts:
            rec["timestamp"] = (base_time + timedelta(minutes=i)).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )

        aq = rec.get("airQualityData")
        if isinstance(aq, dict):
            for key in ("PM25", "NO2"):
                val = aq.get(key)
                try:
                    aq[key] = float(val)
                except (TypeError, ValueError):
                    aq[key] = None
            rec["airQualityData"] = aq

    #save clean json
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Datos limpiados guardados en: {OUTPUT_PATH}")
    print(f"   Registros procesados: {len(data)}")

if __name__ == "__main__":
    main()
