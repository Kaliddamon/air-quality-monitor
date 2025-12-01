import json
import os
import random
from datetime import datetime, timedelta

#base paths relative to project root
base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(base_dir))
input_path = os.path.join(project_root, "data", "sensor_data.json")
output_dir = os.path.join(project_root, "data")
output_path = os.path.join(output_dir, "sensor_data_clean.json")


def main():
    #create output folder
    os.makedirs(output_dir, exist_ok=True)

    #load raw data
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    #base timestamp for invalid ones
    base_time = datetime(2025, 1, 1, 0, 0, 0)

    #possible pollution types
    pollution_types = ["PM2.5", "NO2", "CO"]

    for i, rec in enumerate(data):

        #timestamp cleaning
        ts = rec.get("timestamp")
        if not isinstance(ts, str) or "typeerror" in str(ts).lower():
            rec["timestamp"] = (base_time + timedelta(minutes=i)).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )

        #air quality numeric cleaning
        aq = rec.get("airQualityData")
        if isinstance(aq, dict):
            for key in ("PM25", "NO2"):
                val = aq.get(key)
                try:
                    aq[key] = float(val)
                except (TypeError, ValueError):
                    aq[key] = None
            rec["airQualityData"] = aq
        else:
            rec["airQualityData"] = {"PM25": None, "NO2": None}

        #3.1 pollutiontype
        curr_ptype = rec.get("pollutionType")
        if curr_ptype not in pollution_types:
            rec["pollutionType"] = random.choice(pollution_types)

        #3.2 alertissued
        if not isinstance(rec.get("alertIssued"), bool):
            rec["alertIssued"] = bool(random.getrandbits(1))

        #3.3 geographicType
        if not isinstance(rec.get("geographicType"), str):
            geo_types = ["urban", "suburban", "rural"]
            rec["geographicType"] = random.choice(geo_types)

        #3.4 pollutionSource
        if not isinstance(rec.get("pollutionSource"), str):
            pollution_sources = ["traffic", "industrial", "mixed"]
            rec["pollutionSource"] = random.choice(pollution_sources)

        #3.5 coords
        if not isinstance(rec.get("coords"), tuple):
            x = random.uniform(0, 100)
            y = random.uniform(0, 100)
            rec["coords"] = (x, y)

        #3.6 pollution levels 7 days
        if not isinstance(rec.get("pollutionLevels7"), list):
            pollution_levels = [round(random.uniform(10, 150), 2) for _ in range(7)]
            rec["pollutionLevels7"] = pollution_levels

    #save cleaned data
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"clean data saved in: {output_path}")
    print(f"records processed: {len(data)}")


if __name__ == "__main__":
    main()

