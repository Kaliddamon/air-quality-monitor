import json
import os
import random
from datetime import datetime, timedelta

input_path = "sensor_data.json"
output_dir = "data"
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

    #save cleaned data
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"clean data saved in: {output_path}")
    print(f"records processed: {len(data)}")


if __name__ == "__main__":
    main()

