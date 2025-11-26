import os
import sys

#add src folder to sys.path so workshop2 can be imported
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.append(base_dir)

from workshop2.probabilistic_structures import BloomFilter, FMEstimator
from workshop2.adaptive_sampling import (
    AdaptiveSampler,
    FrequencyMomentsAnalyzer,
    process_stream_with_adaptive_sampling,
)
from workshop2.stream_analysis import StreamAnalyzer


def build_w2_structures(records):
    #create bloom filter and fm estimator
    bloom = BloomFilter(size=4096, hash_count=4)
    fm = FMEstimator(max_bits=64)

    #create adaptive sampler and frequency analyzer
    sampler = AdaptiveSampler(
        base_rate=0.1,
        min_rate=0.01,
        max_rate=0.9,
        pollutant_thresholds={
            "PM25": 35.0,
            "NO2": 100.0,
            "CO": 10.0,
        },
        pollutant_type_weights={
            "PM2.5": 1.2,
            "PM25": 1.2,
            "NO2": 1.0,
            "CO": 0.8,
        },
    )

    analyzer = FrequencyMomentsAnalyzer(
        critical_thresholds={
            "PM25": 35.0,
            "NO2": 100.0,
            "CO": 10.0,
        }
    )

    #create stream analyzer for dgim over pm2.5
    stream = StreamAnalyzer(
        window_size=1024,
        threshold=35.0,
        max_buckets_per_size=2,
    )

    #first pass for bloom, fm and dgim
    for rec in records:
        aq = rec.get("airQualityData") or {}
        pm = aq.get("PM25")
        no2 = aq.get("NO2")
        pollution_type = rec.get("pollutionType")
        city = rec.get("sensorLocation")
        alert_flag = rec.get("alertIssued")

        #bloom item to represent similar air quality levels
        bloom_key = f"{city}|{pollution_type}|{pm}|{no2}"
        bloom.add(bloom_key)

        #distinct pollution events for fm estimator
        event_key = f"{city}|{pollution_type}|{alert_flag}"
        fm.add(event_key)

        #feed dgim stream with pm2.5
        if pm is not None:
            ts = rec.get("timestamp")
            sensor_id = rec.get("sensorId") or city or "unknown"
            stream.add_reading(ts, sensor_id, pm)

    #second pass for adaptive sampling + frequency moments
    sampling_stats = process_stream_with_adaptive_sampling(
        records_iter=records,
        sampler=sampler,
        analyzer=analyzer,
        time_window="hour",
        write_sampled_to=None,
    )

    #compute fm estimate
    distinct_events_estimate = fm.estimate()

    #frequency moment 1 by city (critical alerts)
    moment1_by_city = analyzer.compute_moment1_by_city()
    top_zones = analyzer.rank_zones_by_frequency(top_n=10)
    temporal_patterns = analyzer.temporal_patterns()

    #dgim estimates and trends
    dgim_last_100 = stream.estimate_alerts(k=100)
    dgim_exact_last_100 = stream.exact_alerts(k=100)
    dgim_trend = stream.detect_trend(
        window_length=100,
        step=10,
        lookback_windows=5,
        change_threshold=0.2,
    )
    dgim_prediction = stream.predict_counts(
        window_length=100,
        lookahead=1,
    )

    #result summary
    result = {
        "bloom_filter": bloom,
        "fm_estimator": fm,
        "adaptive_sampler": sampler,
        "frequency_analyzer": analyzer,
        "stream_analyzer": stream,
        "sampling_stats": sampling_stats,
        "distinct_events_estimate": distinct_events_estimate,
        "moment1_by_city": moment1_by_city,
        "top_zones": top_zones,
        "temporal_patterns": temporal_patterns,
        "dgim_last_100": dgim_last_100,
        "dgim_exact_last_100": dgim_exact_last_100,
        "dgim_trend": dgim_trend,
        "dgim_prediction": dgim_prediction,
    }
    return result


def run_w2_from_records(records):
    #helper for external modules (e.g. workshop1 dashboard)
    return build_w2_structures(records)
