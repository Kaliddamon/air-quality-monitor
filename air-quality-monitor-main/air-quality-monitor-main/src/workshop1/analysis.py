import json
from collections import defaultdict
from datetime import datetime

# Funciones analiticas
def get_city_aggregates(hash_table):
    """Agrupa por sensorLocation y calcula conteo, min, max, avg para PM25 y NO2."""
    city = {}
    for rec1 in hash_table.values():
      for rec in rec1:
          loc = rec.get('sensorLocation')
          if not loc:
              continue
          aq = rec.get('airQualityData', {}) or {}
          pm = aq.get('PM25')
          no2 = aq.get('NO2')
          e = city.setdefault(loc, {
              'count': 0,
              'sum_PM25': 0.0,
              'sum_NO2': 0.0,
              'min_PM25': None,
              'max_PM25': None,
              'min_NO2': None,
              'max_NO2': None,
          })
          e['count'] += 1
          if pm is not None:
              e['sum_PM25'] += pm
              e['min_PM25'] = pm if e['min_PM25'] is None else min(e['min_PM25'], pm)
              e['max_PM25'] = pm if e['max_PM25'] is None else max(e['max_PM25'], pm)
          if no2 is not None:
              e['sum_NO2'] += no2
              e['min_NO2'] = no2 if e['min_NO2'] is None else min(e['min_NO2'], no2)
              e['max_NO2'] = no2 if e['max_NO2'] is None else max(e['max_NO2'], no2)
    for loc, e in city.items():
        cnt = e.get('count', 0) or 0
        e['avg_PM25'] = (e['sum_PM25'] / cnt) if cnt else None
        e['avg_NO2'] = (e['sum_NO2'] / cnt) if cnt else None
    return city

def rank_cities_by_pollutant(hash_table, pollutant='PM25', top_n=None, descending=True):
    aggs = get_city_aggregates(hash_table)
    key = f'avg_{pollutant}'
    items = [(city, vals.get(key)) for city, vals in aggs.items() if vals.get(key) is not None]
    items.sort(key=lambda x: x[1], reverse=descending)
    return items[:top_n] if top_n is not None else items

