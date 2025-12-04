from workshop6.unit_test_W6 import testKNN

def run_w6_from_records(records, sensor, k):
    
    simple_knn, geographic_knn, alert_knn, perf, geo_effect = testKNN(records, sensor, k)

    w6_result = {
      "simple_knn": simple_knn,
      "geographic_knn": geographic_knn,
      "alert_knn": alert_knn,
      "perf": perf,
      "geo_effect": geo_effect
    }
    
    return w6_result
