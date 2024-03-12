import requests
import polars as pl

def collect_data_api(system="NAC", start_epoch= 1664299202, end_epoch= 1664385602):
    #Consider this only works for demandaNAC others may vary and require further implementation
    #Predict is CENACE forecast, Forecast is our forecast
    endpoint = f"http://orduna.xyz:5000/api/{system}/{start_epoch}/{end_epoch}"
    print(endpoint)
    response = requests.get(endpoint)
    api_data = response.json()
    data = []
    for item in api_data:
        timestamp = item['timestamp']
        load=item['demandaNAC2']
        predict=item['enlaceNAC']

        try:
            data.append([timestamp, int(load), int(predict)])
        except:
            return None

    return data

def aggregate_data(data):
    collected_data = pl.DataFrame(data,schema=['timestamp','demand','cforecast'])
    collected_data = collected_data.with_columns(pl.from_epoch(pl.col('timestamp'), time_unit='s').cast(pl.Datetime).dt.replace_time_zone("UTC").dt.convert_time_zone(time_zone="America/Mexico_City").alias('datetime'))
    aggregated_data = collected_data.sort('datetime').group_by_dynamic("datetime", every="1h").agg(pl.all().exclude('datetime').mean()).drop('timestamp')

    return aggregated_data
