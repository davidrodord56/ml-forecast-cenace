import requests
import numpy

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

    print(data)




collect_data_api()