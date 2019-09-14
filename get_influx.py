import requests

url = 'http://{influx_host}:8086/query'.format(influx_host='')
directory = '.'

ex = '*'

query = 'select * from {table} where experiment =~ /{ex}/'.format(table='hflow_performance', ex=ex)

req = requests.get(url, params={
    'q': query,
    'db': 'hyperflow-database'
}, headers={'Accept': 'application/csv'})

with open(f'{directory}/hflow_performance.csv', 'w') as fp:
    fp.write(req.content.decode())
