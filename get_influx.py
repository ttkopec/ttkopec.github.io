import requests

influx_host = ''
url = 'http://{influx_host}:8086/query'.format(influx_host='')
results = 'hflow_performance.csv'
experiment = '*'
table = 'hflow_performance'

query = 'select * from {table} where experiment =~ /{ex}/'.format(table=table, ex=experiment)

req = requests.get(url, params={
    'q': query,
    'db': 'hyperflow-database'
}, headers={'Accept': 'application/csv'})

with open(results, 'w') as fp:
    fp.write(req.content.decode())
