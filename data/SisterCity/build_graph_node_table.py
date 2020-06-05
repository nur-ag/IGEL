
capitals = set()
with open('WorldCapitals.csv', 'r') as f:
    for line in f:
        capital = line.strip()
        capitals.add(capital)

city_to_graph = {}
with open('CityCountry.csv', 'r') as f:
    for line in f:
        line = line.strip()
        g_id, city, country = line.split('\t')
        city = city.replace('.', '').replace(',', '').replace("'", '')
        city_to_graph[city] = (g_id, city, country)

print('Id\tLabel\tCountry\tLat\tLng\tIsCapital')
with open('NodeTable.csv', 'r') as f:
    for line in f:
        line = line.strip()
        _, city, lat, lng = line.split('\t')
        city = city.replace('.', '').replace(',', '').replace("'", '')
        if city not in city_to_graph:
            continue
        g_id, city, country = city_to_graph[city]
        is_capital = str(1 if city in capitals else 0)
        new_line = '\t'.join([g_id, city, country, lat, lng, is_capital])
        print(new_line)

