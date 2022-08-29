from faker import Faker
import pandas as pd
fake = Faker()

lat = []
long = []
Faker.seed(0)
for _ in range(5):
    geo = fake.local_latlng(country_code='US')
    print(geo)
    lat.append(geo[0])
    long.append(geo[1])

print(lat)
