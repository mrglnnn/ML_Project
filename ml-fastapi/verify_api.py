"""Run with FastAPI listening on port 8000."""
import requests

base = 'http://127.0.0.1:8000'
assert requests.get(base, timeout=10).status_code == 200
for features, species in [
    ([5.1, 3.5, 1.4, 0.2], 'Iris Setosa'),
    ([6.0, 2.9, 4.5, 1.5], 'Iris Versicolor'),
    ([6.5, 3.0, 5.8, 2.2], 'Iris Virginica'),
]:
    payload = dict(zip(['sepal_length', 'sepal_width', 'petal_length', 'petal_width'], features))
    response = requests.post(base + '/predict', json=payload, timeout=10)
    response.raise_for_status()
    assert response.json()['prediction'] == species, response.json()
    print(f'PASS: {species}')
response = requests.post(base + '/predict', json={'sepal_length': -1}, timeout=10)
assert response.status_code == 422
print('PASS: Invalid input rejected (422)')
print('5/5 HTTP checks passed (health, three classes, validation).')
