import requests
import pandas as pd

# Coordenadas geográficas (Longitud, Latitud) de las 20 ciudades
coordenadas = {
    "Madrid": (-3.7038, 40.4168),
    "Guadalajara": (-3.1608, 40.6327),
    "Cuenca": (-2.1374, 40.0704),
    "Teruel": (-1.1072, 40.3456),
    "Segovia": (-4.1088, 40.9429),
    "Toledo": (-4.0273, 39.8628),
    "Soria": (-2.4688, 41.7666),
    "Ciudad Real": (-3.9271, 38.9863),
    "Puertollano": (-4.1073, 38.6871),
    "Calatayud": (-1.6422, 41.3526),
    "Zaragoza": (-0.8891, 41.6488),
    "Albacete": (-1.8585, 38.9942),
    "Villena": (-0.8658, 38.6358),
    "Alicante": (-0.4815, 38.3452),
    "Elche": (-0.6983, 38.2669),
    "Orihuela": (-0.9442, 38.0853),
    "Murcia": (-1.1307, 37.9922),
    "Valencia": (-0.3763, 39.4699),
    "Gandia": (-0.1814, 38.9671),
    "Tarancón": (-3.0076, 40.0075),
    "Murcia": (-1.1239, 38.0336)
}

ciudades = list(coordenadas.keys())

# Formatear coordenadas para la API de OSRM (lon,lat;lon,lat...)
coords_str = ";".join([f"{lon},{lat}" for lon, lat in coordenadas.values()])

# Petición a la API pública de enrutamiento por carretera OSRM
url = f"http://router.project-osrm.org/table/v1/driving/{coords_str}?annotations=distance"

print("Calculando trayectos por carretera real...")
response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    # Convertir las distancias de metros a kilómetros
    matriz_km = [[round(dist / 1000) for dist in fila] for fila in data["distances"]]

    # Crear DataFrame con pandas
    df_carretera = pd.DataFrame(matriz_km, index=ciudades, columns=ciudades)

    print("\n=== MATRIZ DE DISTANCIAS REALES POR CARRETERA (KM) ===")
    print(df_carretera)
    print('Convierte a lista de python')
    print(df_carretera.values.tolist())
else:
    print("Error al conectar con el servidor de rutas.")