import binascii
import struct
import pandas as pd
import numpy as np
from pyproj import Transformer
from pyspark.sql import SparkSession
from math import sin, cos, sqrt, atan2, radians

from pyspark.sql import DataFrame

from .consts import PROJECTION_FROM, PROJECTION_TO, EARTH_RADIUS, MAIN_BASE, JDBC_URL, CONNECTION_PROPERTIES




def extract_lat_lon(geom_point: str) -> tuple[float, float]:
    x, y = struct.unpack('<dd', binascii.unhexlify(geom_point[18:]))
    transformer = Transformer.from_crs(PROJECTION_FROM, PROJECTION_TO)
    lon, lat = transformer.transform(x, y)

    return (lon, lat)


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> int:
    _lat1 = radians(lat1)
    _lon1 = radians(lon1)
    _lat2 = radians(lat2)
    _lon2 = radians(lon2)

    dlon = _lon2 - _lon1
    dlat = _lat2 - _lat1

    a = sin(dlat / 2) ** 2 + cos(_lat1) * cos(_lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return round(EARTH_RADIUS * c * 1000)


def get_coordinates(df: DataFrame) -> list[tuple[float, float]]:
    coordinates = [MAIN_BASE]

    for row in df.collect():
        coordinates.append(extract_lat_lon(row.geom_point))

    return coordinates



def get_reduced_distance_matrix(geom_points: list[str], no_cache=False) -> tuple[np.ndarray, list[str]]:
    if no_cache:
        create_new_distance_matrix()
    # Load the distance matrix from the CSV file
    df = pd.read_csv('data/luftlinie_distance_matrix.csv', sep=',')

    # Extract the row and column headers
    row_headers = df.iloc[:, 0].tolist()
    column_headers = df.columns[1:].tolist()

    # Extract the distance matrix data
    distance_matrix = df.iloc[:, 1:].to_numpy()

    # Define the indices of the rows and columns you want to extract
    indices = [row_headers.index(point) for point in geom_points]
    indices.sort()

    selected_row_indices = selected_column_indices = indices

    # Extract the partial distance matrix
    partial_distance_matrix = distance_matrix[selected_row_indices, :][:, selected_column_indices]

    # Extract the corresponding row and column headers
    partial_row_headers = [row_headers[i] for i in selected_row_indices]
    partial_column_headers = [column_headers[i] for i in selected_column_indices]

    return partial_distance_matrix, partial_row_headers

def extract_new_lat_lon(geom_point: str) -> tuple:
    x, y = struct.unpack('<dd', binascii.unhexlify(geom_point[18:]))
    transformer = Transformer.from_crs(PROJECTION_FROM, PROJECTION_TO)
    lon, lat = transformer.transform(x, y)

    return (geom_point, lon, lat)


def create_new_distance_matrix(spark: SparkSession, jdbc_url: str=JDBC_URL, connection_properties: dict[str, str]=CONNECTION_PROPERTIES):
    df = spark.read.jdbc(url=jdbc_url, table="public.litter_bin_geoposition", properties=connection_properties)
    indexed_coordinates_rdd = df.rdd.map(lambda row: extract_new_lat_lon(row.geom_point))

    n = indexed_coordinates_rdd.count()

    indexed_coordinates = indexed_coordinates_rdd.collect()
    indexed_coordinates.sort(key=lambda x: x[0])
    index_vector = np.array([x[0] for x in indexed_coordinates])
    coordinates = np.array([(x[1], x[2]) for x in indexed_coordinates])
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                lat1, lon1 = coordinates[i]
                lat2, lon2 = coordinates[j]
                distance_matrix[i][j] = haversine(lat1, lon1, lat2, lon2)
    pd.DataFrame(distance_matrix, index=index_vector, columns=index_vector).to_csv('data/luftlinie_distance_matrix.csv').to_csv('data/luftlinie_distance_matrix.csv')


def create_distance_matrix(coordinates: list[tuple[float, float]]) -> list[list[int]]:
    distance_matrix = []

    n = len(coordinates)

    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append(0)
            else:
                lat1, lon1 = coordinates[i]
                lat2, lon2 = coordinates[j]

                distance = haversine(lat1, lon1, lat2, lon2)
                row.append(distance)
        distance_matrix.append(row)

    return distance_matrix
