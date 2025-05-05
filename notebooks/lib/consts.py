PROJECTION_FROM: str = "EPSG:2056"
PROJECTION_TO: str = "EPSG:4326"
EARTH_RADIUS: float = 6373.0
MAIN_BASE: tuple[float, float] = (47.125835, 7.2596615)

JDBC_URL = "jdbc:postgresql://db:5432/litter_db"
CONNECTION_PROPERTIES = {
    "user": "root",
    "password": "pwd123",
    "driver": "org.postgresql.Driver"
}