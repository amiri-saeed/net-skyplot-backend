# config.py

class FlaskConfig:
    """Configuration settings related to the Flask app."""
    DEBUG = True
    TESTING = False
    SECRET_KEY = 'your-secret-key'  # Replace with an actual secret key
    HOST = '0.0.0.0'
    PORT = 5000


class DatabaseConfig:
    """Configuration for database connections."""
    DB_HOST = 'localhost'
    DB_PORT = 5432
    DB_USER = 'user'
    DB_PASSWORD = 'password'
    DB_NAME = 'my_database'


class APISettings:
    """Settings for API-related parameters."""
    API_KEY = 'your-api-key'  # Example for external service integration
    TIMEOUT = 30  # Timeout duration for API requests in seconds


class ObstacleConfig:
    """Configuration for handling obstacle data in the backend."""
    DEFAULT_HEIGHT = 10.0
    MAX_HEIGHT = 100.0
    MIN_HEIGHT = 0.5
    ALLOW_COMPLEX_SHAPES = True


class GeneralConfig:
    """General configurations for the backend, shared across components."""
    LOG_LEVEL = 'DEBUG'
    ENABLE_FEATURE_X = True


# Aggregated configuration for different environments
class DevelopmentConfig(FlaskConfig, DatabaseConfig, APISettings, ObstacleConfig, GeneralConfig):
    """Development environment configuration, combining all specific configs."""
    DEBUG = True
    TESTING = False


class TestingConfig(FlaskConfig, DatabaseConfig, APISettings, ObstacleConfig, GeneralConfig):
    """Testing environment configuration."""
    DEBUG = True
    TESTING = True


class ProductionConfig(FlaskConfig, DatabaseConfig, APISettings, ObstacleConfig, GeneralConfig):
    """Production environment configuration."""
    DEBUG = False
    TESTING = False
    LOG_LEVEL = 'WARNING'
