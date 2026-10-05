"""
Application settings.

Precedence (highest first): process env vars > secrets.env > src/config/config.yaml > field defaults.
Secrets and connection strings live in secrets.env; non-secret tunables live in config.yaml.
"""
from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_YAML_PATH = ROOT_DIR / "src" / "config" / "config.yaml"
SECRETS_ENV_PATH = ROOT_DIR / "secrets.env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=SECRETS_ENV_PATH,
        env_file_encoding="utf-8",
        yaml_file=CONFIG_YAML_PATH,
        case_sensitive=False,
        extra="ignore",
    )

    # General
    app_env: str = "dev"
    log_level: str = "INFO"
    default_timezone: str = "America/Vancouver"
    company_name: str = "Car Rental Company"
    openapi_docs: bool = True
    cors_origins: list[str] = []
    allowed_hosts: list[str] = ["*"]

    # Auth
    jwt_secret: SecretStr
    jwt_issuer: str = "car-rental-agent"
    jwt_ttl_seconds: int = 3600

    # Datastores
    postgres_dsn: SecretStr = SecretStr("postgresql://rental:rental_dev@localhost:5432/rental")
    mongo_uri: SecretStr = SecretStr("mongodb://localhost:27017")
    mongo_db: str = "car_rental_agent"
    redis_url: SecretStr = SecretStr("redis://localhost:6379/0")

    # Downstream services
    mock_services_url: str = "http://localhost:8001"
    agent_api_url: str = "http://localhost:8000"

    # LLM (Azure OpenAI: llm_model is the deployment name)
    llm_provider: str = ""
    llm_model: str = ""
    llm_api_key: SecretStr = SecretStr("")
    llm_endpoint: str = ""
    llm_api_version: str = ""
    llm_timeout_seconds: float = 30
    llm_max_retries: int = 2

    # Agent behaviour
    agent_max_steps: int = 6
    tool_timeout_seconds: float = 8
    turn_timeout_seconds: float = 45
    kb_min_score: float = 0.35
    rate_limit_per_minute: int = 20
    pending_action_ttl_seconds: int = 300
    quote_ttl_seconds: int = 900

    # Embeddings (mock RAG)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    @model_validator(mode="after")
    def _reject_default_secret_outside_dev(self) -> "Settings":
        if self.app_env != "dev" and self.jwt_secret.get_secret_value() in ("", "change-me"):
            raise ValueError("JWT_SECRET must be set to a strong value outside dev")
        return self

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
