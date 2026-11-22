from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str
    arc_llm_key: str
    arc_llm_endpoint: str
    arc_llm_model: str
    arc_llm_classifier_model: str | None = None
    arc_llm_context_tokens: int | None = None
    embedding_model: str = "Qwen3-Embedding-4B"

    @property
    def classifier_model(self) -> str:
        """Effective model name used by the topic classifier."""
        return self.arc_llm_classifier_model or self.arc_llm_model

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
