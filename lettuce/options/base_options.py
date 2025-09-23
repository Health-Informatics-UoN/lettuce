from options.pipeline_options import LLMModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from .pipeline_options import EmbeddingModelName, InferenceType

class BaseOptions(BaseSettings):
    model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            )
    db_host: str = "localhost"
    db_user: str = "postgres"
    db_password: str = "password"
    db_name: str = "omop"
    db_port: int = 5432
    db_schema: str = "cdm"
    db_vectable: str = "embeddings"
    # have a branch where this comes from database - hard to integrate,#TODO
    db_vecsize: int = 384 

    inference_type: InferenceType = InferenceType.OLLAMA

    ollama_url: str = "http://localhost:11434"    

    llm_model: LLMModel = LLMModel.LLAMA_3_1_8B
    temperature: float = 0.0
    local_llm: str | None = None
    debug_prompt: bool = False

    embedding_model: EmbeddingModelName = EmbeddingModelName.BGESMALL
    embedding_top_k: int = 5
    
    auth_api_key: str | None = None

    def connection_url(self) -> str:
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

    def print(self):
        print("------------ Options -------------")
        for k, v in self.model_dump().items():
            print(f"{str(k)}: {str(v)}")
        print("-------------- End ---------------")
