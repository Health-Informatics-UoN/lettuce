from options.pipeline_options import LLMModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from .pipeline_options import EmbeddingModelName, InferenceType

class BaseOptions(BaseSettings):
    """
    Configuration settings class for the lettuce pipeline.
    
    This class manages all configuration options for the lettuce system, including
    database connections, LLM model settings, embedding configurations, and inference
    parameters. It uses Pydantic BaseSettings to handle environment variable loading
    and configuration validation.
    
    The class automatically loads settings from environment variables and .env files,
    providing sensible defaults for all configuration options.
    
    Attributes
    ----------
    db_host : str
        Database host address. Defaults to "localhost".
    db_user : str  
        Database username. Defaults to "postgres".
    db_password : str
        Database password. Defaults to "password".
    db_name : str
        Database name. Defaults to "omop".
    db_port : int
        Database port. Defaults to 5432.
    db_schema : str
        Database schema name. Defaults to "cdm".
    db_vectable : str
        Name of the vector embeddings table. Defaults to "embeddings".
    db_vecsize : int
        Dimension size of embedding vectors. Defaults to 384.
    inference_type : InferenceType
        Type of inference backend to use. Defaults to InferenceType.OLLAMA.
    ollama_url : str
        URL for Ollama server. Defaults to "http://localhost:11434".
    llm_model : LLMModel
        LLM model to use for inference. Defaults to LLMModel.LLAMA_3_1_8B.
    temperature : float
        Sampling temperature for LLM generation. Defaults to 0.0.
    local_llm : str | None
        Path to local LLM weights file. Defaults to None.
    debug_prompt : bool
        Enable prompt debugging output. Defaults to False.
    embedding_model : EmbeddingModelName
        Embedding model to use. Defaults to EmbeddingModelName.BGESMALL.
    embedding_top_k : int
        Number of top embeddings to retrieve. Defaults to 5.
    auth_api_key : str | None
        API key for authentication. Defaults to None.
    """
    
    model_config = SettingsConfigDict(
            env_file=".env",
env_file_encoding="utf-8",
            )
    
    # Database configuration
    db_host: str = "localhost"
    db_user: str = "postgres"
    db_password: str = "password"
    db_name: str = "omop"
    db_port: int = 5432
    db_schema: str = "cdm"
    db_vectable: str = "embeddings"
    # have a branch where this comes from database - hard to integrate,#TODO
    db_vecsize: int = 384 

    # Inference configuration
    inference_type: InferenceType = InferenceType.OLLAMA
    ollama_url: str = "http://localhost:11434"    

    # LLM model configuration
    llm_model: LLMModel = LLMModel.LLAMA_3_1_8B
    temperature: float = 0.0
    local_llm: str | None = None
    debug_prompt: bool = False

    # Embedding configuration
    embedding_model: EmbeddingModelName = EmbeddingModelName.BGESMALL
    embedding_top_k: int = 5
    
    # Authentication
    auth_api_key: str | None = None

    def connection_url(self) -> str:
        """
        Generate a PostgreSQL connection URL from the database configuration.
        
        Constructs a connection string in the format required by SQLAlchemy
        and other database libraries using the configured database parameters.
        
        Returns
        -------
        str
            PostgreSQL connection URL in the format:
            "postgresql://user:password@host:port/database"
            
        Examples
        --------
        >>> settings = BaseOptions()
        >>> settings.connection_url()
        'postgresql://postgres:password@localhost:5432/omop'
        """
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

    def print(self) -> None:
        """
        Print all configuration settings in a formatted display.
        
        Outputs all current configuration values in a readable format,
        useful for debugging and verification of loaded settings.
        
        Examples
        --------
        >>> options = BaseOptions()
        >>> options.print()
        ------------ Options -------------
        db_host: localhost
        db_user: postgres
        db_password: password
        ...
        -------------- End ---------------
        """
        print("------------ Options -------------")
        for k, v in self.model_dump().items():
            print(f"{str(k)}: {str(v)}")
        print("-------------- End ---------------")

    def hf_hub_config(self) -> dict[str, str]:
        return {
                "repo_id": self.llm_model.repo_id,
                "filename": self.llm_model.filename,
                }
        
