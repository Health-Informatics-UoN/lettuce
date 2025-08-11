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



# class BaseOptions:
#     """
#     This class defines options used during all types of experiments.
#     It also implements several helper functions such as parsing, printing, and saving the options.
#     """
#
#     def __init__(self) -> None:
#         """
#         Initializes the BaseOptions class
#
#         Parameters
#         ----------
#         None
#
#         Returns
#         -------
#         None
#         """
#         self._parser = argparse.ArgumentParser()
#         self._initialized = False
#
#     def initialize(self) -> None:
#         """
#         Initializes the BaseOptions class
#
#         Parameters
#         ----------
#         None
#
#         Returns
#         -------
#         None
#         """
#         self._parser.add_argument(
#             "--llm_model",
#             type=str,
#             required=False,
#             default="LLAMA_3_1_8B",
#             choices=[llm.name for llm in LLMModel],
#         )
#
#         self._parser.add_argument(
#             "--embedding_model",
#             type=lambda s: EmbeddingModelName[s],
#             required=False,
#             default="BGESMALL",
#             choices=[model.name for model in EmbeddingModelName],
#         )
#
#         self._parser.add_argument(
#             "--embed-vocab",
#             type=lambda s: s.split(",") if s else None,
#             default=None,
#             required=False,
#             help="Vocabulary IDs for embedding filtering. If multiple vocabularies, supply a comma-separated list"
#         )
#
#         self._parser.add_argument(
#                 "--standard-concept",
#                 type=bool,
#                 default=False,
#                 required=False,
#                 help="Whether to filter output by the standard_concept field of the concept table"
#         )
#
#         self._parser.add_argument(
#             "--temperature",
#             type=float,
#             required=False,
#             default=0.0,
#             help="temperature to control LLM output randomness",
#         )
#
#         self._parser.add_argument(
#             "--informal_names",
#             type=str,
#             nargs="+",
#             required=True,
#             help="informal medication names",
#         )
#
#         self._parser.add_argument(
#             "--vocabulary_id",
#             type=lambda s: s.split(",") if s else None,
#             required=False,
#             default=None,
#             help="Vocabulary IDs to be queried. If you want multiple"
#             "vocabularies to be used, supply a comma separated list",
#         )
#
#         self._parser.add_argument(
#             "--concept_ancestor",
#             action=argparse.BooleanOptionalAction,
#             required=False,
#             help="concept ancestor",
#         )
#
#         self._parser.add_argument(
#             "--concept_relationship",
#             action=argparse.BooleanOptionalAction,
#             required=False,
#             help="concept relationship",
#         )
#
#         self._parser.add_argument(
#             "--concept_synonym",
#             action=argparse.BooleanOptionalAction,
#             required=False,
#             help="concept synonym",
#         )
#
#         self._parser.add_argument(
#             "--search_threshold",
#             type=int,
#             required=False,
#             default=80,
#             help="search threshold",
#         )
#
#         self._parser.add_argument(
#             "--max_separation_descendants",
#             type=int,
#             required=False,
#             default=1,
#             help="max separation descendants",
#         )
#
#         self._parser.add_argument(
#             "--max_separation_ancestor",
#             type=int,
#             required=False,
#             default=1,
#             help="max separation ancestor",
#         )
#
#         self._parser.add_argument(
#             "--vector_search",
#             action=argparse.BooleanOptionalAction,
#             required=False,
#             default=True,
#             help="Try vector search before LLM?",
#         )
#
#         self._parser.add_argument(
#             "--use_llm",
#             action=argparse.BooleanOptionalAction,
#             required=False,
#             default=True,
#             help="Use LLM?",
#         )
#
#         self._parser.add_argument(
#                 "--embedding-top-k",
#                 type=int,
#                 required=False,
#                 default=5,
#                 help="Number of suggestions to return from vector search for RAG."
#          )
#
#         self._initialized = True
#
#     def parse(self) -> argparse.Namespace:
#         """
#         Parses the arguments passed to the script
#
#         Parameters
#         ----------
#         None
#
#         Returns
#         -------
#         opt: argparse.Namespace
#             The parsed arguments
#         """
#         if not self._initialized:
#             self.initialize()
#         self._opt, _ = self._parser.parse_known_args()
#
#         args = vars(self._opt)
#         self._print(args)
#
#         return self._opt
#
#     def _print(self, args: Dict) -> None:
#         """
#         Prints the arguments passed to the script
#
#         Parameters
#         ----------
#         args: dict
#             The arguments to print
#
#         Returns
#         -------
#         None
#         """
