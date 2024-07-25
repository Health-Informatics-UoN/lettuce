import argparse
import ast
from typing import Dict


class BaseOptions:
    """
    This class defines options used during all types of experiments.
    It also implements several helper functions such as parsing, printing, and saving the options.
    """

    def __init__(self) -> None:
        """
        Initializes the BaseOptions class

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._parser = argparse.ArgumentParser()
        self._initialized = False

    def initialize(self) -> None:
        """
        Initializes the BaseOptions class

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._parser.add_argument(
            "--llm_model",
            type=lambda x: ast.literal_eval(x),
            required=False,
            default={"hub": "LlamaCpp", "model_name": "llama-2-7B-chat"},
            choices=[
                {"OpenAI", "gpt-3.5-turbo-0125"},
                {"OpenAI", "gpt-4"},
                {"LlamaCpp", "llama-2-7B"},
                {"LlamaCpp", "llama-2-7B-chat"},
                {"LlamaCpp", "llama-2-13B"},
                {"LlamaCpp", "llama-2-13B-chat"},
                {"LlamaCpp", "llama-2-70B-chat"},
                {"GPT4All", "mistral-7b-openorca"},  # Best overall fast chat model
                {
                    "GPT4All",
                    "mistral-7b-instruct",
                },  # Best overall fast instruction following model
                {
                    "GPT4All",
                    "gpt4all-falcon-newbpe",
                },  # Very fast model with good quality
            ],
        )

        self._parser.add_argument(
            "--temperature",
            type=float,
            required=False,
            default=0.7,
            help="temperature to control LLM output randomness",
        )

        self._parser.add_argument(
            "--chunk_size",
            type=int,
            required=False,
            default=2000,
            help="chunk size for text splitting",
        )

        self._parser.add_argument(
            "--chunk_overlap",
            type=int,
            required=False,
            default=200,
            help="chunk overlap for text splitting",
        )
        self._parser.add_argument(
            "--df_chunk_size",
            type=int,
            required=False,
            default=20,
            help="chunk size for dataframe splitting",
        )

        self._parser.add_argument(
            "--visualize_chunk",
            type=bool,
            required=False,
            default=True,
            help="whether to visualize the output chunk by chunk",
        )

        self._parser.add_argument(
            "--use_memory",
            type=bool,
            required=False,
            default=False,
            help="whether to use memory in the conversation",
        )

        self._parser.add_argument(
            "--use_simple_prompt",
            type=bool,
            required=False,
            default=False,
            help="whether to use simple prompt in the conversation",
        )

        self._initialized = True

    def parse(self) -> argparse.Namespace:
        """
        Parses the arguments passed to the script

        Parameters
        ----------
        None

        Returns
        -------
        opt: argparse.Namespace
            The parsed arguments
        """
        if not self._initialized:
            self.initialize()
        self._opt = self._parser.parse_args()

        args = vars(self._opt)
        # self._print(args)

        return self._opt

    def _print(self, args: Dict) -> None:
        """
        Prints the arguments passed to the script

        Parameters
        ----------
        args: dict
            The arguments to print

        Returns
        -------
        None
        """
        print("------------ Options -------------")
        for k, v in args.items():
            print(f"{str(k)}: {str(v)}")
        print("-------------- End ---------------")
