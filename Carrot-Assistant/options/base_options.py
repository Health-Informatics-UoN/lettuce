import argparse
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
            type=str,
            required=False,
            default="LLAMA_3_8B",
            choices=[
                "GPT_3_5_TURBO",
                "GPT_4",
                "LLAMA_2_7B",
                "LLAMA_3_8B",
                "LLAMA_3_70B",
                "GEMMA_7B",
                "LLAMA_3_1_8B",
                "MISTRAL_7B",
                "PYTHIA_70M",
                "PYTHIA_410M",
                "PYTHIA_1B",
                "PYTHIA_1_4B",
                "PYTHIA_2_8B",
                "ALPACA_LORA_7B",
            ],
        )

        self._parser.add_argument(
            "--embedding_model",
            type=str,
            required=False,
            default="BGESMALL",
            choices=[
                "BGESMALL",
                "MINILM",
                "GTR_T5_BASE",
                "GTR_T5_LARGE",
                "E5_BASE",
                "E5_LARGE",
                "DISTILBERT_BASE_UNCASED",
                "DISTILUSE_BASE_MULTILINGUAL",
                "CONTRIEVER",
            ],
        )

        self._parser.add_argument(
            "--temperature",
            type=float,
            required=False,
            default=0.0,
            help="temperature to control LLM output randomness",
        )

        self._parser.add_argument(
            "--informal_names",
            type=str,
            nargs="+",
            required=False,
            default=["Omepra", "paracetamol"],
            help="informal medication names",
        )

        self._parser.add_argument(
            "--vocabulary_id",
            type=lambda s: s.split(","),
            #  If we want to have multiple vocabularies (which we might),
            # then we can make this lamdba s: s.split(','),
            # and have them separated by commas (I think).
            # type=str,
            required=False,
            default="RxNorm",
            help="Vocabulary IDs to be queried. If you want multiple"
            "vocabularies to be used, supply a comma separated list",
        )

        self._parser.add_argument(
            "--concept_ancestor",
            type=str,
            required=False,
            default="n",
            choices=["y", "n"],
            help="concept ancestor",
        )

        self._parser.add_argument(
            "--concept_relationship",
            type=str,
            required=False,
            default="n",
            choices=["y", "n"],
            help="concept relationship",
        )

        self._parser.add_argument(
            "--concept_synonym",
            type=str,
            required=False,
            default="n",
            choices=["y", "n"],
            help="concept synonym",
        )

        self._parser.add_argument(
            "--search_threshold",
            type=int,
            required=False,
            default=80,
            help="search threshold",
        )

        self._parser.add_argument(
            "--max_separation_descendants",
            type=int,
            required=False,
            default=1,
            help="max separation descendants",
        )

        self._parser.add_argument(
            "--max_separation_ancestor",
            type=int,
            required=False,
            default=1,
            help="max separation ancestor",
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
        self._opt, _ = self._parser.parse_known_args()

        args = vars(self._opt)
        self._print(args)

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
