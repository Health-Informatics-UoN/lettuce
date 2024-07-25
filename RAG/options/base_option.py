import argparse
import os
import sys
from typing import Dict, Union
import ast

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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
            "--experiment_name",
            type=str,
            required=False,
            default="test",
            help="experiment name",
        )

        self._parser.add_argument(
            "--main_data_folder",
            type=str,
            required=False,
            default="/Users/rezachi/Library/CloudStorage/OneDrive-TheUniversityofNottingham/BRC LLM/Data/tmp_data1",
            help="path to data folder",
        )

        self._parser.add_argument(
            "--log_dir", type=str, required=False, default="./logs", help="path to log"
        )

        self._parser.add_argument(
            "--use_multithreading",
            type=bool,
            required=False,
            default=True,
            help="use multithreading",
        )

        self._parser.add_argument(
            "--num_workers",
            type=int,
            required=False,
            default=4,
            help="number of workers",
        )

        self._parser.add_argument(
            "--seed", type=int, required=False, default=1221, help="random seed"
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
