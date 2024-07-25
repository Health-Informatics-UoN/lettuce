base_options
============

.. py:module:: base_options


Classes
-------

.. autoapisummary::

   base_options.BaseOptions


Module Contents
---------------

.. py:class:: BaseOptions

   This class defines options used during all types of experiments.
   It also implements several helper functions such as parsing, printing, and saving the options.


   .. py:method:: initialize() -> None

      Initializes the BaseOptions class

      Parameters
      ----------
      None

      Returns
      -------
      None



   .. py:method:: parse() -> argparse.Namespace

      Parses the arguments passed to the script

      Parameters
      ----------
      None

      Returns
      -------
      opt: argparse.Namespace
          The parsed arguments



   .. py:method:: _print(args: Dict) -> None

      Prints the arguments passed to the script

      Parameters
      ----------
      args: dict
          The arguments to print

      Returns
      -------
      None



