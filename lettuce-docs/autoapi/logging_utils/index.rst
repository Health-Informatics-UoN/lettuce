:py:mod:`logging_utils`
=======================

.. py:module:: logging_utils


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   logging_utils.Logger




.. py:class:: Logger(logging_level='INFO', console_logger=True, multi_module=True)


   Bases: :py:obj:`object`

   logger preparation


   Parameters
   ----------
   log_dir: string
       path to the log directory

   logging_level: string
       required Level of logging. INFO, WARNING or ERROR can be selected. Default to 'INFO'

   console_logger: bool
       flag if console_logger is required. Default to False

   Returns
   ----------
   logger: logging.Logger
       logger object

   .. py:method:: _make_level()

      Sets the level of logging

      Uses the logging_level to set own _level property.

      Parameters
      ----------
      None

      Returns
      -------
      None


   .. py:method:: make_logger()

      Constructs a Logger instance.

      Parameters
      ----------
      None

      Returns
      -------
      None



