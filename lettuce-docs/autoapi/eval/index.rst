:py:mod:`eval`
==============

.. py:module:: eval


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   eval.EvaluationFramework
   eval.Metric
   eval.TestPipeline
   eval.PipelineTest
   eval.SingleResultMetric
   eval.InformationRetrievalMetric
   eval.SingleResultPipeline
   eval.SingleResultPipelineTest




.. py:class:: EvaluationFramework(results_file='results.json')


   .. py:method:: run_evaluations()


   .. py:method:: _save_evaluations()



.. py:class:: Metric


   Bases: :py:obj:`abc.ABC`

   Base class for all metrics.

   .. py:method:: calculate(*args, **kwargs)
      :abstractmethod:

      Calculate the metric value.



.. py:class:: TestPipeline


   Bases: :py:obj:`abc.ABC`

   Base class for Pipeline runs

   .. py:method:: run(*args, **kwargs)
      :abstractmethod:

      Run the pipeline



.. py:class:: PipelineTest(name: str, pipeline: TestPipeline, metrics: list[Metric])


   Bases: :py:obj:`abc.ABC`

   Base class for Pipeline tests

   .. py:method:: run_pipeline(*args, **kwargs)
      :abstractmethod:


   .. py:method:: evaluate(*args, **kwargs)
      :abstractmethod:



.. py:class:: SingleResultMetric


   Bases: :py:obj:`Metric`

   Metric for evaluating pipelines that return a single result.


.. py:class:: InformationRetrievalMetric


   Bases: :py:obj:`Metric`

   Metric for evaluating information retrieval pipelines.


.. py:class:: SingleResultPipeline


   Bases: :py:obj:`TestPipeline`

   Base class for pipelines returning a single result


.. py:class:: SingleResultPipelineTest(name: str, pipeline: SingleResultPipeline, metrics: list[SingleResultMetric])


   Bases: :py:obj:`PipelineTest`

   Base class for Pipeline tests


