import pytest
from components.result import LettuceResult 
from options.base_options import BaseOptions 
from cli.main import main


@pytest.fixture
def mock_args():
    return {
        'informal_names': ['aspirin', 'tylenol'],
        'vector_search': True,
        'use_llm': True,
        'llm_model': 'GPT4',  # Assuming this is a valid enum value
        'temperature': 0.7,
        'vocabulary_id': 'RxNorm',
        'search_threshold': 0.8
    }

@pytest.fixture
def mock_base_options(mock_args, mocker):
    mock_options = mocker.patch('options.base_options')
    mock_instance = mock_options.return_value
    mock_instance.parse.return_value = type('Args', (), mock_args)()
    return mock_options


def test_main_with_vector_search_and_llm(mock_base_options, mocker):
    # Mock dependencies
    mock_time = mocker.patch('..cli.main.time.time', side_effect=[1, 2, 3, 4])
    mock_pipeline = mocker.patch('..options.LLMPipeline')
    mock_omop = mocker.patch('..components.pipeline')

    # Setup mock pipeline
    mock_rag_assistant = mocker.MagicMock()
    mock_rag_assistant.run.return_value = {
        'retriever': {'documents': [
            mocker.Mock(content='Aspirin info', score=0.95),
            mocker.Mock(content='Tylenol info', score=0.90)
        ]},
        'llm': {'replies': ['Aspirin is a pain reliever']}
    }
    mock_pipeline.return_value.get_rag_assistant.return_value = mock_rag_assistant

    # Mock OMOP
    mock_omop.run.return_value = [
        [{'concept_id': 123, 'score': 0.95}],
        [{'concept_id': 456, 'score': 0.90}]
    ]

    # Capture print output
    with mocker.patch('builtins.print') as mock_print:
        main()
    
    # Assertions
    assert mock_rag_assistant.warm_up.called
    assert mock_pipeline.call_count == 1
    assert mock_omop.run.called
    assert mock_print.called
    assert mock_time.call_count == 4  