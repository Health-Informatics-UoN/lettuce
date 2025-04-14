from itertools import cycle
import pytest
from unittest.mock import patch, MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import Session as SQLAlchemySession
import os
import sys


pytestmark = pytest.mark.skip(reason="Skipping CLI tests")


@pytest.fixture(scope="session", autouse=True)
def mock_sqlalchemy_engine():
    env_vars = {
        "DB_HOST": "mock_db",
        "DB_USER": "mock_user",
        "DB_PASSWORD": "mock_pass",
        "DB_PORT": "5432",
        "DB_NAME": "mock_db_name",
        "DB_SCHEMA": "mock_schema"
    }
    with patch.dict(os.environ, env_vars):
        mock_engine = create_engine("sqlite:///:memory:")
        with patch('sqlalchemy.create_engine', return_value=mock_engine) as mock_create:
            print("Mock SQLAlchemy engine applied in test_cli.py")
            from lettuce.omop import db_manager
            db_manager.engine = mock_engine
            mock_session = MagicMock(spec=SQLAlchemySession)
            db_manager.db_session = mock_session
            yield mock_create

@pytest.fixture
def mock_args():
    return {
        'informal_names': ['aspirin', 'tylenol'],
        'vector_search': True,
        'use_llm': True,
        'llm_model': 'LLAMA_3_1_8B',
        'temperature': 0.7,
        'vocabulary_id': 'RxNorm',
        'search_threshold': 0.8,
        'concept_ancestor': False,
        'concept_relationship': False,
        'concept_synonym': False,
        'max_separation_descendants': None,
        'max_separation_ancestor': None
    }

@pytest.fixture
def mock_base_options(mock_args):
    with patch('lettuce.cli.main.BaseOptions', autospec=True) as mock_options_class:
        mock_instance = mock_options_class.return_value
        mock_instance.parse.return_value = type('Args', (), mock_args)()
        yield mock_options_class

@pytest.fixture
def mock_llm_pipeline():
    print("Applying mock_llm_pipeline fixture")
    # Create the mock class and instance
    mock_pipeline_class = MagicMock()
    mock_pipeline_instance = MagicMock()
    
    # Set up the RAG assistant mock
    mock_rag_assistant = MagicMock()
    mock_rag_assistant.run.return_value = {
        'retriever': {
            'documents': [
                MagicMock(content='Aspirin info', score=0.95),
                MagicMock(content='Tylenol info', score=0.90)
            ]
        },
        'llm': {
            'replies': ['Aspirin is a pain reliever']
        }
    }
    mock_rag_assistant.warm_up.return_value = None
    
    # Configure the instance methods
    mock_pipeline_instance.get_rag_assistant.return_value = mock_rag_assistant
    mock_pipeline_instance.get_simple_assistant.return_value = MagicMock()
    
    # Define the side_effect for instantiation
    def create_mock_instance(*args, **kwargs):
        print("Mock LLMPipeline instantiated with args:", args, "kwargs:", kwargs)
        return mock_pipeline_instance
    
    mock_pipeline_class.side_effect = create_mock_instance
    
    # Return a tuple of the class mock and instance for flexibility
    return mock_pipeline_class, mock_pipeline_instance

def test_main_with_vector_search_and_llm(mock_base_options, mock_llm_pipeline):
    mock_llm_class, mock_pipeline_instance = mock_llm_pipeline  # Unpack the fixture
    
    with patch('time.time') as mock_time, \
         patch('builtins.print') as mock_print, \
         patch('huggingface_hub.snapshot_download', side_effect=lambda *args, **kwargs: AssertionError("Hugging Face snapshot download detected")), \
         patch('requests.get', side_effect=lambda *args, **kwargs: AssertionError("HTTP request detected")), \
         patch('lettuce.cli.main.LLMPipeline', autospec=True) as mock_llm_patch, \
         patch('lettuce.cli.main.run') as mock_omop_run:
        
        # Configure the patch to use the fixture's mock behavior
        mock_llm_patch.side_effect = mock_llm_class.side_effect
        
        # Mock run() return value
        mock_omop_run.return_value = [
            {'search_term': 'aspirin', 'CONCEPT': [{'concept_id': 123, 'concept_name': 'Aspirin'}]},
            {'search_term': 'tylenol', 'CONCEPT': [{'concept_id': 456, 'concept_name': 'Tylenol'}]}
        ]
        
        # Debug: Verify args values
        args = mock_base_options.return_value.parse.return_value
        sys.stdout.write(f"Debug: vector_search={args.vector_search}, use_llm={args.use_llm}, condition={args.vector_search & args.use_llm}\n")
        sys.stdout.flush()
        
        from lettuce.cli.main import main
        
        mock_time.side_effect = cycle([1, 2, 3, 4, 5, 6])
        main()
        
        assert mock_base_options.return_value.parse.return_value.vector_search
        assert mock_base_options.return_value.parse.return_value.use_llm
        assert mock_print.called
        assert mock_llm_patch.called, "LLMPipeline was not instantiated"
        assert mock_pipeline_instance.get_rag_assistant.called, "get_rag_assistant was not called"


def test_main_with_vector_search_only(mock_base_options, mock_args):
    mock_args_copy = mock_args.copy()
    mock_args_copy['vector_search'] = True
    mock_args_copy['use_llm'] = False 
    mock_base_options.return_value.parse.return_value = type('Args', (), mock_args_copy)()
    
    with patch('time.time'), \
         patch('builtins.print') as mock_print, \
         patch('lettuce.cli.main.Embeddings') as mock_embeddings, \
         patch('lettuce.cli.main.run') as mock_omop_run:
        mock_embeddings_instance = mock_embeddings.return_value
        mock_embeddings_instance.search.return_value = [[{'content': 'Aspirin info'}], [{'content': 'Tylenol info'}]]
        mock_omop_run.return_value = [{'search_term': 'aspirin'}, {'search_term': 'tylenol'}]
        
        from lettuce.cli.main import main
        main()
        
        assert mock_embeddings.called
        assert mock_embeddings_instance.search.called
        assert mock_omop_run.called
        assert mock_print.called


def test_main_with_use_llm_only(mock_base_options, mock_args):
    mock_args_copy = mock_args.copy()
    mock_args_copy['vector_search'] = False
    mock_args_copy['use_llm'] = True
    mock_base_options.return_value.parse.return_value = type('Args', (), mock_args_copy)()
    
    with patch('time.time') as mock_time, \
         patch('builtins.print') as mock_print, \
         patch('lettuce.cli.main.LLMPipeline', autospec=True) as mock_llm_patch, \
         patch('lettuce.cli.main.run') as mock_omop_run:
        mock_pipeline_instance = MagicMock()
        mock_simple_assistant = MagicMock()
        mock_simple_assistant.run.return_value = {'llm': {'replies': ['Answer']}}
        mock_pipeline_instance.get_simple_assistant.return_value = mock_simple_assistant
        
        mock_llm_patch.side_effect = lambda *args, **kwargs: mock_pipeline_instance
        
        mock_omop_run.return_value = [{'search_term': 'aspirin'}, {'search_term': 'tylenol'}]
        mock_time.side_effect = cycle([1, 2, 3, 4])
        
        from lettuce.cli.main import main
        main()
        
        assert mock_llm_patch.called
        assert mock_pipeline_instance.get_simple_assistant.called
        assert mock_simple_assistant.warm_up.called
        assert mock_omop_run.called
        assert mock_print.called