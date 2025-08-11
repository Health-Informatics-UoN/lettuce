from itertools import cycle
import pytest
from unittest.mock import patch, MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import Session as SQLAlchemySession
import os
from typer.testing import CliRunner


@pytest.fixture(scope="session", autouse=True)
def mock_sqlalchemy_engine():
    env_vars = {
        "DB_HOST": "mock_db",
        "DB_USER": "mock_user",
        "DB_PASSWORD": "mock_pass",
        "DB_PORT": "5432",
        "DB_NAME": "mock_db_name",
        "DB_SCHEMA": "mock_schema",
        "DB_VECTABLE": "mock_table",
        "DB_VECSIZE": "384",
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
def cli_runner():
    """Fixture to provide Typer CLI runner"""
    return CliRunner()

@pytest.fixture
def mock_llm_pipeline():
    print("Applying mock_llm_pipeline fixture")
    mock_pipeline_class = MagicMock()
    mock_pipeline_instance = MagicMock()
    
    mock_rag_assistant = MagicMock()
    mock_rag_assistant.run.return_value = {
        'retriever': {
            'documents': [
                MagicMock(concept='Aspirin info', score=0.95),
                MagicMock(concept='Tylenol info', score=0.90)
            ]
        },
        'llm': {
            'replies': ['Aspirin is a pain reliever']
        }
    }
    mock_rag_assistant.warm_up.return_value = None
    
    mock_pipeline_instance.get_rag_assistant.return_value = mock_rag_assistant
    mock_pipeline_instance.get_simple_assistant.return_value = MagicMock()
    
    def create_mock_instance(*args, **kwargs):
        print("Mock LLMPipeline instantiated with args:", args, "kwargs:", kwargs)
        return mock_pipeline_instance
    
    mock_pipeline_class.side_effect = create_mock_instance
    
    return mock_pipeline_class, mock_pipeline_instance


def test_main_with_vector_search_and_llm(cli_runner, mock_llm_pipeline):
    """Test the CLI command with vector search and LLM enabled"""
    mock_llm_class, mock_pipeline_instance = mock_llm_pipeline
    
    with patch('time.time') as mock_time, \
         patch('huggingface_hub.snapshot_download', side_effect=lambda *args, **kwargs: AssertionError("Hugging Face snapshot download detected")), \
         patch('requests.get', side_effect=lambda *args, **kwargs: AssertionError("HTTP request detected")), \
         patch('lettuce.cli.main.LLMPipeline', autospec=True) as mock_llm_patch, \
         patch('lettuce.cli.main.OMOPMatcher') as mock_OMOPMatcher:
        
        mock_llm_patch.side_effect = mock_llm_class.side_effect
        
        mock_OMOPMatcher.return_value.run.return_value = [
            {'search_term': 'aspirin', 'CONCEPT': [{'concept_id': 123, 'concept_name': 'Aspirin'}]},
            {'search_term': 'tylenol', 'CONCEPT': [{'concept_id': 456, 'concept_name': 'Tylenol'}]}
        ]
        
        mock_time.side_effect = cycle([1, 2, 3, 4, 5, 6])
        
        from lettuce.cli.main import app
        
        result = cli_runner.invoke(app, [
            "aspirin", "tylenol",
            "--search-threshold", "80",
        ])
        
        assert result.exit_code == 0
        assert mock_llm_patch.called, "LLMPipeline was not instantiated"
        assert mock_pipeline_instance.get_rag_assistant.called, "get_rag_assistant was not called"


def test_main_with_vector_search_only(cli_runner):
    """Test the CLI command with only vector search enabled"""
    with patch('time.time'), \
         patch('lettuce.cli.main.Embeddings') as mock_embeddings, \
         patch('lettuce.cli.main.OMOPMatcher') as mock_OMOPMatcher:
        
        mock_OMOPMatcher.return_value.run.return_value = [
            {'search_term': 'aspirin', 'CONCEPT': [{'concept_id': 123, 'concept_name': 'Aspirin'}]},
            {'search_term': 'tylenol', 'CONCEPT': [{'concept_id': 456, 'concept_name': 'Tylenol'}]}
        ]
        
        mock_embeddings_instance = mock_embeddings.return_value
        mock_embeddings_instance.search.return_value = [[{'concept': 'Aspirin info'}], [{'concept': 'Tylenol info'}]]
        
        from lettuce.cli.main import app
        
        result = cli_runner.invoke(app, [
            "aspirin", "tylenol",
            "--vector-search",
            "--no-use-llm",
            "--search-threshold", "80"
        ])
        
        assert result.exit_code == 0
        assert mock_embeddings.called
        assert mock_embeddings_instance.search.called
        assert mock_OMOPMatcher.return_value.run.called


def test_main_with_use_llm_only(cli_runner):
    """Test the CLI command with only LLM enabled"""
    with patch('time.time') as mock_time, \
         patch('lettuce.cli.main.LLMPipeline', autospec=True) as mock_llm_patch, \
         patch('lettuce.cli.main.OMOPMatcher') as mock_OMOPMatcher:
        
        mock_OMOPMatcher.return_value.run.return_value = [
            {'search_term': 'aspirin', 'CONCEPT': [{'concept_id': 123, 'concept_name': 'Aspirin'}]},
            {'search_term': 'tylenol', 'CONCEPT': [{'concept_id': 456, 'concept_name': 'Tylenol'}]}
        ]
        
        mock_pipeline_instance = MagicMock()
        mock_simple_assistant = MagicMock()
        mock_simple_assistant.run.return_value = {'llm': {'replies': ['Answer']}}
        mock_pipeline_instance.get_simple_assistant.return_value = mock_simple_assistant
        
        mock_llm_patch.side_effect = lambda *args, **kwargs: mock_pipeline_instance
        
        mock_time.side_effect = cycle([1, 2, 3, 4])
        
        from lettuce.cli.main import app
        
        result = cli_runner.invoke(app, [
            "aspirin", "tylenol",
            "--no-vector-search",
            "--use-llm",
            "--search-threshold", "80"
        ])
        
        assert result.exit_code == 0
        assert mock_llm_patch.called
        assert mock_pipeline_instance.get_simple_assistant.called
        assert mock_simple_assistant.warm_up.called
        assert mock_OMOPMatcher.return_value.run.called


def test_cli_help(cli_runner):
    """Test that the CLI help message works"""
    from lettuce.cli.main import app
    
    result = cli_runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "help" in result.stdout.lower()


def test_cli_invalid_arguments(cli_runner):
    """Test CLI with invalid arguments"""
    from lettuce.cli.main import app
    
    result = cli_runner.invoke(app, [])
    assert result.exit_code != 0
    
    result = cli_runner.invoke(app, [
        "aspirin",
        "--search-threshold", "invalid_number"
    ])
    assert result.exit_code != 0


def test_cli_output_format(cli_runner):
    """Test that CLI produces expected output format"""
    with patch('lettuce.cli.main.OMOPMatcher') as mock_OMOPMatcher:
        mock_OMOPMatcher.return_value.run.return_value = [
            {'search_term': 'aspirin', 'CONCEPT': [{'concept_id': 123, 'concept_name': 'Aspirin'}]}
        ]
        
        from lettuce.cli.main import app
        
        result = cli_runner.invoke(app, [
            "aspirin",
            "--no-vector-search",
            "--no-use-llm"
        ])
        
        assert result.exit_code == 0
