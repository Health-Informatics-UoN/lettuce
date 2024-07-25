import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_experimental.agents import create_csv_agent
from utils.utils import load_csv_file_names
from llm.models import get_model


class Agents:
    def __init__(self, opt):
        self._opt = opt
        self._llm_model = opt.llm_model["model"]
        self._llm_model_name = opt.llm_model["model_name"]
        self._llm_temperature = opt.llm_temperature
        self._llm = get_model(
            self._llm_model, self._llm_model_name, temperature=self._llm_temperature
        )
        self._agent = None

    def get_csv_agent(self):
        self._agent = self._csv_agent()

    def _csv_agent(self):
        files = load_csv_file_names(self._opt.main_data_folder)
        agent = create_csv_agent(
            llm=self._llm,
            path=files,
            pandas_kwargs={"delimiter": "\t"},
            agent_type="zero-shot-react-description",
            verbose=True,
        )

        return agent

    def invoke(self, question):
        return self._agent.invoke(question)
