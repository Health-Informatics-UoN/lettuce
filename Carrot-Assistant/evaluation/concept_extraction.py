import time
from typing import List, Dict, Any
import assistant

from dotenv import load_dotenv
from utils.logging_utils import Logger
from omop import OMOP_match
from options.base_options import BaseOptions
import sys
import os

import pandas as pd

# Add the root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class OmopConceptNameTesting:
    """
    This class is used to extract the expected results from the test dataset.
    
    Parameters:
    -----------
    logger: Logger
        The logger object to use for logging
    """
    def __init__(self, logger: Logger = None):
        if logger is None:
            logger = Logger().make_logger()
        self.logger = logger
        load_dotenv()

        # Initialize BaseOptions and assign it to an instance variable
        self.opt = BaseOptions()
        self.opt.initialize()
        self.opt = self.opt.parse()

    def query_omop_concept(self, informal_names: List[str]) -> List[Dict[str, Any]]:
        """
        This function is used to query the OMOP database for the informal names.
        
        Parameters:
        -----------
        informal_names: List[str]
            A list of informal names to query the OMOP database.
            
        Returns:
        --------
        results: List[Dict[str, Any]]
            A list of dictionaries containing the results of the OM
            
        Workflow:
        ---------
        1. Query the OMOP database for each informal name.
        2. If a match is found, add the match to the results.
        3. If no match is found, add the name to the no_omop_matches list.
        4. Return the results and the no_omop_matches list.
        
        Raises:
        -------
        ValueError:
            If there is an error querying the OMOP database.
        """
        try: 
            results = []
            no_omop_matches = []

            for name in informal_names:
                self.logger.info(f"Querying OMOP database for informal name: {name}")
                omop_output = OMOP_match.run(self.opt, name, self.logger)

                if omop_output and any(concept["CONCEPT"] for concept in omop_output):
                    matched_name = omop_output[0]["CONCEPT"][0]["concept_name"]  
                    result = {"informal_name": name, "match": "Yes", "informal_match_name": matched_name}
                    self.logger.info(f"Match found for {name}: {matched_name}")
                else:
                    result = {"informal_name": name, "match": "No", "informal_match_name": "no_match"}
                    self.logger.info(f"No match found for {name}")
                    no_omop_matches.append(name)

                results.append(result)

            return results, no_omop_matches
        
        except Exception as e:
            raise ValueError(f"Error querying OMOP database: {e}")
    
    
    def llm_all_concept_name(self, informal_names: List[str]) -> List[Dict[str, str]]:
        """
        This function is used to predict the formal names for all informal names using LLM.
        
        Parameters:
        -----------
        informal_names: List[str]
            A list of informal names to predict the formal names.
            
        Returns:
        --------
        all_llm_results: List[Dict[str, str]]
            A list of dictionaries containing the informal and formal names predicted
            by LLM.
            
        Workflow:
        ---------
        1. Predict the formal names for all informal names using LLM.
        2. Add the results to the all_llm_results list.
        3. Return the all_llm_results list.
        
        Raises:
        -------
        ValueError:
            If there is an error predicting the formal names using LLM.
        """
        try:
            
            all_llm_results = []
            for name in informal_names:
                llm_output = assistant.run(
                    opt=self.opt, informal_names=[name], logger=self.logger
                )

                if llm_output:
                    formal_name = llm_output[0]["reply"].splitlines()[
                        0
                    ] 
                    formal_name = formal_name.replace(
                        "Response:", ""
                    ).strip()  
                    self.logger.info(f"LLM output for {name}: {formal_name}")
                    all_llm_results.append({"informal_name": name, "formal_name": formal_name})  

            return all_llm_results
        
        except Exception as e:
            raise ValueError(f"Error predicting formal names using LLM: {e}")

    def llm_concept_name(self, non_matches: List[str]) -> List[Dict[str, str]]:
        """
        This function is used to predict the formal names for non-matching names using LLM.
        
        Parameters:
        -----------
        non_matches: List[str]
            A list of non-matching names to predict the formal names.
            
        Returns:
        --------
        llm_results: List[Dict[str, str]]
            A list of dictionaries containing the non-matching names and their formal names
            predicted by LLM.
            
        Workflow:
        ---------
        1. Predict the formal names for non-matching names using LLM.
        2. Add the results to the llm_results list.
        3. Return the llm_results list.
        
        Raises:
        -------
        ValueError:
            If there is an error predicting the formal names using LLM.
        """
        
        try:
            
            llm_results = []

            for name in non_matches:
                llm_output = assistant.run(
                    opt=self.opt, informal_names=[name], logger=self.logger
                )

                if llm_output:
                    formal_name = llm_output[0]["reply"].splitlines()[
                        0
                    ]  
                    formal_name = formal_name.replace(
                        "Response:", ""
                    ).strip()  
                    self.logger.info(f"LLM output for {name}: {formal_name}")
                    llm_results.append({"informal_name": name, "formal_name": formal_name})

            return llm_results

        except Exception as e:
            raise ValueError(f"Error predicting formal names using LLM: {e}")
        
    def query_omop_with_formal_names(
        self, formal_names: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """
        This function is used to query the OMOP database for the formal names.
        
        Parameters:
        -----------
        formal_names: List[Dict[str, str]]
            A list of dictionaries containing the formal names to query the OMOP database.
            
        Returns:
        --------
        formal_name_results: List[Dict[str, Any]]
            A list of dictionaries containing the formal names and whether they matched
            in the OMOP database.
            
        Workflow:
        ---------
        1. Query the OMOP database for each formal name.
        2. If a match is found, add the match to the formal_name_results.
        3. Return the formal_name_results.
        
        Raises:
        -------
        ValueError:
            If there is an error querying the OMOP database.
        """
        try:
            
            formal_name_results = []

            for item in formal_names:
                formal_name = item["formal_name"]
                self.logger.info(f"Querying OMOP database for formal name: {formal_name}")
                omop_output = OMOP_match.run(self.opt, formal_name, self.logger)

                if omop_output and any(concept["CONCEPT"] for concept in omop_output):
                    result = {"formal_name": formal_name, "omop_match": "Yes"}
                    self.logger.info(f"Match found for {formal_name}: Yes")
                else:
                    result = {"formal_name": formal_name, "omop_match": "No"}
                    self.logger.info(f"Match found for {formal_name}: No")

                formal_name_results.append(result)

            return formal_name_results
        
        except Exception as e:
            raise ValueError(f"Error querying OMOP database for formal names: {e}")

    def run(self, informal_names: List[str]):
        """
        This function is used to run the OMOP concept name testing.
        
        Parameters:
        -----------
        informal_names: List[str]
            A list of informal names to test.
            
        Returns:
        --------
        results: List[Dict[str, Any]]
            A list of dictionaries containing the results of the OMOP query.
            
        Workflow:
        ---------
        1. Query the OMOP database for the informal names.
        2. Predict the formal names for all informal names using LLM.
        3. Predict the formal names for non-matching names using LLM.
        4. Query the OMOP database with the formal names from LLM results.
        5. Return the results, llm_results, formal_name_results, and llm_all_results.
        """
        start_time = time.time()
        self.logger.info(
            f"Starting OMOP concept name testing for {len(informal_names)} names."
        )

        results, no_omop_matches = self.query_omop_concept(informal_names)

        llm_results = []
        formal_name_results = []

        # Predict for all informal names using LLM
        llm_all_results = self.llm_all_concept_name(informal_names)

        if no_omop_matches:
            llm_results = self.llm_concept_name(no_omop_matches)

            # Query OMOP with formal names from LLM results
            formal_name_results = self.query_omop_with_formal_names(llm_results)

        self.logger.info(
            f"Completed OMOP concept name testing in {time.time() - start_time} seconds."
        )

        print("\n" + "=" * 50)
        print("OMOP Query Results:")
        print("=" * 50)
        print(results)
        print("=" * 50)

        print("\n" + "=" * 50)
        print("LLM Query Results (for non-matching names):")
        print("=" * 50)
        print(llm_all_results)
        print("=" * 50 + "\n")

        print("\n" + "=" * 50)
        print("LLM Query Results (for all informal names):")
        print("=" * 50)
        print(llm_results)
        print("=" * 50 + "\n")

        print("\n" + "=" * 50)
        print("OMOP Query Results for Formal Names (LLM Output):")
        print("=" * 50)
        print(formal_name_results)
        print("=" * 50)

        return results, llm_results, formal_name_results, llm_all_results


    def create_concept_extracted_dataframe(
    self,
    results: List[Dict[str, Any]],
    llm_results: List[Dict[str, str]],
    formal_name_results: List[Dict[str, Any]],
    llm_all_results: List[Dict[str, str]],
) -> pd.DataFrame:
        
        """
        This function is used to create a DataFrame of the extracted results.
        
        Parameters:
        -----------
        results: List[Dict[str, Any]]
            A list of dictionaries containing the results of the OMOP query.
            
        llm_results: List[Dict[str, str]]
            A list of dictionaries containing the non-matching names and their formal names
            predicted by LLM.
            
        formal_name_results: List[Dict[str, Any]]
            A list of dictionaries containing the formal names and whether they matched
            in the OMOP database.
            
        llm_all_results: List[Dict[str, str]]
            A list of dictionaries containing the informal and formal names predicted
            by LLM.
            
        Returns:
        --------
        extracted_df: pd.DataFrame
            The DataFrame containing the extracted results
        """
        # Initialize the dataframe with the desired columns
        data = []

        # Convert the results into dictionaries for easy access
        formal_name_dict = {
            item["formal_name"]: item["omop_match"] for item in formal_name_results
        }
        
        llm_dict = {item["informal_name"]: item["formal_name"] for item in llm_results}
        llm_all_dict = {item["informal_name"]: item["formal_name"] for item in llm_all_results}

        for result in results:
            informal_name = result["informal_name"]
            informal_omop_match = result["match"]
            informal_match_name = result["informal_match_name"]

            # Determine if LLM was used and retrieve its output
            llm_querying = "Yes" if informal_name in llm_dict else "Not using LLM"
            llm_predicted_name = llm_dict.get(informal_name, "Not using LLM")
            llm_omop_match = formal_name_dict.get(llm_predicted_name, "Not using LLM")
            
            # LLM prediction for all informal names
            llm_all_predicted_name = llm_all_dict.get(informal_name, "Not available")

            # Append the row to data
            data.append(
                {
                    "Informal Name": informal_name,
                    "Informal OMOP Match": informal_omop_match,
                    "Informal Match Name": informal_match_name, 
                    "LLM Querying": llm_querying,
                    "LLM Predicted Name": llm_predicted_name,
                    "LLM OMOP Match": llm_omop_match,
                    "LLM All Predicted Name": llm_all_predicted_name, 
                }
            )

        extracted_df = pd.DataFrame(data)
        
        pd.set_option('display.max_columns', None) 
        pd.set_option('display.max_rows', None)     
        pd.set_option('display.width', None)        
        pd.set_option('display.max_colwidth', None) 

        print("\n" + "=" * 50)
        print("Dataframe of Extracted Results:")
        print("=" * 50)
        print(extracted_df)
        print("=" * 50)

        return extracted_df

"""
# Test Cases:
if __name__ == "__main__":
    # Example usage
    opt = BaseOptions().parse()
    informal_names = opt.informal_names

    tester = OmopConceptNameTesting()
    results, llm_results, formal_name_results, llm_all_results = tester.run(informal_names)

    for result in results:
        print(f"Informal Name: {result['informal_name']}, Match: {result['match']}")

    if llm_results:
        print("LLM Results:")
        for result in llm_results:
            print(
                f"Informal Name: {result['informal_name']}, Formal Name: {result['formal_name']}"
            )

    if formal_name_results:
        print("OMOP Results for Formal Names:")
        for result in formal_name_results:
            print(
                f"Formal Name: {result['formal_name']}, OMOP Match: {result['omop_match']}"
            )


    extracted_df_results = tester.create_concept_extracted_dataframe(
            results, llm_results, formal_name_results, llm_all_results 
        )
    print(extracted_df_results)

   




    # -> Use this command to run this script:
    # python -m evaluation.concept_extraction


# query_omop_concept

result = {"informal_name": name, "match": "Yes"}
self.logger.info(f"Match found for {name}: Yes")


"""