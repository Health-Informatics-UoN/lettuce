import json
from typing import Dict


def get_informal_name(json_arr: str) -> str:
    """
    Gets the informal name from the response

    Parameters
    ----------
    json_arr: str
        The json response

    Returns
    -------
    str
        The informal name
    """
    response = json.loads(json_arr)
    return response.get("informal_name")
