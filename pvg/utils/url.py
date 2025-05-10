import requests


def url_exists(url: str) -> bool:
    """
    Checks if a URL exists by making a HEAD request.
    Parameters:
    - url (str): The URL to check.
    Returns:
    - bool: True if the URL exists and does not redirect to a different URL, False otherwise.
    """
    try:
        response = requests.head(url, allow_redirects=True)
        # Check if the status code is less than 400 and the URL has not changed
        return response.status_code < 400 and response.url == url
    except requests.RequestException:
        # In case of network problems, SSL errors, etc.
        return False
