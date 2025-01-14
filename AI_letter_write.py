import os
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from openai import OpenAI
from typing import Tuple, List, Dict


# Load environment variables
load_dotenv(override=True)

# Constants
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/117.0.0.0 Safari/537.36"
    )
}
SYSTEM_PROMPT = (
    "You are a hardworking student trying to apply for jobs. "
    "Look at the webpage for the company's website and write a strong motivation letter "
    "focusing on specific points mentioned in the page that you will be particularly good at."
)

class OpenAIClient:
    """Handles interactions with the OpenAI API."""

    def __init__(self, api_key: str):
        self._validate_api_key(api_key)
        self.client = OpenAI()

    @staticmethod
    def _validate_api_key(api_key: str) -> None:
        if not api_key:
            raise ValueError("No API key was found.")
        if not api_key.startswith("sk-proj-"):
            raise ValueError("API key doesn't start with 'sk-proj-'. Check your key.")
        if api_key.strip() != api_key:
            raise ValueError("API key has leading/trailing whitespace. Please clean it.")

    def generate_response(self, model: str, messages: List[Dict[str, str]]) -> str:
        try:
            response = self.client.chat.completions.create(model=model, messages=messages)
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Failed to generate response from OpenAI API: {e}")


class Website:
    """Handles fetching and parsing website content."""

    def __init__(self, url: str):
        self.url = url
        self.title, self.text = self._fetch_and_parse()

    def _fetch_and_parse(self) -> Tuple[str, str]:
        try:
            response = requests.get(self.url, headers=HEADERS)
            response.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to fetch the website: {e}")

        soup = BeautifulSoup(response.content, "html.parser")
        title = soup.title.string.strip() if soup.title else "No title found"

        # Clean unwanted elements
        for tag in soup.body(["script", "style", "img", "input"]):
            tag.decompose()

        text = soup.body.get_text(separator="\n", strip=True)
        return title, text


def build_user_prompt(website: Website) -> str:
    """Constructs the user prompt based on the website content."""
    return (
        f"You are looking at a website titled '{website.title}'.\n"
        "The contents of this website are as follows. Please write a motivation letter to apply for a job at this company:\n\n"
        f"{website.text}"
    )


def build_messages(website: Website) -> List[Dict[str, str]]:
    """Creates the message payload for the OpenAI API."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(website)},
    ]


def summarize_website(url: str, openai_client: OpenAIClient) -> str:
    """Fetches a website and generates a motivation letter using OpenAI."""
    website = Website(url)
    messages = build_messages(website)
    return openai_client.generate_response(model="gpt-4o-mini", messages=messages)


def main() -> None:
    """Main entry point for the script."""
    api_key = os.getenv("OPENAI_API_KEY")
    openai_client = OpenAIClient(api_key)
    url = "https://www.vincit.com/insights"  # Change this to the desired URL

    try:
        summary = summarize_website(url, openai_client)
        print(summary)
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
