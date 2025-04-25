"""
A simple web crawler that fetches and saves HTML content from a documentation website.
"""

# cSpell:words manim

from typing import List, Literal, overload
from bs4 import BeautifulSoup
from time import sleep
import requests
import urllib.parse
import re
import os
import logging


# Configure logging
# Define ANSI color codes
COLORS = {
    "INFO": "\033[92m",  # Green
    "WARNING": "\033[93m",  # Yellow
    "ERROR": "\033[91m",  # Red
    "CRITICAL": "\033[91m\033[1m",  # Bold Red
    "DEBUG": "\033[94m",  # Blue
    "RESET": "\033[0m",  # Reset
}


# Custom formatter with colors
class ColoredFormatter(logging.Formatter):
    def format(self, record):
        log_color = COLORS.get(record.levelname, COLORS["RESET"])
        msg = super().format(record)
        return f"{log_color}{msg}{COLORS['RESET']}"


# Configure logging
handler = logging.StreamHandler()
handler.setFormatter(
    ColoredFormatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False


class Content:
    def __init__(self, title: str, url, content: str) -> None:
        self.title = title
        self.url = url
        self.body = content

    def save(self, file_path: str):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if os.path.isdir(file_path):
            safe_title = re.sub(r'[\\/*?:<>|\'"]', "_", self.title)
            file_path = os.path.join(file_path, f"{safe_title}.html")
        logger.info(f"Saving content to {file_path}")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(self.body)


class Website:
    def __init__(
        self, name: str, url: str, /, body_tag: str, title_tag: str = "h1"
    ) -> None:
        self.name = name
        self.url = url
        self.title_tag = title_tag
        self.body_tag = body_tag
        self.link_pattern = r"^(?!https?://|#).*\.html(#.*)?$"


class Crawler:

    @overload
    def __init__(self, site: Website) -> None: ...
    @overload
    def __init__(
        self, name: str, url: str, /, body_tag: str, title_tag: str = "h1"
    ) -> None: ...

    def __init__(self, *args, **kwargs) -> None:
        self.visited = []
        if len(args) == 1 and isinstance(args[0], Website):
            self.site = args[0]

        elif len(args) == 2 and isinstance(args[0], str) and isinstance(args[1], str):
            self.site = Website(*args, **kwargs)
        else:
            raise TypeError(
                "Invalid arguments. Expected a Website object or name, url, body_tag, and title_tag."
            )
        if not os.path.exists(self.site.name):
            os.makedirs(self.site.name)
        self.to_visit: List[str] = [""]

    def get_page(self, local_url: str):
        for i in range(3):
            abs_url = urllib.parse.urljoin(self.site.url, local_url)
            try:
                req = requests.get(abs_url, timeout=20)
                return BeautifulSoup(req.text, "lxml")
            except requests.exceptions.RequestException:
                sleep(i + 2)
        logger.warning(f"Failed to fetch {local_url}, skipping...")
        return None

    def safe_get_tag(
        self,
        page_obj: BeautifulSoup,
        selector: str,
        content: Literal["text", "html"] = "text",
    ):
        """
        Get corresponding tag from `page_obj`, specifically used for title and body
        """
        selected_objs = page_obj.select(selector)
        if content == "text":
            if selected_objs is not None and len(selected_objs) > 0:
                return "\n".join([elem.get_text() for elem in selected_objs])
            return ""
        if content == "html":
            if selected_objs is not None and len(selected_objs) > 0:
                return "\n".join([str(elem) for elem in selected_objs])
            return ""

    def save(self, bs: BeautifulSoup, local_url: str):
        content = None
        if bs is not None:
            title: str = self.safe_get_tag(bs, self.site.title_tag)
            body: str = self.safe_get_tag(bs, self.site.body_tag, "html")
            if title != "" and body != "":
                content = Content(title, local_url, body)
        if not content:
            logger.warning(f"Failed to fetch content from {local_url}. Skipping...")
            return None
        content.save(self.site.name + "/" + local_url)

    def scrape(self, local_url: str):
        abs_url = urllib.parse.urljoin(self.site.url, local_url)
        bs = self.get_page(abs_url)
        if not bs:
            logger.warning(f"Failed to fetch the page:{abs_url}")
            return None
        self.save(bs, local_url)
        logger.info(f"Successfully crawled and saved: {local_url}")

        target_pages = bs.find_all("a", href=re.compile(self.site.link_pattern))
        for target_page in target_pages:
            try:
                target_url: str = target_page.attrs["href"]  # type: ignore
            except KeyError:
                logger.warning("No href attribute found in anchor tag. Skipping...")
                continue
            # target_link format handling
            if target_url.startswith(".."):
                target_url = target_url[3:]
            if target_url.startswith("."):
                target_url = target_url[1:]

            # Append legitimate URL to list
            if target_url not in self.visited and target_url not in self.to_visit:
                self.to_visit.append(target_url)

    def crawl(self):
        """
        Start the crawling process.
        """
        logger.info("Crawling started...")
        while self.to_visit:
            self.scrape(self.to_visit.pop(0))

        logger.info(
            f"Crawling completed. A total of {len(self.visited)} pages were crawled."
        )


if __name__ == "__main__":
    crawler = Crawler(
        "manim_doc",
        "https://docs.manim.community/en/stable/",
        body_tag="#furo-main-content",
        title_tag="h1",
    )
    crawler.crawl()
# TODO: Resolve issues with relative links like `../` and `./`. and `...#...`
