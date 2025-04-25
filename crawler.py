# cSpell:words manim

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

    def save(self, file_path):
        file_name = file_path + self.title.replace(" ", "_") + ".html"
        # Ensure the directory exists before saving
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, "w", encoding="utf-8") as file:
            file.write(self.body)


class Website:
    def __init__(
        self, name: str, url: str, /, body_tag: str, title_tag: str = "h1"
    ) -> None:
        self.name = name
        self.url = url
        self.title_tag = title_tag
        self.body_tag = body_tag
        self.target_pattern = r"^(?!https?://).*\.html$"


class Crawler:
    def __init__(self, site: Website) -> None:
        self.site = site
        self.visited = []
        if not os.path.exists(site.name):
            os.makedirs(site.name)

    def get_page(self, local_url):
        for i in range(3):
            abs_url = urllib.parse.urljoin(self.site.url, local_url)
            try:
                req = requests.get(abs_url, timeout=20)
                return BeautifulSoup(req.text, "lxml")
            except requests.exceptions.RequestException:
                sleep(i + 2)
        logger.warning(f"Failed to fetch {local_url}, skipping...")
        return None

    def safe_get_tag(self, page_obj: BeautifulSoup, selector: str):
        """
        Get corresponding tag from `page_obj`, specifically used for title and body
        """
        selected_objs = page_obj.select(selector)
        if selected_objs is not None and len(selected_objs) > 0:
            return "\n".join([elem.get_text() for elem in selected_objs])
        return ""

    def save(self, bs: BeautifulSoup, local_url: str):
        content = None
        if bs is not None:
            title = self.safe_get_tag(bs, self.site.title_tag)
            body = bs.prettify()
            if title != "" and body != "":
                content = Content(title, local_url, body)
        if not content:
            logger.warning(f"Failed to fetch content from {local_url}. Skipping...")
            return None
        content.save(self.site.name + "/" + local_url)

    def crawl(self):
        main_bs = self.get_page(self.site.url)
        if not main_bs:
            logger.error("Failed to fetch the main page.")
            raise Exception("Failed to fetch the main page.")
        self.save(main_bs, "")
        logger.info(f"Successfully crawled and saved: {self.site.url}")

        target_pages = main_bs.find_all("a", href=re.compile(self.site.target_pattern))
        for target_page in target_pages:
            try:
                target_url: str = target_page.attrs["href"]  # type: ignore
            except KeyError:
                logger.warning("No href attribute found in anchor tag. Skipping...")
                continue
            if target_url not in self.visited:
                self.visited.append(target_url)
                abs_url = urllib.parse.urljoin(self.site.url, target_url)

                if bs := self.get_page(abs_url):
                    self.save(bs, target_url)
                logger.info(f"Successfully crawled and saved: {abs_url}")


manim_doc = Website(
    "manim_doc",
    "https://docs.manim.community/en/stable/",
    body_tag="#furo-main-content",
    title_tag="h1",
)

crawler = Crawler(manim_doc)
crawler.crawl()
logger.info("Crawling completed.")
