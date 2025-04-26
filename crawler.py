"""
A simple web crawler that fetches and saves HTML content from a documentation website.
"""

# cSpell:words manim

from typing import Literal, overload
from bs4 import BeautifulSoup
from dataclasses import dataclass  # Import dataclass
from asyncio import Queue
from pathlib import Path, PurePosixPath
import httpx
import urllib.parse
import re
import logging
import asyncio


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


@dataclass
class Content:
    title: str
    url: str
    body: str

    def save(self, path: Path) -> None:
        if path.suffix == "":
            escape_characters = re.compile(r'[\\/*?:<>|\'"]')
            safe_title = re.sub(escape_characters, "_", self.title)
            path = path / f"{safe_title}.html"
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving content to {path}")
        with path.open("w", encoding="utf-8") as file:
            file.write(self.body)


@dataclass
class Website:
    name: str
    url: str
    body_tag: str
    title_tag: str = "h1"
    link_pattern: str = r"^(?!https?://|#).*\.html(#.*)?$"  # Can be defined directly
    block_ref_pattern: str = r"#.*$"  # Can be defined directly

    @property
    def link_pattern_compiled(self) -> re.Pattern:
        return re.compile(self.link_pattern)

    @property
    def block_ref_pattern_compiled(self) -> re.Pattern:
        return re.compile(self.block_ref_pattern)


class Crawler:

    @overload
    def __init__(self, site: Website) -> None: ...
    @overload
    def __init__(
        self, name: str, url: str, /, body_tag: str, title_tag: str = "h1"
    ) -> None: ...

    def __init__(self, *args, **kwargs) -> None:
        if len(args) == 1 and isinstance(args[0], Website):
            self.site = args[0]

        elif len(args) == 2 and isinstance(args[0], str) and isinstance(args[1], str):
            self.site = Website(*args, **kwargs)
        else:
            raise TypeError(
                "Invalid arguments. Expected a Website object or name, url, body_tag, and title_tag."
            )
        Path(self.site.name).mkdir(parents=True, exist_ok=True)
        self.recorded: set[str] = set()
        self.client = httpx.AsyncClient(follow_redirects=True, timeout=20.0)
        self.to_visit: Queue[str] = Queue()

    async def get_page(self, local_url: str):  # Make async
        abs_url = urllib.parse.urljoin(self.site.url, local_url)
        for i in range(3):
            try:
                response = await self.client.get(abs_url)  # Use await client.get
                response.raise_for_status()  # Check for HTTP errors
                return BeautifulSoup(response.text, "lxml")
            except httpx.RequestError as exc:  # Catch httpx errors
                logger.warning(f"Request failed for {abs_url} (attempt {i+1}): {exc}")
                await asyncio.sleep(i + 2)  # Use asyncio.sleep
            except httpx.HTTPStatusError as exc:
                logger.error(
                    f"HTTP error for {abs_url}: {exc.response.status_code} - {exc}"
                )
                break  # Don't retry on HTTP errors like 404
        logger.warning(
            f"Failed to fetch {local_url} after multiple attempts, skipping..."
        )
        return None

    def safe_get_tag(
        self,
        page_obj: BeautifulSoup,
        selector: str,
        content: Literal["text", "html"] = "text",
    ) -> str:
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
        """
        Save the content to corresponding file path
        """
        content = None
        if bs is not None:
            title: str = self.safe_get_tag(bs, self.site.title_tag)
            body: str = self.safe_get_tag(bs, self.site.body_tag, "html")
            if title != "" and body != "":
                content = Content(title, local_url, body)
        if not content:
            logger.warning(f"Failed to fetch content from {local_url}. Skipping...")
            return None
        content.save(Path(self.site.name + "/" + local_url))

    def _local_url_formatter(self, local_url: str, target_url: str) -> str:
        """
        Format the local URL to a valid path
        """
        if local_url == "":
            return target_url
        parent_url = str(PurePosixPath(local_url).parent)
        while target_url.startswith("../"):
            parent_url = str(PurePosixPath(parent_url).parent)
            target_url = target_url[3:]
        if parent_url == ".":
            return target_url
        if target_url.startswith("./"):
            target_url = target_url[2:]
        elif target_url.startswith("/"):
            target_url = target_url[1:]

        return parent_url + "/" + target_url

    async def scrape(self, local_url: str) -> None:
        abs_url = urllib.parse.urljoin(self.site.url, local_url)
        bs = await self.get_page(abs_url)
        if not bs:
            logger.warning(f"Failed to fetch the page:{abs_url}")
            return None
        self.save(bs, local_url)
        logger.info(f"Successfully crawled and saved: {local_url}")

        target_pages = bs.find_all(
            "a", href=re.compile(self.site.link_pattern_compiled)
        )
        for target_page in target_pages:
            try:
                target_url: str = target_page.attrs["href"]  # type: ignore
            except KeyError:
                logger.warning("No href attribute found in anchor tag. Skipping...")
                continue
            # target_link format handling
            target_url = re.sub(self.site.block_ref_pattern_compiled, "", target_url)
            target_url = self._local_url_formatter(local_url, target_url)
            # Append legitimate URL to list
            if target_url not in self.recorded:
                self.recorded.add(target_url)
                await self.to_visit.put(target_url)

    def crawl(self):  # Public synchronous method
        """
        Start the crawling process. This method is synchronous and
        runs the internal async crawling loop.
        """
        logger.info("Initiating crawling process...")
        try:
            asyncio.run(self._async_crawl())
        except KeyboardInterrupt:
            logger.info("Crawling interrupted by user.")
        except Exception as e:
            logger.critical(
                f"An unexpected error occurred during crawl initiation or execution: {e}",
                exc_info=True,
            )
        finally:
            logger.info(
                f"Crawling process finished. A total of {len(self.recorded)} URLs were recorded."
            )

    async def _async_crawl(self):  # Private async method
        """
        Internal asynchronous crawling loop.
        """
        logger.info(f"Async crawling started for {self.site.url}...")

        self.to_visit.put_nowait("")
        self.recorded.add("")
        while True:
            try:
                # Wait briefly for an item, allows checking if crawling should stop
                url_to_scrape = await asyncio.wait_for(self.to_visit.get(), timeout=1.0)
            except asyncio.TimeoutError:
                # Check if the queue is truly empty after the timeout
                if self.to_visit.empty():
                    logger.info("Queue appears empty after timeout, checking join...")
                    break  # Exit the loop, proceed to join
                else:
                    continue  # Queue not empty, maybe just slow producer, continue waiting
            # Process the URL
            await self.scrape(url_to_scrape)  # Process one URL at a time
            self.to_visit.task_done()  # Mark task as done

        # Wait for all tasks in the queue to be processed
        logger.info("Waiting for any remaining tasks in queue to complete...")
        await self.to_visit.join()  # Wait until task_done called for all items

        logger.info("Closing HTTP client...")
        await self.client.aclose()


if __name__ == "__main__":
    crawler = Crawler(
        "manim_doc",
        "https://docs.manim.community/en/stable/",
        body_tag="#furo-main-content",
        title_tag="h1",
    )
    crawler.crawl()
# TODO: Handling paths
# TODO: What's all these async functions do?
# TODO: async version of `crawl`
