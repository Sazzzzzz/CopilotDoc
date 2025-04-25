# cSpell:words manim

from bs4 import BeautifulSoup
import requests
from pprint import pprint
import urllib.parse
import re
from time import sleep


class Content:
    def __init__(self, title: str, url, content: str) -> None:
        self.title = title
        self.url = url
        self.content = content

    def save(self, file_path):
        file_name = file_path + self.title.replace(" ", "_") + ".html"
        with open(file_name, "w", encoding="utf-8") as file:
            file.write(self.content)


class Website:
    def __init__(
        self, name: str, url: str, /, body_selector: str, title_tag: str = "h1"
    ) -> None:
        self.name = name
        self.url = url
        self.title_tag = title_tag
        self.body_selector = body_selector


class Crawler:
    def __init__(self, site: Website) -> None:
        self.site = site
        self.visited = []

    def get_page(self, url):
        for _ in range(3):
            try:
                req = requests.get(url)
                return BeautifulSoup(req.text, "lxml")
            except requests.exceptions.RequestException:
                sleep(1)
        return None

    def safe_get_tag(self, page_obj: BeautifulSoup, selector: str):
        """
        Get corresponding tag from `page_obj`, specifically used for title and body
        """
        selected_objs = page_obj.select(selector)
        if selected_objs is not None and len(selected_objs) > 0:
            return "\n".join([elem.get_text() for elem in selected_objs])
        return ""

    def parse(self, url):
        bs = self.get_page(url)
        if bs is not None:
            title = self.safe_get_tag(bs, self.site.title_tag)
            body = self.safe_get_tag(bs, self.site.bodyTag)
            if title != "" and body != "":
                content = Content(url, title, body)
                content.print()

    def crawl(self):
        """
        获取网站主页的页面链接
        """
        bs = self.get_page(self.site.url)
        targetPages = bs.findAll("a", href=re.compile(self.site.targetPattern))
        for targetPage in targetPages:
            targetPage = targetPage.attrs["href"]
            if targetPage not in self.visited:
                self.visited.append(targetPage)
                if not self.site.absoluteUrl:
                    targetPage = "{}{}".format(self.site.url, targetPage)
                self.parse(targetPage)


manim_doc = Website(
    "manim_doc",
    "https://docs.manim.community/en/stable/",
    body_selector="#furo-main-content",
    title_tag="h1",
)
