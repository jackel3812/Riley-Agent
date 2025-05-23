"""
RILEY - Text Preprocessing
This module handles preprocessing of text data for RILEY's language models.
"""

from html import unescape
from html.parser import HTMLParser


def get_html_paragraphs(src: str):
    """
    Return plain text paragraphs from an HTML source

    :param src: HTML document to convert to plain text paragraphs
    :return: Plain text paragraphs of document

    This function is designed to be quick rather than robust.

    It follows a simple approach to extracting text:

    1. Ignore all content inside the following elements listed in `ignore`.
    2. Merge inline text content into paragraphs from `inlines` set.
    3. Convert any newly merged text element with at least `min_length`
    characters to a paragraph in the output text.
    """

    class ParagraphExtractor(HTMLParser):
        paras = [""]
        ignoring = []
        ignore = ("script", "style", "header", "footer")
        ignore_attrs = {('hidden', 'hidden'), }
        inlines = ("a", "b", "i", "span", "sup", "sub", "strong", "em")

        def handle_starttag(self, tag, attrs):
            if tag in self.ignore or self.ignore_attrs & set(attrs):
                self.ignoring.append(tag)

            if tag not in self.inlines and self.paras[-1]:
                self.paras.append("")

        def handle_endtag(self, tag):
            if self.ignoring and self.ignoring[-1] == tag:
                self.ignoring.pop()

            if tag not in self.inlines and self.paras[-1]:
                self.paras.append("")

        def handle_data(self, data):
            if not self.ignoring:
                if self.paras and self.paras[-1]:
                    self.paras[-1] += unescape(data)
                else:
                    self.paras.append(data)

        def get_plain(self):
            return "\n\n".join([p.rstrip() for p in self.paras if len(p.strip()) > 140])

    extractor = ParagraphExtractor()
    extractor.feed(src)
    return extractor.get_plain()