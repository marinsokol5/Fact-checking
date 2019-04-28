# Imports from external libraries
from bs4 import BeautifulSoup
import requests
import re
import codecs

# Imports from internal libraries


def google_the_claim(claim: str, pages_to_check: int = 1, verbose: bool = False) -> set:
    wikipedia_titles = set()
    try:
        google_query = "+".join(claim.split(" "))
        for i in range(pages_to_check):
            start = i * 10

            page = requests.get(f"https://www.google.com/search", params={"q": google_query, "start": start})
            soup = BeautifulSoup(page.content, "html.parser")

            for div in soup.find_all("div", class_="g"):
                if div.find("h3") is None:
                    continue
                link = div.find("h3").a.get("href")
                match = re.search("https://[^&]+", link)
                if match is not None:
                    url = match.group()
                    if verbose:
                        print(f"Found url: {url}")
                    if re.search("[en|hr].wikipedia", url):
                        title = url.split("/")[-1]
                        while True:
                            match = re.search("(%25|%)[a-fA-f0-9]{2}(%25|%)[a-fA-F0-9]{2}", title)
                            if match is None:
                                if re.search("hr.wikipedia", url):
                                    url_0 = "/".join(url.split("/")[:-1])
                                    url_tmp = f"{url_0}/{title}"
                                    page_tmp = requests.get(url_tmp)
                                    soup_tmp = BeautifulSoup(page_tmp.content, "html.parser")

                                    engl_link = \
                                        [a.get("href") for a in
                                         soup_tmp.find_all("a", class_="interlanguage-link-target") if
                                         a.get("lang") == "en"][0]
                                    match = re.search("https://[^&]+", engl_link)

                                    if verbose:
                                        print(f"Switching from {url} to {match.group()}")

                                    url = match.group()
                                    title = url.split("/")[-1]
                                else:
                                    wikipedia_titles.add(title)
                                    break
                            else:
                                corrupted = match.group()
                                if re.search("%25[a-fA-f0-9]{2}%25[a-fA-F0-9]{2}", corrupted):
                                    hex_charachter = corrupted.replace("%25", "").lower()
                                else:
                                    hex_charachter = corrupted.replace("%", "").lower()
                                utf_decoded = codecs.decode(hex_charachter, "hex").decode('utf-8')
                                title = title.replace(corrupted, utf_decoded)
        return wikipedia_titles
    except Exception:
        return wikipedia_titles

