# Imports from external libraries
from bs4 import BeautifulSoup
import requests
import re
import codecs
from selenium import webdriver

# Imports from internal libraries


def google_the_claim(claim: str, pages_to_check: int = 1, verbose: bool = False, limit_to_wiki=False, starting_from_page=0, search_engine="google") -> list:
    assert search_engine in ["google", "yahoo", "bing"]

    if search_engine == "google":
        search_page = "https://www.google.com/search?q="
        start_symbol = "start"
    elif search_engine == "yahoo":
        search_page = "https://search.yahoo.com/search?p="
        start_symbol = "b"
    elif search_engine == "bing":
        search_page = "https://www.bing.com/search?q="
        start_symbol = "first"

    wikipedia_titles = list()
    try:
        google_query = "+".join(claim.split(" "))
        if limit_to_wiki:
            site = "en.wikipedia.org"
            google_query += f"+site%3A{site}"
        for i in range(pages_to_check):
            start = i * 10 + starting_from_page * 10
            if start > 0 and search_engine in ["yahoo", "bing"]:
                start += 1

            page_url = f"{search_page}{google_query}&{start_symbol}={str(start)}"

            if verbose:
                print(page_url)

            page = requests.get(page_url)
            page.raise_for_status()
            soup = BeautifulSoup(page.content, "html.parser")


            if search_engine == "google":
                links = [div.find("h3").a.get("href") for div in soup.find_all("div", class_="g") if div.find("h3")]
            elif search_engine == "yahoo":
                links = [a.a.get('href') for a in soup.find_all("h3", class_="title") if a.a]
            elif search_engine == "bing":
                links = [s.h2.a.get('href') for s in soup.find_all("li", class_="b_algo")]

            for link in links:
                match = re.search("https://[^&]+", link)
                if match is not None:
                    url = match.group()
                    if verbose:
                        print(f"Found url: {url}")
                    if re.search("[en|hr].wikipedia", url):
                        title = url.split("/")[-1]
                        while True:
                            match = re.search("(%25|%)[a-fA-f0-9]{2}((%25|%)[a-fA-F0-9]{2}){0,2}", title)
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
                                    wikipedia_titles.append(title)
                                    break
                            else:
                                corrupted = match.group()
                                if re.search("%25[a-fA-f0-9]{2}(%25[a-fA-F0-9]{2}){0,2}", corrupted):
                                    hex_charachter = corrupted.replace("%25", "").lower()
                                else:
                                    hex_charachter = corrupted.replace("%", "").lower()
                                utf_decoded = codecs.decode(hex_charachter, "hex").decode('utf-8')
                                title = title.replace(corrupted, utf_decoded)
        return wikipedia_titles
    except Exception as e:
        # print(e)
        # return wikipedia_titles
        raise e


def google_the_claim_text_results(claim: str, concatenation=" ", browser=None):
    google_query = "+".join(claim.split(" "))
    page_url = f"https://www.google.com/search?q={google_query}"

    if browser is None:
        browser = webdriver.Chrome(executable_path='./chromedriver')

    browser.get(page_url)
    soup = BeautifulSoup(browser.page_source, 'html.parser')

    rcs = soup.find_all("div", class_="rc")

    titles = [rc.find("div", class_="r").find("h3", class_="LC20lb").text.replace(u'\xa0', u' ') for rc in rcs]
    titles_no_page = [t.split(" - ", 1)[0] for t in titles]
    titles_no_page_no_dots = [t[:-4] if t.endswith("...") else t for t in titles_no_page]

    summaries = [rc.find("div", class_="s").find("span", class_="st").text.replace(u'\xa0', u' ') if rc.find("div", class_="s") and rc.find("div", class_="s").find("span", class_="st") else "" for rc in rcs]
    summaries_no_date = [s.split(" - ", 1)[-1] for s in summaries]
    summaries_no_date_no_dots = [s.replace(" ... ", " ").replace(" ...", "") for s in summaries_no_date]

    result = [f"{t}{concatenation}{s}" for t, s in zip(titles_no_page_no_dots, summaries_no_date_no_dots)]
    final_result = f"{concatenation}".join(result)

    return final_result


if __name__ == '__main__':
    # print(google_the_claim("Nelson Mandela was Chinese.", starting_from_page=2, verbose=True))
    # google_the_claim("Arnold acted in Conan", limit_to_wiki=True, search_engine="yahoo", verbose=True)
    google_the_claim(
        "Luis Armstrong won the superbowl.",
        search_engine="bing",
        limit_to_wiki=True,
        starting_from_page=0,
        verbose=True
    )
