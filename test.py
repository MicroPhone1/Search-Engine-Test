import requests
from bs4 import BeautifulSoup
from collections import defaultdict
from fuzzywuzzy import fuzz
from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# 1. Web Crawling - ดึงข้อมูลจากหลายเว็บไซต์
def crawl_websites(urls):
    titles = []
    content = []

    for url in urls:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        titles.extend([title.text for title in soup.find_all("h1")])
        content.extend([p.text for p in soup.find_all("p")])

    return titles, content


# ใส่ URLs ที่ต้องการรวบรวมข้อมูล
urls = [
    "https://en.wikipedia.org/wiki/Web_scraping",
    "https://www.bbc.com/news",
]

titles, content = crawl_websites(urls)


# 2. Indexing - การสร้าง inverted index
def create_inverted_index(titles, content):
    index = defaultdict(list)

    for idx, title in enumerate(titles):
        for word in title.split():
            index[word.lower()].append(idx)

    for idx, text in enumerate(content):
        for word in text.split():
            index[word.lower()].append(idx)

    return index


inverted_index = create_inverted_index(titles, content)


# 3. Search - การค้นหาหลายคำและ Fuzzy Search
def search(query, inverted_index):
    words = query.split()
    results = [
        set(inverted_index[word.lower()])
        for word in words
        if word.lower() in inverted_index
    ]

    if results:
        return set.intersection(*results)  # ผลลัพธ์ที่มีทุกคำค้นหา
    return set()  # ถ้าไม่มีคำค้นหาใด ๆ


def fuzzy_search(query, inverted_index):
    results = set()
    for word in inverted_index.keys():
        if fuzz.partial_ratio(query.lower(), word) > 70:  # เปรียบเทียบคำใกล้เคียง
            results.update(inverted_index[word])
    return results


# 4. Ranking - จัดอันดับผลลัพธ์ด้วย TF-IDF
def rank_results(results, content, query):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(content)

    query_vector = vectorizer.transform([query])
    cosine_sim = cosine_similarity(query_vector, tfidf_matrix)

    ranked_results = sorted(
        zip(results, cosine_sim[0]), key=lambda x: x[1], reverse=True
    )

    return [content[i] for i, score in ranked_results if score > 0]


# 5. Web Application - Flask app
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/search", methods=["POST"])
def search_results():
    query = request.form["query"]
    results = search(query, inverted_index)

    if not results:  # ถ้าผลลัพธ์ว่าง, ลองใช้ fuzzy search
        results = fuzzy_search(query, inverted_index)

    ranked_results = rank_results(results, content, query)

    return render_template("results.html", results=ranked_results)


if __name__ == "__main__":
    app.run(debug=True)
