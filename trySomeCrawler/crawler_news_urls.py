__author__ = 'liushuman'

import urllib2
import sys
import re


def get_urls_div(url):
    response = urllib2.urlopen(url)
    html = response.read()
    idx1 = html.index("w650")
    idx2 = html.index("</div>", idx1)
    return html[idx1: idx2]

def get_urls_vector(urls_div):
    urls_vector = []
    idx = 0
    while True:
        try:
            idx = urls_div.index("a href='/n", idx+1)
            idx2 = urls_div.index(".html", idx)
            urls_vector.append("http://bj.people.com.cn" + urls_div[idx+8: idx2+5])
        except:
            return urls_vector


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print "usage -candidateFile -urlFile"
        sys.exit(1)

    f_can = open(sys.argv[1], 'r')
    f_urls = open(sys.argv[2], 'w')

    for candidate in f_can:
        urls_div = get_urls_div(candidate)
        urls_vector = get_urls_vector(urls_div)

        for url in urls_vector:
            f_urls.write(url + "\n")

    f_can.close()
    f_urls.close()