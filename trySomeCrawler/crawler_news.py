__author__ = 'liushuman'

import urllib2
import sys
import re
import time


def get_content(url):
    response = urllib2.urlopen(url)
    html = response.read()
    idx1 = html.index("box_con")
    idx2 = html.index("box_pic", idx1)
    return html[idx1+10: idx2-14]


def com_filter(html_text):
    strinfo = re.compile(r"<([^>]*)>")
    filter_result = strinfo.sub('', html_text)
    return filter_result.replace('\n\n', '\n')


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print "usage -urlFile -corpusFile"
        sys.exit(1)

    f_urls = open(sys.argv[1], 'r')
    f_corpus = open(sys.argv[2], 'w')

    cnt = 0
    for url in f_urls:
        if url.startswith("###"):
            break

        cnt += 1
        if cnt % 50 == 0:
            print "crawler cnt: ", cnt, time.ctime()
        content = get_content(url)
        filter_content = com_filter(content)
        filter_content_utf8 = filter_content.decode('gbk').encode('utf-8')

        f_corpus.write(filter_content_utf8)
        #print filter_content_utf8

    f_urls.close()
    f_corpus.close()