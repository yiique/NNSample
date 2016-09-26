__author__ = 'liushuman'

import urllib
import urllib2
import json
import random
import sys
import time

USER_AGENT_LIST = ["Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_6_8; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50",     # safari mac
                   "Mozilla/5.0 (Windows; U; Windows NT 6.1; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50",              # safari win
                   "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0)",   # IE9
                   "Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0)",   # IE8
                   "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.6; rv:2.0.1) Gecko/20100101 Firefox/4.0.1",   # firefox mac
                   "Mozilla/5.0 (Windows NT 6.1; rv:2.0.1) Gecko/20100101 Firefox/4.0.1",                   # firefox win
                   "Opera/9.80 (Macintosh; Intel Mac OS X 10.6.8; U; en) Presto/2.8.131 Version/11.11",     #opera mac
                   "Opera/9.80 (Windows NT 6.1; U; en) Presto/2.8.131 Version/11.11",                       #opera win
                   "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_0) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11",    # chrome mac
                   "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; TencentTraveler 4.0)",   # tecent
                   "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Trident/4.0; SE 2.X MetaSr 1.0; SE 2.X MetaSr 1.0; .NET CLR 2.0.50727; SE 2.X MetaSr 1.0)",  # sougou
                   "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; 360SE)"]     # 360

def request_ajax_data(url, data, referer=None, **headers):
    req = urllib2.Request(url)
    req.add_header('Content-Type', 'application/x-www-form-urlencoded; charset=UTF-8')
    req.add_header('X-Requested-With', 'XMLHttpRequest')
    req.add_header('User-Agent',
                   USER_AGENT_LIST[random.randint(0, 11)])
    if referer:
        req.add_header('Referer',referer)
    if headers:
        for k in headers.keys():
            req.add_header(k, headers[k])

    params = urllib.urlencode(data)
    response = urllib2.urlopen(req, params)
    json_text = response.read()
    return json.loads(json_text)

# sample
# ajaxRequestBody = {"from": 'en', "to": 'zh', "query": 'cellphone', "transtype": 'enter', "simple_means_flag": '3'}
# ajaxResponse = request_ajax_data('http://fanyi.baidu.com/v2transapi', ajaxRequestBody)
# print ajaxResponse


url = 'http://fanyi.baidu.com/v2transapi'
form_dict = {}
dictionaryName = ''
corpusName = ''


LIJU_RESULT = unicode('liju_result', "UTF-8")
DOUBLE = unicode('double', "UTF-8")


def form_generator(f, t):
    cf = open("form_template.txt", 'r')
    for line in cf:
        if "from=" + f in line and "to=" + t in line:
            config_sequence = line[:-1].split(" ")
            print config_sequence
            for config in config_sequence:
                pair = config.split('=')
                form_dict[pair[0]] = pair[1]
    cf.close()


def pairs_generator(src_vector, tgt_vector):
    src_sent = [x[0] for x in src_vector]
    tgt_sent = [x[0] for x in tgt_vector]
    alignment = []

    idx_src = int(src_vector[0][1][2:])
    len_src = len(src_vector)

    for tgt_w in tgt_vector:
        if tgt_w[2] == '':
            continue

        align_info = ""
        align_pairs = tgt_w[2].split(',')
        se = False
        for a in align_pairs:
            align = int(a[2:])
            if align >= idx_src + len_src and se is False:
                align_info += "_" + str(align - idx_src - len_src)
                se = True
            else:
                align_info += "-" + str(align - idx_src)

        if align_info[1:] not in alignment:
            alignment.append(align_info[1:])

    return src_sent, tgt_sent, alignment


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print "usage -srcLanguage -tgtLanguage -dictName -corpusName"
        sys.exit(1)

    form_generator(sys.argv[1], sys.argv[2])
    dictionaryName = sys.argv[3]
    corpusName = sys.argv[4]

    df = open(dictionaryName, 'r')
    dc = open(corpusName, 'w')
    count = 0
    for line in df:
        count += 1
        if count % 100 == 0:
            print "processing...", count, time.ctime()
            time.sleep(random.random()*3)

        form_dict['query'] = line[:-1]
        ajaxResponse = request_ajax_data(url, form_dict)

        double_str = json.loads(ajaxResponse[LIJU_RESULT][DOUBLE])

        for sample_pairs in double_str:
            sample_src = sample_pairs[0]        # src_list
            sample_tgt = sample_pairs[1]

            src_sent, tgt_sent, alignment = pairs_generator(sample_src, sample_tgt)
            # print " ".join(src_sent).encode('utf-8')
            # print " ".join(tgt_sent).encode('utf-8')
            # print " ".join(alignment).encode('utf-8')
            dc.write(" ".join(src_sent).encode('utf-8') + '\n')
            dc.write(" ".join(tgt_sent).encode('utf-8') + '\n')
            dc.write(" ".join(alignment).encode('utf-8') + '\n\n')

    print "Done"
    df.close()
    dc.close()
