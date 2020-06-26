import pandas as pd

enum = enumerate

if __name__ == "__main__":
    wufi = pd.read_csv('wufi.csv')
    winnie = pd.read_csv('gwytb.csv')

    contents = wufi['content']

    del contents[294]
    del contents[100]

    # print(contents[49])
    # exit()

    for i, c in enum(contents):

        if '觀眾朋友大家好，我是許世楷，歡迎收看台灣廣場' in c:
            c_splitted = c.split('。')
            start = c_splitted.index('觀眾朋友大家好，我是許世楷，歡迎收看台灣廣場')
            c = '。'.join(c_splitted[start+1:])

        c = '。'.join(c.split())

        while '◎' in c:
            c_splitted = c.split('。')
            del_idx = -1
            for j in range(len(c_splitted)):
                if '◎' in c_splitted[j]:
                    del_idx = j
                    break
            del c_splitted[del_idx]
            c = '。'.join(c_splitted)

        if len(c.split('。')[0]) == 3:
            c = '。'.join(c.split('。')[1:])

        contents[i] = c


    # for i, c in enum(contents):
    #     print(i, c[:52])


    sentence_list = []

    for c in contents:
        for sentence in c.split('。'):
            if '，' in sentence and len(sentence) > 6:
                sentence_list.append(sentence)


    for i, s in enum(sentence_list):
        if len(s) > 60:
            if '，' in s[60:]:
                pivot = s[60:].find('，')
                a, b = s[60:][:pivot], s[60:][pivot+1:]
                sentence_list[i] = s[:60] + a
                sentence_list.append(b)
        
    
    for s in sentence_list:
        print(s)

    print(len(sentence_list))



    train, test = sentence_list[:int(len(sentence_list)*0.8)], sentence_list[int(len(sentence_list)*0.8): ]
    dev, test = test[:int(len(test)*0.5)], test[int(len(test)*0.5): ]

    with open('train.1', 'w', encoding='utf-8') as f:
        for t in train:
            f.write(t + '\n')

    with open('dev.1', 'w', encoding='utf-8') as f:
        for t in dev:
            f.write(t + '\n')

    with open('test.1', 'w', encoding='utf-8') as f:
        for t in test:
            f.write(t + '\n')