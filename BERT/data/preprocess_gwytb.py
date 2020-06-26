import pandas as pd

enum = enumerate

if __name__ == "__main__":
    winnie = pd.read_csv('gwytb.csv')

    contents = winnie['content']
    contents = [c for c in contents if type(c) == str]
    
    for i, c in enum(contents):

        if c[0] == '國':
            start = c.find('發言人')
            if c[start+3] == '、':
                continue
            speaker = c[start+3:start+6]
            
            filtered = []
            for p in c.split('。'):
                if speaker+'：' in p and '上午好！' not in p and '早上好！' not in p:
                    filtered.append(p[p.find(speaker)+4:])

            if filtered != []:
                contents[i] = '。'.join(filtered)
        else:
            for speaker in ['楊毅', '范麗青']:
                filtered = []
                for p in c.split('。'):
                    if speaker+'：' in p and '上午好！' not in p and '早上好！' not in p:
                        filtered.append(p[p.find(speaker)+len(speaker)+1:])

                if filtered != []:
                    contents[i] = '。'.join(filtered)

        filtered = []
        for p in contents[i].split('。')[3:]:
            if '上午好' not in p and '早上好' not in p:
                filtered.append(p[p.find(speaker)+4:])

        if filtered != []:
            contents[i] = '。'.join(filtered)

    contents = [c for c in contents if len(c) > 5]



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

    with open('train.0', 'w', encoding='utf-8') as f:
        for t in train:
            f.write(t + '\n')

    with open('dev.0', 'w', encoding='utf-8') as f:
        for t in dev:
            f.write(t + '\n')

    with open('test.0', 'w', encoding='utf-8') as f:
        for t in test:
            f.write(t + '\n')