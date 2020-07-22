import torch


class CTCLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character,subword):
        dict_character = list(character)
        dict_subword = subword

        self.dict = {}
        self.maxLenOfSubword=2
        for i in dict_subword:
            self.maxLenOfSubword=max(len(i),self.maxLenOfSubword)

        for i, char in enumerate(dict_character + dict_subword):
            # NOTE: 0 is reserved for 'blank' token required by CTCLoss
            self.dict[char] = i + 1

        self.character = ['[blank]'] + dict_character + dict_subword  
        self.subword = dict_subword


    def mylen(self,s):
        if self.subword==[]:
            return len(s)
        count = 0
        i = 0
        while (i < len(s)):
            flag = 0
            for j in range(self.maxLenOfSubword, 1, -1):
                if s[i:i + j] in self.subword:
                    i = i + j
                    count += 1
                    flag = 1
                    break
            if (flag == 0):
                i += 1
                count += 1
        assert count <= len(s) and count > 0
        return count

    def myIndex(self,text):
        New=[]
        for s in text:
            i=0
            while(i<len(s)):
                flag=0
                for j in range(self.maxLenOfSubword,1,-1):
                    if s[i:i + j] in self.subword:
                        New.append(self.dict[s[i:i + j]])
                        i = i + j
                        flag=1
                        break
                if(flag==0):
                    New.append(self.dict[s[i]])
                    i += 1
        return New


    def encode(self, text):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]

        output:
            text: concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length: length of each text. [batch_size]
        """
        length = [self.mylen(s) for s in text]
        text_all=self.myIndex(text)
        sum = 0
        for a in length:
            sum = sum + a
        if sum != len(text_all):
            print(sum,len(text_all))
            assert sum == len(text_all)

        return (torch.IntTensor(text_all), torch.IntTensor(length))


    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        index = 0
        for l in length:
            t = text_index[index:index + l]

            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
            index += l
        return texts


class AttnLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character,subword,batch_max_length):
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        list_token = ['[GO]', '[s]']  
        list_character = list(character)
        list_subword = subword
        self.character = list_token + list_character+list_subword
        self.subword = list_subword
        self.batch_max_length = batch_max_length
        self.dict = {}
        for i, char in enumerate(self.character):
            # print(i, char)
            self.dict[char] = i
        self.maxLenOfSubword = 2
        for i in list_subword:
            self.maxLenOfSubword = max(len(i), self.maxLenOfSubword)

    def mylen(self,s):
        # for i in self.subword:
        #     s=s.replace(i,'$')
        # return len(s)
        count = 0
        i = 0
        while (i < len(s)):
            flag = 0
            for j in range(self.maxLenOfSubword, 1, -1):
                if s[i:i + j] in self.subword:
                    i = i + j
                    count += 1
                    flag = 1
                    break
            if (flag == 0):
                i += 1
                count += 1
        assert count <= len(s) and count > 0

        return count


    def myIndex(self,s):
        New=[]
        i=0
        while (i < len(s)):
            flag = 0
            for j in range(self.maxLenOfSubword, 1, -1):
                if s[i:i + j] in self.subword:
                    New.append(self.dict[s[i:i + j]])
                    i = i + j
                    flag = 1
                    break
            if (flag == 0):
                New.append(self.dict[s[i]])
                i += 1
        return New


    def encode(self, text):
        """ convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """
        length = [self.mylen(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        batch_max_length=self.batch_max_length
        #print('---------------------',batch_max_length)
        batch_max_length += 1
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = torch.cuda.LongTensor(len(text), batch_max_length + 1).fill_(0)
        for i, t in enumerate(text):
            #print("LOAD TEXT", t)
            text_one = list(t)
            text_one.append('[s]')
            #text_one = [self.dict[char] for char in text_one]
            text_one = self.myIndex(text_one)
            batch_text[i][1:1 + len(text_one)] = torch.cuda.LongTensor(text_one)  
        return (batch_text, torch.cuda.IntTensor(length))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts


class Averager(object):
    """Compute average for torch.Tensor, used for loss average."""

    def __init__(self):
        self.reset()

    def add(self, v):
        count = v.data.numel()
        v = v.data.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res
