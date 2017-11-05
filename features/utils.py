import re
def preprocess(text, pre_trained_fastText=False):
    text = text.replace('"', ' " ')    
    text = re.sub(r"(')(\s|$)", r" \1 ", text)
    text = re.sub(r"(^|\s)(')", r" \2 ", text)

    for sign in ';:,': #?
        text = re.sub(r'(\s|^)({})'.format(sign), r' \2 ', text)
        text = re.sub(r'({})($|\s)'.format(sign), r' \1 ', text)

    text = re.sub(r'(\.+)(\s|$)', r' \1 ', text)

    text = re.sub(r"(')(\s|$)", r" \1 ", text) # special case: 'hoge'. 
    
    text = re.sub(r"(\?)(\s|$)", r' \1 ', text)
    text = re.sub(r"(^|\s)(\?+)", r' \2 ', text)    
    
    if pre_trained_fastText:
        text = text.replace('\'', ' \' ').replace(';', ' ; ').replace(',', ' , ')

    return text

