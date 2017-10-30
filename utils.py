import re
def preprocess(text):
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
    
    return text
