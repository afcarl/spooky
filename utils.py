def preprocess(text):
    text = text.replace("' ", " ' ")
    signs = set(',.:;"?')
    prods = set(text) & signs
    if not prods:
        return text

    for sign in prods:
        text = text.replace(sign, ' {} '.format(sign) )
    return text