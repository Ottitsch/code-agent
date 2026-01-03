import codecs

def rot13(text):
    return codecs.decode(text, 'rot_13')

message = 'Pbatenghyngvbaf ba ohvyqvat n pbqr rqvgvat ntrag!'
decoded_message = rot13(message)
print(decoded_message)