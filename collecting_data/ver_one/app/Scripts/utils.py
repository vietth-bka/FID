import hashlib

hasher = hashlib.md5()

def getmd5(path):
    file = open(path, 'rb')
    data = file.read(20)
    while len(data) > 0:
        hasher.update(data)
        data = file.read(20)
    return hasher.hexdigest()