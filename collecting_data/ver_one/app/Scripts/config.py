
class Config():
    pass

class DevelopmentConfig(Config):
    POSTGRES_USER_NAME = 'postgres'
    POSTGRES_USER_PASS = '123456'
    POSTGRES_HOSTNAME = '172.21.100.15'
    POSTGRES_PORT = '5432'
    POSTGRES_DATABASE = 'laymau'

    MINIO_ACCESS_KEY = 'minio'
    MINIO_SECRET_KEY = 'minio123'
    MINIO_HOSTNAME = '172.21.100.15'
    MINIO_PORT = '9001'
    MINIO_BUCKETNAME = 'laymau'

class ProductionConfigure(Config):
    POSTGRES_USER_NAME = 'dat'
    POSTGRES_USER_PASS = '123456'
    POSTGRES_HOSTNAME = '172.21.100.54'
    POSTGRES_PORT = '5432'
    POSTGRES_DATABASE = 'laymau'

    MINIO_ACCESS_KEY = 'ucQbeazlRY2PdwUp'
    MINIO_SECRET_KEY = 'xYdCTj1SJVhbnkmM'
    MINIO_HOSTNAME = '172.21.100.51'
    MINIO_PORT = '9000'
    MINIO_BUCKETNAME = 'laymau'

import os
def getConfig():
    #return DevelopmentConfig()
    # if 'PRODUCTION' in os.environ or 'PRODUCTION' in os.environ:
    #     return ProductionConfigure()
    # else:
    #     return DevelopmentConfig()\
    return ProductionConfigure()
def cal():
    return 0
if __name__ == "__main__":
    con = getConfig()
    print(dir(con))
