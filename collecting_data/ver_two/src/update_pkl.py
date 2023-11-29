"""load pkl file, update to milvus + database"""
import os
import pickle
import sys
from argparse import ArgumentParser
import requests
from tqdm import tqdm
from milvus import Milvus, IndexType, MetricType
from milvus_utils import MyDatabase

def update_pkl(args):
    """the main function"""
    # print(args)
    # return True
    CENTRAL_STRING = f"http://{args['central_host']}:{args['central_port']}"
    MILVUS_STRING = f"tcp://{args['milvus_host']}:{args['milvus_port']}"
    COLLECTION_NAME = args['milvus_collection']
    TABLE_NAME = COLLECTION_NAME.lower()

    DB = MyDatabase(table_name=TABLE_NAME,
                    user=args['db_user'], pwd=args['db_pass'], host=args["db_host"], port=args["db_port"],
                    dbname=args["db_name"])

    dataset = pickle.load(open(args['pkl_path'], "rb"))

    print(dataset[0].keys())
    assert isinstance(dataset, list)
    assert isinstance(dataset[0], dict)
    assert "mail_code" in dataset[0].keys()
    assert "emb" in dataset[0].keys()
    assert "full_name" in dataset[0].keys()

    # create lists
    list_emb = []
    list_mail_code_full_name = []
    for record in tqdm(dataset, desc="preprocess data"):
        list_mail_code_full_name.append((record['mail_code'], record['full_name']))
        list_emb.append(list(record["emb"]))

    # find all distince mail_code in dataset. If a mail_code is not
    # presented in database, add it.
    staffs = list(set(list_mail_code_full_name))
    for staff in staffs:
        DB.add_staff_or_change_fullname(staff[0], staff[1])


    ###############################
    ####### notify changes ########
    ###############################

    # check if milvus is reachable
    check_string = MILVUS_STRING.replace('tcp', 'http')
    check_string = check_string.replace('19530', '19121')
    if not requests.get(check_string).ok:
        print("error connect to milvus server")
        raise SystemExit(0)

    response = requests.get(CENTRAL_STRING + "/api/milvus/startupdate")
    if response.status_code != 200:
        print("error connect to central server")
        raise SystemExit(0)

    ###############################
    ####### add to database #######
    ###############################
    DB.remove_table_data()

    milvus_index = list(range(len(list_emb)))
    staff_ids = []
    for staff in list_mail_code_full_name:
        id = DB.email_to_id(staff[0])
        staff_ids.append(id)

    DB.add_milvus_data(milvus_index, staff_ids)


    ###############################
    ####### add to milvus #########
    ###############################
    client = Milvus(uri=MILVUS_STRING)
    status, ok = client.has_collection(COLLECTION_NAME)
    if ok:
        client.drop_collection(COLLECTION_NAME)

    status, ok = client.has_collection(COLLECTION_NAME)
    if ok:
        sys.exit(-1)
    else:
        param = {"collection_name": COLLECTION_NAME,
                "dimension": 512,
                "metric_type": MetricType.IP}

        client.create_collection(param)

    status, ok = client.has_collection(COLLECTION_NAME)
    if not ok:
        sys.exit(-1)

    status, ids = client.insert(
        collection_name=COLLECTION_NAME, records=list_emb, ids=milvus_index)
    if not status.OK():
        print("Insert failed: {}".format(status))
        sys.exit(-1)

    client.flush([COLLECTION_NAME])
    # Get demo_collection row count
    status, result = client.count_entities(COLLECTION_NAME)
    print("added", result)

    ivf_param = {"nlist": 1024}
    status = client.create_index(COLLECTION_NAME, IndexType.IVF_FLAT, ivf_param)


    ###############################
    ### notify changes done #######
    ###############################
    response = requests.get(CENTRAL_STRING + "/api/milvus/finishupdate")
    if response.status_code != 200:
        print("error connect to central server")
        raise SystemExit(0)

    ###############################
    ######## sanity check #########
    ###############################


    def test_search():
        search_param = {"nprobe": 16}
        status, result = client.search(
            collection_name=COLLECTION_NAME,
            top_k=1,
            params=search_param,
            query_records=[list_emb[0]],
        )
        print(status, result)


    test_search()


    def consistency_check():
        """return True if milvus and database is identical"""
        import collections

        def is_list_equal(list1, list2):
            """compare elements of two list, ignoring the order"""
            if collections.Counter(list1) != collections.Counter(list2):
                return False
            return True

        def get_milvus_index():
            """get list of milvus indexes"""
            status, result = client.get_collection_stats(COLLECTION_NAME)
            segment_name = result["partitions"][0]["segments"][0]["name"]
            status, milvus_milvus_indexes = client.list_id_in_segment(
                COLLECTION_NAME, segment_name
            )
            assert status.OK()
            return milvus_milvus_indexes

        status, result = client.count_entities(COLLECTION_NAME)
        db_count = DB.get_milvus_count_in_db()
        print(
            "consistent check: milvus count in db {}, in milvus {}".format(
                db_count, result)
        )
        if db_count != result:
            return False
        status, result = client.get_collection_stats(COLLECTION_NAME)
        if len(result["partitions"][0]["segments"]) != 1:
            return False
        milvus_milvus_indexes = get_milvus_index()
        db_milvus_indexes = DB.get_milvus_index_in_db()
        if not is_list_equal(milvus_milvus_indexes, db_milvus_indexes):
            return False
        return True


    return consistency_check()

if __name__ == "__main__":
    parser = ArgumentParser(usage="update pkl file to milvus and database")
    parser.add_argument("--pkl_path", type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'pkl', 'base.pkl'))
    parser.add_argument("--milvus_host", type=str, default="127.0.0.1")
    parser.add_argument("--milvus_port", type=str, default="19530")
    parser.add_argument("--milvus_collection", type=str, default="milvus")
    parser.add_argument("--central_host", type=str,
                        default="localhost", help="central host without http://")
    parser.add_argument("--central_port", type=str, default="8080")
    parser.add_argument("--db_user", type=str, default="postgres", help="database")
    parser.add_argument("--db_pass", type=str, default="postgres", help="database")
    parser.add_argument("--db_host", type=str,
                        default="localhost", help="database")
    parser.add_argument("--db_port", type=str, default="5432")
    parser.add_argument("--db_name", type=str, default="faceiddb", help="db name")
    args = vars(parser.parse_args())
    print(update_pkl(args))