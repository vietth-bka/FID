from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
import sqlalchemy

class MyDatabase:
    def __init__(self, table_name="milvus", user="postgres", pwd="postgres", host="localhost", port=5432, dbname="faceiddb") -> None:
        POSTGRES_STRING = f"postgres://{user}:{pwd}@{host}:{port}/{dbname}?application_name=laymau"
        self.table_name = table_name
        self.engine = sqlalchemy.create_engine(POSTGRES_STRING)
        self.session = Session(bind=self.engine)
        
        self.Base = declarative_base(bind=self.engine)
        self.STAFF = type('STAFF', (self.Base,), {
                          '__tablename__': 'staff', '__table_args__': {'autoload': True}})
        self.create_table()
        self.MV = type('MV', (self.Base,), {
                        '__tablename__': table_name, '__table_args__': {'autoload': True}})

    def email_to_id(self, email: str) -> int:
        """convert from email to id"""
        _staff = self.session.query(self.STAFF).filter_by(mail_code=email).first()
        if _staff is None:
            return None
        return _staff.id

    def add_staff_or_change_fullname(self, email: str, full_name: str) -> int:
        """look for email in database. 
        If such email is not exists, add to the database
        if such email exists and full_name is changes, change it
        """
        _staff = self.session.query(self.STAFF).filter_by(mail_code=email).first()
        if _staff is None:
            """add"""
            cur_max = self.session.query(sqlalchemy.func.max(self.STAFF.id)).first()[0]
            if cur_max is None:
                cur_max = 0

            new_staff = self.STAFF(
                id=cur_max + 1,
                staff_code=str(cur_max + 1),
                full_name=full_name,
                mail_code=email,
                cellphone='0123456789',
                unit='a',
                department='a',
                date_of_birth='1920-01-01',
                sex='male',
                title='a',
                # note=str(x['Ghi chú']) + ' ' + str(x['Data']) + ' ' + str(x['Details']),
                should_diemdanh=True,
                activate=True)
            self.session.add(new_staff)
            self.session.commit()
            print('added {} {}'.format(email, full_name))
        else:
            if _staff.full_name != full_name:
                _staff.full_name = full_name
                self.session.commit()
                print('updated {} {}'.format(email, full_name))
            else:
                print("do nothing")

    def create_table(self) -> None:
        """create table_name"""
        with self.engine.connect() as con:
            rs = con.execute(
            """
            CREATE TABLE IF NOT EXISTS {} (
                milvus_index bigint NOT NULL PRIMARY KEY,
                staff_id integer,
                key varchar,
                CONSTRAINT milvus_staff_id_fkey FOREIGN KEY (staff_id)
                    REFERENCES public.staff (id) MATCH SIMPLE
                    ON UPDATE NO ACTION
                    ON DELETE NO ACTION
                    NOT VALID
            );
        """.format(
                self.table_name
            )
        )

    def remove_table_data(self) -> None:
        """truncate table data"""

        with self.engine.connect() as con:
            # rs = con.execute('DROP TABLE IF EXISTS {}'.format(table_name))
            rs = con.execute(
                "SELECT EXISTS (SELECT * FROM pg_tables where tablename = '{}')".format(
                    self.table_name
                )
            )
            for row in rs:
                if row[0] == True:
                    print(
                        "In database, table {} exists. Delete its content now".format(
                            self.table_name
                        )
                    )
                    rs = con.execute("DELETE FROM {}".format(self.table_name))
                    # rs = con.execute('VACUUM {}'.format(table_name))
                    print("Table {} is truncated".format(self.table_name))

    def add_milvus_data(self, milvus_index, staff_ids):
        """add milvus data to table_name
        milvus_index: 0 1 2 3 4 5
        staff_ids: 1 1 1 2 2 2 3
        """
        object_to_add = []
        for mv_index, staff_id in zip(milvus_index, staff_ids):
            temp_record = self.MV(milvus_index=mv_index, staff_id=staff_id, key='')
            object_to_add.append(temp_record)
        self.session.add_all(object_to_add)
        self.session.commit()

    def get_milvus_index_in_db(self):
        """get list of index stored in database"""
        db_milvus_indexes = self.session.query(self.MV.milvus_index).all()
        db_milvus_indexes = [a[0] for a in db_milvus_indexes]
        return db_milvus_indexes

    def get_milvus_count_in_db(self):
        """count the number of rows in table_name"""
        db_count = self.session.query(self.MV).count()
        return db_count