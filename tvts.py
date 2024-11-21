import re
import pymongo as pm
import os
import sys
import datetime
import numpy as np
from typing import Optional, List, Sequence, Union
from collections.abc import Iterable
import requests
import json
import traceback
import copy
import math
from dataclasses import dataclass

from PyCmpltrtok.common import long_text_to_block, md5, has_content
from PyCmpltrtok.auth.mongo.conn import conn

DEFAULT_LINK = 'local'
DEFAULT_HOST = 'localhost'
DEFAULT_PORT = 27017
DEFAULT_DB_NAME = 'tvts'
DEFAULT_TABLE_PREFIX = 'train_log'
DEFAULT_SAVE_FREQ = 1
DEFAULT_LOCATE_SERVICE_PORT = 7654
INDEX_NAME = 'id_epoch'
MUST_PARAMS = set(['epoch'])
INDEX_NAME_4BATCH = 'id_epoch_batch'
INDEX_NAME2_4BATCH = 'id_global_batch'
DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'
RESERVED_KEYS = set([
    'batch',
    'datetime',
    'duration_in_sec',
    'epoch',
    'from_datetime',
    'global_batch',
    'init_weights',
    'is_temp',
    'memo',
    'name',
    'parent_epoch',
    'parent_id',
    'save_dir',
    'save_freq',
    'to_datetime',
    'train_id',
])

LOCAT_SERVICE_PWD = 'my_password_001'


def locate_service(path, host, port):
    xjson = dict()
    xjson['path'] = path
    xjson['check'] = md5(path + LOCAT_SERVICE_PWD)

    if port is None:
        port = DEFAULT_LOCATE_SERVICE_PORT

    res = requests.post(f'http://{host}:{port}/api', json=xjson, timeout=3.0)
    xjson = json.loads(res.text)
    result = xjson['result']
    return result


def is_exist(path, host=None, port=None):
    if host is None or host in set(['localhost', '127.0.0.1']):
        return has_content(path)[1]
    return locate_service(path, host, port)


class TvtsException(Exception):
    pass


def get_better_repr(val) -> str:
    try:
        val = val.item()  # Numpy obj => scalar
    except:
        pass
    if val is None:
        val_str = 'None'
    elif type(val) == float:
        val_str = f'{val:.5f}'
    else:
        val_str = f'{val}'
    return val_str


def shorter_dt(dt: datetime.datetime) -> str:
    assert isinstance(dt, datetime.datetime)
    return '\n'.join(dt.strftime(DATETIME_FORMAT).split(' '))


@dataclass
class Tvts(object):

    name: str
    host: str
    port: int
    db_name: str
    table_name: str
    table_name_4batch: str
    train_id: int
    is_temp: bool
    params: dict
    save_freq: int
    save_dir: str
    memo: Optional[str] = 'No MEMO.'
    init_weights: Optional[str] = None

    def __init__(
            self,
            name: str,
            mongo_link: Optional[pm.MongoClient] = None,
            host: Optional[str] = DEFAULT_HOST,
            port: Optional[int] = DEFAULT_PORT,
            db: Optional[str] = DEFAULT_DB_NAME,
            table_prefix: Optional[str] = DEFAULT_TABLE_PREFIX,
            old_train_id: Optional[int] = None,
            memo: Optional[str] = '(No memo)',
            is_temp: Optional[bool] = False,
            save_freq: Optional[int] = DEFAULT_SAVE_FREQ,
            save_dir: Optional[str] = None,
            init_weights: Optional[str] = None,
            params: Optional[dict] = {},
            **kwargs,
    ):
        assert isinstance(name, str)
        assert isinstance(memo, str)
        assert isinstance(is_temp, bool)
        assert isinstance(host, str)
        assert isinstance(port, int)
        assert isinstance(db, str)
        assert isinstance(table_prefix, str)
        assert isinstance(params, dict)
        assert isinstance(save_freq, int)
        if old_train_id is not None:
            assert isinstance(old_train_id, int)
        if old_train_id is None:
            if save_dir is None:
                raise TvtsException(f'Please specify the save_dir at where weights are saved!')
            else:
                assert isinstance(save_dir, str)
        if init_weights is not None:
            assert isinstance(init_weights, str) and len(init_weights) > 0
        if old_train_id is not None:
            self.is_resume = True
        else:
            self.is_resume = False

        memo = '(tvts.py) ' + memo
        self.name = name

        self.mongo_link = mongo_link
        if self.mongo_link is not None:
            self.host = self.mongo_link.HOST
            self.port = self.mongo_link.PORT
        else:
            self.host = host
            self.port = port
        self.db_name = db
        self.table_name = table_prefix + '_' + name
        self.table_name_4batch = table_prefix + '_' + name + '_4batch'

        # conn
        self.conn()

        if old_train_id is not None:
            # get train id
            self.train_id = old_train_id
            cur = self.table.find({
                'train_id': self.train_id,
            }).sort('epoch', pm.ASCENDING)
            first_row = None
            for first_row in cur:
                break
            if first_row is None:
                raise TvtsException(f'Old train id {old_train_id} is not existed!')

            self.memo = first_row['memo']
            self.is_temp = first_row['is_temp']

            self.params = copy.deepcopy(first_row)
            del self.params['_id']
            self.save_freq = first_row['save_freq']
            self.save_dir = first_row['save_dir']
            self.init_weights = first_row.get('init_weights', None)
        else:
            # get train id
            self.train_id = self._get_next_train_id()
            print(f'> TVTS: id of this training is {self.train_id}', file=sys.stderr)

            self.memo = memo
            self.is_temp = is_temp

            self.params = params
            for k in self.params.keys():
                if k in RESERVED_KEYS:
                    raise TvtsException(f'Key "{k}" is reserved and cannot be used by the user.')

            self.save_freq = save_freq
            self.save_dir = save_dir
            self.init_weights = init_weights

            # set important params
            self.params['name'] = self.name
            self.params['memo'] = self.memo
            self.params['is_temp'] = int(self.is_temp)

            self.params['train_id'] = self.train_id
            self.params['save_freq'] = self.save_freq
            self.params['save_dir'] = self.save_dir
            if self.init_weights is not None:
                self.params['init_weights'] = self.init_weights

        # datetime recorder
        self.dt = datetime.datetime.now()
        self.dt4batch = datetime.datetime.now()

    def conn(self) -> None:
        if self.mongo_link is not None:
            self.client = self.mongo_link
        else:
            self.client = pm.MongoClient(self.host, self.port)
        self.db = self.client[self.db_name]
        self.table = self.db[self.table_name]
        self.table_4batch = self.db[self.table_name_4batch]

    def before_save_to_pickle(self) -> None:
        self.client = None
        self.db = None
        self.table = None
        self.table_4batch = None

    def after_load_from_pickle(self) -> None:
        self.conn()

    def mark_start_dt(self) -> None:
        """
        Explicitly set the start datetime of an epoch.

        Start datetime of an epoch is set implicitly as the end datetime of last epoch, but if there are time-cost
        operations between 2 epochs, the explicitly setting is needed.
        :return: void
        """
        self.dt = datetime.datetime.now()
        self.dt4batch = datetime.datetime.now()

    def resume(
            self,
            xtrain_id: int,
            xepoch: Optional[int] = 0,
            keys_of_data_to_restore: Optional[Sequence[str]] = None
    ) -> Optional[List[str]]:
        """
        Specify the starting point of this training and get the path to the saved model or weights.

        :param xtrain_id: From which training to resume.
        :param xepoch: From which epoch of that training to resume.
                       If xepoch = 0, we will resume it is the last epoch with saved weights.
        :param keys_of_data_to_restore: Which keys to restore from the parent record.
        :return: None (xtrain_id=0) or List[ relative path to saved weights, dir of the base ]
        """
        assert isinstance(xtrain_id, int)
        assert isinstance(xepoch, float) or isinstance(xepoch, int)
        xepoch = float(xepoch)
        if keys_of_data_to_restore is not None:
            assert isinstance(keys_of_data_to_restore, Sequence)
            for el in keys_of_data_to_restore:
                assert isinstance(el, str)
                if el in RESERVED_KEYS:
                    raise TvtsException(f'Key "{el}" is reserved and cannot be used by the user.')

        if xtrain_id == 0:
            return None

        # get check point
        if not xepoch:
            ckpts = list(self.table.find({
                'train_id': xtrain_id,
                'save_rel_path': {'$ne': None},
            }).sort('epoch', pm.DESCENDING).limit(1))
            if not len(ckpts):
                raise TvtsException(f'resume: There is no saved weight in train with id {xtrain_id}!')
            ckpt = ckpts[0]
        else:
            if xepoch <= 0:
                raise TvtsException(f'Epoch value {xepoch} must be >= 1!')
            ckpt = self.table.find_one({
                'train_id': xtrain_id,
                'epoch': xepoch,
            })
            if ckpt is None:
                print(f'PI={xtrain_id}, PE={xepoch} is not found!')
                return None

        # validate
        if ckpt is None:
            raise TvtsException(f'resume: There is no saved record of epoch {xepoch} in train with id {xtrain_id}!')
        if ckpt.get('is_temp', 0) and not self.is_temp:
            raise TvtsException(f'resume: You cannot run a formal training with a temporary parent!')
        save_rel_path = ckpt.get('save_rel_path', None)
        if not save_rel_path:
            raise TvtsException(f'resume: There is no saved weight of epoch {xepoch} in train with id {xtrain_id}!')
        assert isinstance(save_rel_path, str)

        # restore the data from parent record
        if keys_of_data_to_restore is not None:
            for k in keys_of_data_to_restore:
                self.params[k] = ckpt.get(k, None)

        # set parent
        self.params['parent_id'] = xtrain_id
        self.params['parent_epoch'] = float(ckpt['epoch'])

        return save_rel_path, ckpt['save_dir']

    def get_save_name(self, epoch: int) -> str:
        assert isinstance(epoch, int)
        return f'{self.name}-{self.train_id}-{epoch}'

    def save_batch(
            self,
            xepoch: float,
            xbatch: int,
            params: Optional[dict] = {},
            is_batch_global: Optional[bool] = False
    ) -> None:
        req = 'Epoch must be decimal equal to or greater than 0.0. Batch must be integers equal to or greater than 1 !'
        ex = TvtsException(req)
        try:
            xepoch = float(xepoch)
            xbatch = int(xbatch)
            if xepoch < 0:
                raise ex
            if xbatch <= 0:
                raise ex
            params = dict(params)
            for k in params.keys():
                if k in RESERVED_KEYS:
                    raise TvtsException(f'Key "{k}" is reserved and cannot be used by the user.')
            is_batch_global = bool(is_batch_global)
        except ValueError as vex:
            raise vex

        # a copy of params
        data = copy.deepcopy(self.params)
        # apply update
        for k in params.keys():
            v = params[k]
            data[k] = v
        data['epoch'] = xepoch
        data['batch'] = xbatch

        if is_batch_global:
            data['global_batch'] = xbatch

        # datetime of record
        xnow = datetime.datetime.now()
        data['datetime'] = xnow
        data['from_datetime'] = self.dt4batch
        data['to_datetime'] = xnow
        data['duration_in_sec'] = (xnow - self.dt4batch).total_seconds()
        self.dt4batch = xnow

        # insert data into db
        # IMPORTANT: db and table will be newly created at the 1st insertion if they are not there yet.
        # self.table_4batch.insert_one(data)
        glb = data.get('global_batch', None)
        if glb is not None:
            self.table_4batch.update_one({
                'train_id': self.train_id,
                'global_batch': glb,
            }, {
                '$set': data,
            },  upsert=True)
        else:
            self.table_4batch.update_one({
                'train_id': self.train_id,
                'epoch': xepoch,
                'batch': xbatch,
            }, {
                '$set': data,
            },  upsert=True)

        # add index if it is not there yet
        indexes = set(self.table_4batch.index_information().keys())
        if not INDEX_NAME_4BATCH in indexes:
            self.table_4batch.create_index([
                ('train_id', pm.ASCENDING),
                ('epoch', pm.ASCENDING),
                ('batch', pm.ASCENDING),
            ], unique=True, name=INDEX_NAME_4BATCH)
            print(f'> TVTS: Created index {INDEX_NAME_4BATCH}', file=sys.stderr)
        if not INDEX_NAME2_4BATCH in indexes:
            self.table_4batch.create_index([
                ('train_id', pm.ASCENDING),
                ('global_batch', pm.ASCENDING),
            ], name=INDEX_NAME2_4BATCH)
            print(f'> TVTS: Created index {INDEX_NAME2_4BATCH}', file=sys.stderr)

    def save_epoch(
            self,
            xepoch: float,
            params: Optional[dict] = {},
            save_rel_path=None,
            save_dir=None
    ) -> None:
        req = 'Epoch must be a positive decimal !'
        ex = TvtsException(req)
        try:
            xepoch = float(xepoch)
            if xepoch <= 0.0:
                raise ex
            params = dict(params)
            for k in params.keys():
                if k in RESERVED_KEYS:
                    raise TvtsException(f'Key "{k}" is reserved and cannot be used by the user.')
            if save_rel_path is not None:
                save_rel_path = str(save_rel_path)
        except ValueError as vex:
            raise vex

        # a copy of params
        data = self.table.find_one({
            'train_id': self.train_id,
            'epoch': xepoch,
        })
        if data is None:
            data = copy.deepcopy(self.params)
        # apply update
        for k in params.keys():
            v = params[k]
            self.params.pop(k, None)  # This value does not belong to self.params but introduced by resuming old_train_id.
            data[k] = v
        data['epoch'] = xepoch

        # save path of model
        if save_rel_path is not None:
            data['save_rel_path'] = save_rel_path
        if save_dir is not None:
            data['save_dir'] = save_dir

        # datetime of record
        xnow = datetime.datetime.now()
        data['datetime'] = xnow
        data['from_datetime'] = self.dt
        data['to_datetime'] = xnow
        data['duration_in_sec'] = (xnow - self.dt).total_seconds()
        self.dt = xnow

        # upsert data into db
        # IMPORTANT: db and table will be newly created at the 1st insertion if they are not there yet.
        # IMPORTANT: data with the existed key will be merged into the original data
        # https://stackoverflow.com/questions/60883397/using-pymongo-upsert-to-update-or-create-a-document-in-mongodb-using-python
        self.table.update_one({
            'train_id': data['train_id'],
            'epoch': data['epoch'],
        }, {
            '$set': data,
        }, upsert=True)

        # add index if it is not there yet
        indexes = set(self.table.index_information().keys())
        if not INDEX_NAME in indexes:
            self.table.create_index([
                ('train_id', pm.ASCENDING),
                ('epoch', pm.ASCENDING),
            ], unique=True, name=INDEX_NAME)
            print(f'> TVTS: Created index {INDEX_NAME}', file=sys.stderr)

    def _get_next_train_id(self) -> int:
        """
        Pick up the next train id automatically.
        This is just a private routine for this class, not for directly usage from user.

        :return: int Next train id.
        """
        cursor = self.table.find().sort('train_id', pm.DESCENDING).limit(1)
        id = 1
        for i, row in enumerate(cursor):
            id = int(row['train_id']) + 1
            break

        cursor4batch = self.table_4batch.find().sort('train_id', pm.DESCENDING).limit(1)
        id4batch = 1
        for i, row in enumerate(cursor4batch):
            id4batch = int(row['train_id']) + 1
            break

        xmax = max(id, id4batch)

        return xmax


class TvtsVisualization(object):

    def __init__(
        self,
        name: Union[str, Sequence[str]],
        is_temp: Optional[bool] = None,
        host: Optional[str] = DEFAULT_HOST,
        port: Optional[int] = DEFAULT_PORT,
        db: Optional[str] = DEFAULT_DB_NAME,
        table_prefix: Optional[str] = DEFAULT_TABLE_PREFIX,
        save_dir: Optional[str] = None,
        client: Optional[pm.MongoClient] = None,
        service_host: Optional[str] = None,
        service_port: Optional[int] = None,
    ):
        assert isinstance(name, str) or isinstance(name, Iterable)
        if isinstance(name, Iterable):
            for the_name in name:
                assert isinstance(the_name, str)
        if is_temp is not None:
           assert isinstance(is_temp, bool)
        assert isinstance(host, str)
        assert isinstance(port, int)
        assert isinstance(db, str)
        assert isinstance(table_prefix, str)
        if save_dir is not None:
            assert isinstance(save_dir, str)
        if client is not None:
            assert isinstance(client, pm.MongoClient)

        if isinstance(name, str):
            self.names = [name]
        else:
            self.names = []
            for the_name in name:
                self.names.append(the_name)
        if is_temp is not None:
            self.temp_value = int(is_temp)
        else:
            self.temp_value = None
        self.host = host
        self.port = port
        self.db_name = db
        self.table_names = []
        self.table_names_4batch = []
        for the_name in self.names:
            self.table_names.append(table_prefix + '_' + the_name)
            self.table_names_4batch.append(table_prefix + '_' + the_name + '_4batch')
        self.save_dir = save_dir

        self.service_host = service_host
        self.service_port = service_port

        # conn
        if client is not None:
            self.client = client
        else:
            self.client = pm.MongoClient(host, port)
        self.db = self.client[self.db_name]
        self.tables = [self.db[table_name] for table_name in self.table_names]
        self.tables_4batch = [self.db[table_name] for table_name in self.table_names_4batch]

    def setTemp(self, is_temp: Optional[bool] = None) -> None:
        if is_temp is not None:
            assert isinstance(is_temp, bool)
            self.temp_value = int(is_temp)
        else:
            self.temp_value = None

    def setSaveDir(self, save_dir: Optional[str] = None) -> None:
        if save_dir is not None:
            assert isinstance(save_dir, str)
            self.save_dir = save_dir
        else:
            self.save_dir = None

    def show_title(self):
        # print key info
        if self.temp_value == 0:
            temp_repr = 'only formal data'
        elif self.temp_value == 1:
            temp_repr = 'only temporary data'
        else:
            temp_repr = 'all data'
        xinfo_bar = f'Name: {self.names[0]} from {self.host}:{self.port} {self.db_name}.{self.table_names[0]}' \
                    f' @{str(datetime.datetime.now())[:-3]} ({temp_repr})\nSpecified save_dir: "{self.save_dir}"'
        for i, name in enumerate(self.names):
            if i == 0:
                continue
            xinfo_bar += f'\n@{i}: {name}'
        print(xinfo_bar)

    def summary(self, keys_str: Optional[str] = None) -> None:
        LONG_TEXT_WIDTH_OF_COLUMN = 20

        self.show_title()

        if keys_str is not None:
            assert isinstance(keys_str, str)
            keys_str = keys_str.split(',')
        else:
            keys_str = []
        xvalue_dict = {}

        # table title
        # https://stackoverflow.com/questions/9535954/printing-lists-as-tabular-data
        t = Texttable(max_width=0)
        title_list = [
            'temp',
            'train\nid',
            'parent\nrecord',
            'init\nweights',

            'epoch',
            'epoch\ncount',
            'save\nexisted/all',
            'save\ndir',

            'from', 
            'to',
            'duration',
            'batch\nrecords',

            *[f'{k}\n(begin:end\nmin:max\nexp:std)' for k in keys_str],
            'memo'
        ]

        def insert_titles():
            t.add_row(title_list)

        insert_titles()

        # aggregate info
        xmatch = {}
        if self.temp_value is not None:
            xmatch = {'is_temp': self.temp_value}
        pipeline = [
            {
                '$match': xmatch
            },
            {
                '$group': {
                    '_id': '$train_id',
                    'epoch_count': {
                        '$sum': 1,
                    }
                }
            },
            {
                '$sort': {'_id': pm.ASCENDING},
            }
        ]
        cursor = self.tables[0].aggregate(pipeline)
        pipeline4batch = [
            {
                '$match': xmatch
            },
            {
                '$group': {
                    '_id': '$train_id',
                    'batch_count': {
                        '$sum': 1,
                    }
                }
            },
            {
                '$sort': {'_id': pm.ASCENDING},
            }
        ]
        cursor4batch = self.tables_4batch[0].aggregate(pipeline4batch)

        # get the batch records count aggregation
        train2batch_cnt_map = {}
        train2batch_data_map = {}
        train2batch_epoch_cnt_map = {}
        xset_train_ids_4batch = set()
        na = 'N/A'
        for k in keys_str:
            xvalue_dict[k] = dict(begin=na, end=na, min=na, max=na, exp=na, std=na, val=[])
        for row in cursor4batch:
            train_id = int(row['_id'])
            count = int(row['batch_count'])
            train2batch_cnt_map[train_id] = count
            xset_train_ids_4batch.add(train_id)
            cur = self.tables_4batch[0].find({
                'train_id': train_id,
            }).sort([
                ('epoch', pm.DESCENDING),
                ('batch', pm.DESCENDING)
            ])
            last_row = cur[0]
            first_row = cur[count - 1]

            for k in keys_str:
                xvalue_dict[k]['begin'] = first_row.get(k, na)
                xvalue_dict[k]['end'] = last_row.get(k, na)

            xset_epochs = set()
            for irow in cur:
                xset_epochs.add(irow['epoch'])
                for k in keys_str:
                    v = irow.get(k, na)
                    if not (type(v) == str and v == na):
                        xvalue_dict[k]['val'].append(v)

            for k in keys_str:
                vals = np.array(xvalue_dict[k]['val'])
                if len(vals) > 0:
                    xvalue_dict[k]['min'] = vals.min()
                    xvalue_dict[k]['max'] = vals.max()
                    xvalue_dict[k]['exp'] = vals.mean()
                    xvalue_dict[k]['std'] = vals.std()

            train2batch_data_map[train_id] = {
                'last_row': last_row,
                'first_row': first_row,
                'values': xvalue_dict,
            }
            train2batch_epoch_cnt_map[train_id] = len(xset_epochs)

        # get the table
        n_rows = 0
        xset_train_ids = set()
        for row in cursor:
            n_rows += 1
            train_id = int(row['_id'])
            count = int(row['epoch_count'])
            cur = self.tables[0].find({
                'train_id': train_id,
            }).sort('epoch', pm.DESCENDING)

            last_row = cur[0]
            first_row = cur[count - 1]
            for k in keys_str:
                xvalue_dict[k] = dict(
                    begin=first_row.get(k, na),
                    end=last_row.get(k, na),
                    min=na,
                    max=na,
                    exp=na,
                    std=na,
                    val=[]
                )

            count_path, count_path_exists = 0, 0
            for irow in cur:
                for k in keys_str:
                    v = irow.get(k, na)
                    if not(type(v) == str and v == na):
                        xvalue_dict[k]['val'].append(v)

                # count of saved
                save_rel_path = irow.get('save_rel_path', '')
                save_dir = self.save_dir if self.save_dir is not None else irow.get('save_dir', '')
                if save_rel_path:
                    count_path += 1
                    abs_path = os.path.join(save_dir, save_rel_path)
                    if is_exist(abs_path, self.service_host, self.service_port):
                        count_path_exists += 1

            for k in keys_str:
                vals = np.array(xvalue_dict[k]['val'])
                if len(vals) > 0:
                    xvalue_dict[k]['min'] = vals.min()
                    xvalue_dict[k]['max'] = vals.max()
                    xvalue_dict[k]['exp'] = vals.mean()
                    xvalue_dict[k]['std'] = vals.std()

            xvalues_list = []
            for k in keys_str:
                xvalues_list.append(
                    f"{get_better_repr(xvalue_dict[k]['begin'])}"
                    f":{get_better_repr(xvalue_dict[k]['end'])}"
                    f"\n{get_better_repr(xvalue_dict[k]['min'])}"
                    f":{get_better_repr(xvalue_dict[k]['max'])}"
                    f"\n{get_better_repr(xvalue_dict[k]['exp'])}"
                    f":{get_better_repr(xvalue_dict[k]['std'])}"
                )

            t.add_row([
                "Y" if last_row.get("is_temp", 0) else "",
                train_id,
                f'{last_row.get("parent_id", 0)}-{last_row.get("parent_epoch", 0.0)}',
                long_text_to_block(str(last_row.get('init_weights', '')), LONG_TEXT_WIDTH_OF_COLUMN),

                f'{first_row["epoch"]} ~ {last_row["epoch"]}',
                count,
                f'{count_path_exists}/{count_path}',
                long_text_to_block(str(last_row.get('save_dir', '')), LONG_TEXT_WIDTH_OF_COLUMN),

                shorter_dt(first_row['from_datetime']),
                shorter_dt(last_row['to_datetime']),
                last_row['to_datetime'] - first_row['from_datetime'],
                train2batch_cnt_map.get(train_id, 0),

                *xvalues_list,

                long_text_to_block(str(first_row.get('memo', '')), LONG_TEXT_WIDTH_OF_COLUMN),
            ])
            xset_train_ids.add(train_id)
            if n_rows and 0 == n_rows % 5:
                insert_titles()

        # if there are records for batch only
        xset_train_id_batch_only = xset_train_ids_4batch - xset_train_ids
        if xset_train_id_batch_only:
            xlist = list(xset_train_id_batch_only)
            xlist = sorted(xlist)
            for train_id in xlist:
                n_rows += 1
                data = train2batch_data_map[train_id]
                epoch_count = train2batch_epoch_cnt_map.get(train_id, 0)
                last_row = data['last_row']
                first_row = data['first_row']

                batch_cnt = train2batch_cnt_map.get(train_id, 0)

                xvalue_dict = data['values']
                xvalues_list = []
                for k in keys_str:
                    xvalues_list.append(
                        f"{get_better_repr(xvalue_dict[k]['begin'])}"
                        f":{get_better_repr(xvalue_dict[k]['end'])}"
                        f"\n{get_better_repr(xvalue_dict[k]['min'])}"
                        f":{get_better_repr(xvalue_dict[k]['max'])}"
                        f"\n{get_better_repr(xvalue_dict[k]['exp'])}"
                        f":{get_better_repr(xvalue_dict[k]['std'])}"
                    )

                xlist = [
                    "Y" if last_row.get("is_temp", 0) else "",
                    f'{train_id}(batch only)',
                    f'{last_row.get("parent_id", 0)}-{last_row.get("parent_epoch", 0.0)}',
                    long_text_to_block(str(last_row.get('init_weights', '')), LONG_TEXT_WIDTH_OF_COLUMN),

                    ' ',
                    epoch_count,
                    ' ',
                    long_text_to_block(str(last_row.get('save_dir', '')), LONG_TEXT_WIDTH_OF_COLUMN),

                    shorter_dt(first_row['datetime']),
                    shorter_dt(last_row['datetime']),
                    last_row['datetime'] - first_row['datetime'],
                    batch_cnt,

                    *xvalues_list,

                    long_text_to_block(str(last_row.get('memo', '')), LONG_TEXT_WIDTH_OF_COLUMN),
                ]
                t.add_row(xlist)
                if n_rows and 0 == n_rows % 5:
                    insert_titles()
        if n_rows and 0 != n_rows % 5:
            insert_titles()
        elif 0 == n_rows:
            t.add_row(['N/A'] * len(title_list))

        print(t.draw())

    plot_4batch_alpha = 0.6

    def plot_4batch(
            self,
            xtrain_id=1,
            xmetrics_groups='cost',
            xm_hyper=None,
            xepoch_range='0:0',
            xbatch_range='0:0',  # TODO
    ) -> None:
        # validate parameters
        assert isinstance(xtrain_id, int)
        assert isinstance(xmetrics_groups, str)
        if xm_hyper is not None:
            assert isinstance(xm_hyper, str)
        assert isinstance(xepoch_range, str)

        try:
            xepoch_begin, xepoch_end = xepoch_range.split(':')
            xepoch_begin = float(xepoch_begin)
            xepoch_end = float(xepoch_end)
        except Exception as ex:
            print(traceback.format_exc())
        
        assert xepoch_begin >= 0.
        assert xepoch_end >= 0.
        if xepoch_begin + xepoch_end > 0 and xepoch_end:
            assert xepoch_begin < xepoch_end

        try:
            xbatch_begin, xbatch_end = xbatch_range.split(':')
            xbatch_begin = int(xbatch_begin)
            xbatch_end = int(xbatch_end)
        except Exception as ex:
            print(traceback.format_exc())
            
        assert xbatch_begin >= 0
        assert xbatch_end >= 0
        if xbatch_begin + xbatch_end > 0 and xbatch_end:
            assert xbatch_begin < xbatch_end
        
        if xepoch_begin + xepoch_end > 0 and xbatch_begin + xbatch_end > 0:
            raise f'You cannot set epoch range and batch range at the same time!'

        # parse metrics groups
        xmetrics_groups = xmetrics_groups.split(',')
        xm_name_idxs = []  # 0|1,0|0
        xm_train_ids = []  # 4|3,4|4
        xm_groups = []  # key1|key2,key3|key4
        xm_groups_with_idx_and_id = []  # key1@0-7|key2@1-5,key3@0-7|key4@1-5
        for group in xmetrics_groups:
            name_idxs = []
            ids = []
            keys = []
            keys_with_idx_and_id = group.split('|')
            for i, xstr in enumerate(keys_with_idx_and_id):
                xlist = xstr.split('@')
                xlen = len(xlist)
                
                xname_idx = 0
                xid = xtrain_id
                xkey = xstr
                if xlen == 1:
                    keys_with_idx_and_id[i] = f'{xstr}@{xname_idx}-{xid}'
                elif xlen == 2:
                    xstr, name_and_id = xlist
                    xkey = xstr
                    try:
                        xname_idx, xid = name_and_id.split('-')
                        xname_idx = int(xname_idx)
                        xid = int(xid)
                    except:
                        raise Exception(f'Format of name and id is not right in "@{name_and_id}"')
                else:
                    raise Exception(f'Only 1 @ allowed in "{xstr}"')
                
                name_idxs.append(xname_idx)
                ids.append(xid)
                keys.append(xkey)
                
            xm_name_idxs.append(name_idxs)
            xm_train_ids.append(ids)
            xm_groups.append(keys)
            xm_groups_with_idx_and_id.append(keys_with_idx_and_id)
        xm_n_groups = len(xm_groups)
        
        # what data to fetch for each row
        xm_keys = []
        for group in xm_groups:
            xm_keys.extend(group)
        if xm_hyper is not None:
            xm_keys.append(xm_hyper)
        xm_keys = sorted(set(xm_keys))  # sigularization    
            
        def find_record(xc_name_idx, xc_train_id, xc_up_to_epoch=0):
            """
            Enclosure method.
            Find records of training with id of xc_train_id up to epoch of xc_up_to_epoch.

            :param xc_train_id: The training's id.
            :param xc_up_to_epoch: Up to this epoch.
            :return: dict or None
            """
            print(f'**** find_record(xc_name_idx={xc_name_idx}, xc_train_id={xc_train_id}, xc_up_to_epoch={xc_up_to_epoch}')

            # get cursor from db
            xc_key2y_map, xc_key2x_map = {}, {}
            xc_key2glb_x_map = {}

            query = {
                'train_id': xc_train_id,
            }
            if xc_up_to_epoch:
                query['epoch'] = {
                    '$lte': xc_up_to_epoch
                }
            cursor = self.tables_4batch[xc_name_idx].find(query).sort([
                ('epoch', pm.ASCENDING),
                ('batch', pm.ASCENDING)
            ])

            # init containers
            for k in xm_keys:
                xc_key2y_map[k] = []
                xc_key2x_map[k] = []
                xc_key2glb_x_map[k] = []

            # get data from cursor
            count = 0
            for i, row in enumerate(cursor):
                count += 1
                xi_epoch = float(row['epoch'])
                for k in xm_keys:
                    v = row.get(k, None)
                    if v is None:
                        continue
                    xc_key2y_map[k].append(v)
                    xc_key2x_map[k].append(xi_epoch)  # useless
                    xc_key2glb_x_map[k].append(xi_epoch)  # will be revised later

            # if no data found
            if count == 0:
                return None

            xc_last_row = row

            # the dict to return
            xc_map = {
                'key2y_map': xc_key2y_map,
                'key2x_map': xc_key2x_map,  # useless
                'key2glb_x_map': xc_key2glb_x_map,  # will be revised later
                'key2glb_epoch_map': copy.deepcopy(xc_key2glb_x_map),  # will be revised later
                'train_id': xc_train_id,
                'up_to_epoch': float(xc_last_row['epoch']),
                'parent_id': int(xc_last_row.get('parent_id', 0)),
                'parent_epoch': float(xc_last_row.get('parent_epoch', 0.0)),
                'base_epoch': 0,
            }
            return xc_map

        # fetch data
        xm_key2glb_epoch_map_list_list, xm_key2glb_x_map_list_list, xm_key2y_map_list_list, = [], [], [], 
        xm_list_train_ids_show_list_list = []
        xm_train_id2map_map_list_list = []
        xcache = {}
        for iiii, idxs in enumerate(xm_name_idxs):
            xm_key2glb_epoch_map_list, xm_key2glb_x_map_list, xm_key2y_map_list, = [], [], [], 
            xm_list_train_ids_show_list = []
            xm_train_id2map_map_list = []
            for jjjj, xi_the_id in enumerate(xm_train_ids[iiii]):
                idx = idxs[jjjj]    
                xtuple = (idx, xi_the_id)
                if xtuple in xcache:
                    print(f'**** Hit cache: {xtuple}')
                    xm_key2glb_epoch_map_final,\
                        xm_key2glb_x_map_final,\
                        xm_key2y_map_final,\
                        xm_list_train_ids_show_final,\
                        xm_train_id2map_map_final, = xcache[xtuple]
                else:
                
                    # recursively get data of ancestor trainings
                    xm_train_id2map_map = {}
                    xm_set_train_ids, xm_list_train_ids = set(), []
                    xi_train_id = xi_the_id
                    xi_up_to_epoch = 0
                    while True:
                        if xi_train_id in xm_set_train_ids:
                            raise TvtsException(f'Error: loop back! {xm_list_train_ids} meet {xi_train_id}')
                        xm_set_train_ids.add(xi_train_id)
                        xm_list_train_ids.append(xi_train_id)
                        xi_map = find_record(idx, xi_train_id, xi_up_to_epoch)
                        if xi_map is None:
                            print(f'> Error: find no record with train_id {xi_train_id} !')
                            return
                        xm_train_id2map_map[xi_train_id] = xi_map
                        xi_train_id = int(xi_map.get('parent_id', 0))
                        xi_up_to_epoch = float(xi_map.get('parent_epoch', 0.0))
                        if not xi_train_id:  # if it is the eldest ancestor
                            break
                    # order by datetime
                    xm_list_train_ids = xm_list_train_ids[::-1]

                    # build global epoch numbers
                    xm_list_global_entries = ['key2glb_epoch_map', ]
                    xm_n_train_ids = len(xm_list_train_ids)
                    for i, xi_train_id in enumerate(xm_list_train_ids[:-1]):
                        xi_map = xm_train_id2map_map[xi_train_id]
                        xi_last_epoch_loaded = xi_map['up_to_epoch']
                        for j in range(i + 1, xm_n_train_ids):
                            xj_train_id = xm_list_train_ids[j]
                            xj_map = xm_train_id2map_map[xj_train_id]
                            xj_map['base_epoch'] += xi_last_epoch_loaded
                            for entry in xm_list_global_entries:
                                for k in xm_keys:
                                    v = xj_map[entry].get(k, None)
                                    if v is None:
                                        continue
                                    for ii, _ in enumerate(xj_map[entry][k]):
                                        xj_map[entry][k][ii] += xi_last_epoch_loaded

                    # build global batch numbers
                    xm_list_global_entries = ['key2glb_x_map', ]
                    i_glb_batch_base = 0
                    for i, xi_train_id in enumerate(xm_list_train_ids):
                        xi_map = xm_train_id2map_map[xi_train_id]
                        for entry in xm_list_global_entries:
                            for k in xm_keys:
                                i_glb_batch = i_glb_batch_base
                                v = xi_map[entry].get(k, None)
                                if v is None:
                                    continue
                                for ii, _ in enumerate(xi_map[entry][k]):
                                    i_glb_batch += 1
                                    xi_map[entry][k][ii] = i_glb_batch
                        i_glb_batch_base = i_glb_batch
                        
                    # Make data for finally plotting
                    xm_key2glb_epoch_map, xm_key2glb_x_map, xm_key2y_map, xm_key2id_map = {}, {}, {}, {}
                    for k in xm_keys:
                        xm_key2glb_epoch_map[k] = []
                        xm_key2glb_x_map[k] = []
                        xm_key2y_map[k] = []
                        xm_key2id_map[k] = []

                    xm_list_train_ids_show = copy.deepcopy(xm_list_train_ids)

                    for xi_train_id in xm_list_train_ids_show:
                        xi_map = xm_train_id2map_map[xi_train_id]
                        for k in xm_keys:
                            xm_key2glb_epoch_map[k].extend(xi_map['key2glb_epoch_map'][k])
                            xm_key2glb_x_map[k].extend(xi_map['key2glb_x_map'][k])
                            xm_key2y_map[k].extend(xi_map['key2y_map'][k])
                            xm_key2id_map[k].extend([xi_train_id] * len(xi_map['key2glb_x_map'][k]))
                    
                    # epoch range
                    if xepoch_begin + xepoch_end == 0:
                        xm_key2glb_epoch_map_final, xm_key2glb_x_map_final, xm_key2y_map_final, = xm_key2glb_epoch_map, xm_key2glb_x_map, xm_key2y_map
                        xm_key2id_map_final = xm_key2id_map
                        xm_list_train_ids_show_final = xm_list_train_ids_show
                        xm_train_id2map_map_final = xm_train_id2map_map
                    else:
                        if not xepoch_begin:
                            xepoch_begin = -math.inf
                        if not xepoch_end:
                            xepoch_end = math.inf
                        xm_key2glb_epoch_map_final, xm_key2glb_x_map_final, xm_key2y_map_final, = {}, {}, {}, 
                        xm_key2id_map_final = {}
                        xm_train_id2map_map_final = {}
                        xm_set_train_ids_show_final = set()
                        for k in xm_keys:
                            xm_key2glb_epoch_map_final[k] = []
                            xm_key2glb_x_map_final[k] = []
                            xm_key2y_map_final[k] = []
                            xm_key2id_map_final[k] = []
                            for o, xi_epoch in enumerate(xm_key2glb_epoch_map[k]):
                                if xepoch_begin <= xi_epoch <= xepoch_end:
                                    xm_key2glb_epoch_map_final[k].append(xi_epoch)
                                    xm_key2glb_x_map_final[k].append(xm_key2glb_x_map[k][o])
                                    xm_key2y_map_final[k].append(xm_key2y_map[k][o])
                                    xi_id = xm_key2id_map[k][o]
                                    xm_set_train_ids_show_final.add(xi_id)
                                    xm_key2id_map_final[k].append(xi_id)
                        
                        xm_list_train_ids_show_final = sorted(xm_set_train_ids_show_final)
                        
                        for xi_id in sorted(xm_list_train_ids_show_final):
                            xm_train_id2map_map_final[xi_id] = {}
                            for kkkk, vvvv in xm_train_id2map_map[xi_id].items():
                                key_prefix = "key2"
                                if kkkk[:len(key_prefix)] != key_prefix:
                                    xm_train_id2map_map_final[xi_id][kkkk] = copy.deepcopy(vvvv)
                                else:
                                    xm_train_id2map_map_final[xi_id][kkkk] = {}
                                    for k in xm_keys:
                                        xm_train_id2map_map_final[xi_id][kkkk][k] = []
                                        for o, xi_epoch in enumerate(xm_train_id2map_map[xi_id]['key2glb_epoch_map'][k]):
                                            if xepoch_begin <= xi_epoch <= xepoch_end:
                                                xm_train_id2map_map_final[xi_id][kkkk][k].append(vvvv[k][o])
                                                
                    # batch range
                    # Note: "epoch range" and "batch range" are exclusive due to the "validation of parameters"
                    if xbatch_begin + xbatch_end == 0:
                        xm_key2glb_batch_map_final, xm_key2glb_x_map_final, xm_key2y_map_final, = xm_key2glb_x_map, xm_key2glb_x_map, xm_key2y_map
                        xm_key2id_map_final = xm_key2id_map
                        xm_list_train_ids_show_final = xm_list_train_ids_show
                        xm_train_id2map_map_final = xm_train_id2map_map
                    else:
                        if not xbatch_begin:
                            xbatch_begin = -math.inf
                        if not xbatch_end:
                            xbatch_end = math.inf
                        xm_key2glb_batch_map_final, xm_key2glb_x_map_final, xm_key2y_map_final, = {}, {}, {}, 
                        xm_key2id_map_final = {}
                        xm_train_id2map_map_final = {}
                        xm_set_train_ids_show_final = set()
                        for k in xm_keys:
                            xm_key2glb_batch_map_final[k] = []
                            xm_key2glb_x_map_final[k] = []
                            xm_key2y_map_final[k] = []
                            xm_key2id_map_final[k] = []
                            for o, xi_batch in enumerate(xm_key2glb_x_map[k]):
                                if xbatch_begin <= xi_batch <= xbatch_end:
                                    xm_key2glb_batch_map_final[k].append(xi_batch)
                                    xm_key2glb_x_map_final[k].append(xm_key2glb_x_map[k][o])
                                    xm_key2y_map_final[k].append(xm_key2y_map[k][o])
                                    xi_id = xm_key2id_map[k][o]
                                    xm_set_train_ids_show_final.add(xi_id)
                                    xm_key2id_map_final[k].append(xi_id)
                        
                        xm_list_train_ids_show_final = sorted(xm_set_train_ids_show_final)
                        
                        for xi_id in sorted(xm_list_train_ids_show_final):
                            xm_train_id2map_map_final[xi_id] = {}
                            for kkkk, vvvv in xm_train_id2map_map[xi_id].items():
                                key_prefix = "key2"
                                if kkkk[:len(key_prefix)] != key_prefix:
                                    xm_train_id2map_map_final[xi_id][kkkk] = copy.deepcopy(vvvv)
                                else:
                                    xm_train_id2map_map_final[xi_id][kkkk] = {}
                                    for k in xm_keys:
                                        xm_train_id2map_map_final[xi_id][kkkk][k] = []
                                        for o, xi_batch in enumerate(xm_train_id2map_map[xi_id]['key2glb_x_map'][k]):
                                            if xbatch_begin <= xi_batch <= xbatch_end:
                                                xm_train_id2map_map_final[xi_id][kkkk][k].append(vvvv[k][o])
                                    
                    xcache[xtuple] = [
                        copy.deepcopy(xm_key2glb_epoch_map_final),
                        copy.deepcopy(xm_key2glb_x_map_final),
                        copy.deepcopy(xm_key2y_map_final),
                        copy.deepcopy(xm_list_train_ids_show_final),
                        copy.deepcopy(xm_train_id2map_map_final),
                    ]
                            
                # accumulate a group
                xm_key2glb_epoch_map_list.append(xm_key2glb_epoch_map_final)
                xm_key2glb_x_map_list.append(xm_key2glb_x_map_final)
                xm_key2y_map_list.append(xm_key2y_map_final)
                
                xm_list_train_ids_show_list.append(xm_list_train_ids_show_final)
                xm_train_id2map_map_list.append(xm_train_id2map_map_final)
                
            # accumulate all the groups
            xm_key2glb_epoch_map_list_list.append(xm_key2glb_epoch_map_list)
            xm_key2glb_x_map_list_list.append(xm_key2glb_x_map_list)
            xm_key2y_map_list_list.append(xm_key2y_map_list)
            
            xm_list_train_ids_show_list_list.append(xm_list_train_ids_show_list)
            xm_train_id2map_map_list_list.append(xm_train_id2map_map_list)

        # finally plot them
        spr = 1
        spc = xm_n_groups
        spn = 0
        fig = plt.figure(figsize=[5*xm_n_groups, 5])
        plt.subplots_adjust(wspace=0.4, top=0.8)

        for i, group in enumerate(xm_groups):
            spn += 1
            xi_ax2 = plt.subplot(spr, spc, spn)
            xi_ax1 = xi_ax2.twinx()

            xi_ax2.set_ylabel(xm_hyper)
            xi_ax1.set_title(f'{self.names[0]}@{self.host}:{self.port}\n{xm_groups_with_idx_and_id[i]}')

            # plot
            hyper_set = set()
            vertical_set = set()
            for j, k in enumerate(group):
                
                # the metrics curves
                x = xm_key2glb_x_map_list_list[i][j][k]
                x = np.array(x, dtype=float)
                y = xm_key2y_map_list_list[i][j][k]
                    
                # plot hyper param
                if xm_hyper is not None:
                    xtuple = (xm_name_idxs[i][j], xm_train_ids[i][j], )
                    if xtuple not in hyper_set:
                        xi_ax2.plot(x, xm_key2y_map_list_list[i][j][xm_hyper], alpha=TvtsVisualization.plot_4batch_alpha, label=f'{xm_hyper}@{xm_name_idxs[i][j]}-{xm_train_ids[i][j]}')
                        hyper_set.add(xtuple)
                
                    
                xline, = xi_ax1.plot(x, y, label=xm_groups_with_idx_and_id[i][j], linewidth=1, alpha=TvtsVisualization.plot_4batch_alpha)
                xcolor = xline.get_color()

                # separator vertical line for ancestor trainings
                for xi_train_id in xm_list_train_ids_show_list_list[i][j]:
                    xtuple = (xm_name_idxs[i][j], xi_train_id, )
                    if xtuple in vertical_set:
                        continue
                    xi_map = xm_train_id2map_map_list_list[i][j][xi_train_id]
                    y_min, y_max = xi_ax1.get_ylim()
                    base = min(xi_map['key2glb_x_map'][k])
                    xi_ax1.plot([base, base], [y_min, y_max], color=xcolor, alpha=TvtsVisualization.plot_4batch_alpha)
                    annot = f'#id@{xm_name_idxs[i][j]}-{xi_train_id}(glb_{base})'
                    xi_ax1.annotate(annot, xy=[base, y_min], color='black', alpha=TvtsVisualization.plot_4batch_alpha)
                    vertical_set.add(xtuple)

            xi_ax1.legend()
            xi_ax1.grid()
            xi_ax1.set_xlabel('Global Epoch')
            xi_ax2.legend()

        # show the plot window
        print('> Check and close the plotting window to continue ...')
        plt.show()
        

    decorator_shift_regexp = re.compile(r'([^\-\+]+)([-+])([^\-\+]+)')
    shift_categories_set = set(['-', '+', '0'])
    plot_alpha = 0.7

    def plot(
            self,
            xm_train_id: Optional[int] = 1,
            xm_metrics_groups: Optional[str] = 'cost',
            xm_hyper: Optional[str] = None
    ) -> None:
        """
        Plot the train with xm_train_id and its ancestors showing specified metrics and hyper params.

        :param xm_train_id: The id of the train to plot.
        :param xm_metrics_groups: The metrics groups specifier. Eg. "cost|cost_val,acc|acc_val|f1|f1_val"
        :param xm_hyper: The hyper param to check. Eg. "lr"
        :return: void
        """
        assert isinstance(xm_train_id, int)
        assert isinstance(xm_metrics_groups, str)
        if xm_hyper is not None:
            assert isinstance(xm_hyper, str)

        # get xm_metrics_groups
        xm_metrics_groups = xm_metrics_groups.split(',')
        xm_name_idxs = []  # 0|1,0|0
        xm_train_ids = []  # 4|3,4|4
        xm_groups = []  # key1|key2,key3|key4
        xm_groups_decorated = []  # key1-0.5|key2+0.5,key3-0.5|key4+0.5
        xm_groups_shift_dir = []  # -|0,+|0
        xm_groups_shift_len = []  # 0.5|0.0,0.5|0.0
        for group in xm_metrics_groups:
            keys_decorated = group.split('|')
            name_idxs = []
            ids = []
            keys = []
            dirs = []  # '-' / '+' / '0'
            lens = []  # float
            for i, xstr in enumerate(keys_decorated):
                
                xlist = xstr.split('@')
                xlen = len(xlist)
                
                xname_idx = 0
                xid = xm_train_id
                if xlen == 1:
                    keys_decorated[i] = f'{xstr}@{xname_idx}-{xid}'
                elif xlen == 2:
                    xstr, name_and_id = xlist
                    try:
                        xname_idx, xid = name_and_id.split('-')
                        xname_idx = int(xname_idx)
                        xid = int(xid)
                    except:
                        raise Exception(f'Format of name and id is not right in "@{name_and_id}"')
                else:
                    raise Exception(f'Only 1 @ allowed in "{xstr}"')
                
                matches = TvtsVisualization.decorator_shift_regexp.match(xstr)
                xkey = xstr
                xdir = '0'
                xlen = 0.0
                if matches is not None:
                    try:
                        xkey = matches[1]
                        
                        xdir = matches[2]
                        assert xdir in TvtsVisualization.shift_categories_set, f'direction "{xdir}" should be in the set {TvtsVisualization.shift_categories_set}'
                        
                        xlen = float(matches[3])
                    except Exception as ex:
                        print(ex, file=sys.stderr, flush=True)

                name_idxs.append(xname_idx)
                ids.append(xid)
                keys.append(xkey)
                dirs.append(xdir)
                lens.append(xlen)
            
            xm_name_idxs.append(name_idxs)
            xm_train_ids.append(ids)
            xm_groups.append(keys)
            xm_groups_decorated.append(keys_decorated)
            xm_groups_shift_dir.append(dirs)
            xm_groups_shift_len.append(lens)
            
        xm_n_groups = len(xm_groups)
        
        # what data to fetch for each row
        xm_keys = []
        for group in xm_groups:
            xm_keys.extend(group)
        if xm_hyper is not None:
            xm_keys.append(xm_hyper)
        xm_keys = sorted(set(xm_keys))  # sigularization

        def find_record(xc_name_idx, xc_train_id, xc_up_to_epoch=0):
            """
            Enclosure method.
            Find records of training with id of xc_train_id up to epoch of xc_up_to_epoch.

            :param xc_train_id: The training's id.
            :param xc_up_to_epoch: Up to this epoch.
            :return: dict or None
            """
            print(f'**** find_record(xc_name_idx={xc_name_idx}, xc_train_id={xc_train_id}, xc_up_to_epoch={xc_up_to_epoch}')

            # get cursor from db
            xc_key2y_map, xc_key2x_map = {}, {}
            xc_key2y_map_with_save_path, xc_key2x_map_with_save_path = {}, {}
            xc_key2glb_x_map = {}
            xc_key2glb_x_map_with_save_path = {}
            query = {
                'train_id': xc_train_id,
            }
            if xc_up_to_epoch:
                query['epoch'] = {
                    '$lte': xc_up_to_epoch
                }
            cursor = self.tables[xc_name_idx].find(query).sort('epoch', pm.ASCENDING)

            # init containers
            for k in xm_keys:
                xc_key2y_map[k] = []
                xc_key2x_map[k] = []
                xc_key2glb_x_map[k] = []
                xc_key2y_map_with_save_path[k] = []
                xc_key2x_map_with_save_path[k] = []
                xc_key2glb_x_map_with_save_path[k] = []

            # get data from cursor
            count = 0
            for i, row in enumerate(cursor):
                count += 1
                xi_epoch = float(row['epoch'])
                for k in xm_keys:
                    v = row.get(k, None)
                    if v is None:
                        continue
                    xc_key2y_map[k].append(v)
                    xc_key2x_map[k].append(xi_epoch)
                    xc_key2glb_x_map[k].append(xi_epoch)
                    save_rel_path = row.get('save_rel_path', None)
                    if save_rel_path is not None:
                        xc_key2y_map_with_save_path[k].append(v)
                        xc_key2x_map_with_save_path[k].append(xi_epoch)
                        xc_key2glb_x_map_with_save_path[k].append(xi_epoch)

            # if no data found
            if count == 0:
                return None

            xc_last_row = row

            # the dict to return
            xc_map = {
                'key2y_map': xc_key2y_map,
                'key2x_map': xc_key2x_map,
                'key2glb_x_map': xc_key2glb_x_map,
                'key2y_map_with_save_path': xc_key2y_map_with_save_path,
                'key2x_map_with_save_path': xc_key2x_map_with_save_path,
                'key2glb_x_map_with_save_path': xc_key2glb_x_map_with_save_path,
                'train_id': xc_train_id,
                'up_to_epoch': float(xc_last_row['epoch']),
                'parent_id': int(xc_last_row.get('parent_id', 0)),
                'parent_epoch': float(xc_last_row.get('parent_epoch', 0.0)),
                'base_epoch': 0,
            }
            return xc_map

        xm_key2glb_x_map_list_list, xm_key2y_map_list_list, xm_key2glb_x_map_with_save_path_list_list, xm_key2y_map_with_save_path_list_list = [], [], [], []
        xm_list_train_ids_show_list_list = []
        xm_train_id2map_map_list_list = []

        xcache = {}

        for iiii, idxs in enumerate(xm_name_idxs):
            
            xm_key2glb_x_map_list, xm_key2y_map_list, xm_key2glb_x_map_with_save_path_list, xm_key2y_map_with_save_path_list = [], [], [], []
            xm_list_train_ids_show_list = []
            xm_train_id2map_map_list = []
            for jjjj, xi_the_id in enumerate(xm_train_ids[iiii]):
                idx = idxs[jjjj]    
                xtuple = (idx, xi_the_id)
                if xtuple in xcache:
                    print(f'**** Hit cache: {xtuple}')
                    xm_key2glb_x_map,\
                        xm_key2y_map,\
                        xm_key2glb_x_map_with_save_path,\
                        xm_key2y_map_with_save_path,\
                        xm_list_train_ids_show,\
                        xm_train_id2map_map, = xcache[xtuple]
                else:
                
                    # recursively get data of ancestor trainings
                    xm_train_id2map_map = {}
                    xm_set_train_ids, xm_list_train_ids = set(), []
                    xi_train_id = xi_the_id
                    xi_up_to_epoch = 0
                    while True:
                        if xi_train_id in xm_set_train_ids:
                            raise TvtsException(f'Error: loop back! {xm_list_train_ids} meet {xi_train_id}')
                        xm_set_train_ids.add(xi_train_id)
                        xm_list_train_ids.append(xi_train_id)
                        xi_map = find_record(idx, xi_train_id, xi_up_to_epoch)
                        if xi_map is None:
                            print(f'> Error: find no record with train_id {xi_train_id} !' \
                                f' If there are only batch records, please add -b to arguments.')
                            return
                        xm_train_id2map_map[xi_train_id] = xi_map
                        xi_train_id = int(xi_map.get('parent_id', 0))
                        xi_up_to_epoch = float(xi_map.get('parent_epoch', 0.0))
                        if not xi_train_id:  # if it is the eldest ancestor
                            break
                    # order by datetime
                    xm_list_train_ids = xm_list_train_ids[::-1]

                    # build global epoch numbers
                    xm_list_global_entries = ['key2glb_x_map', 'key2glb_x_map_with_save_path']
                    xm_n_train_ids = len(xm_list_train_ids)
                    for i, xi_train_id in enumerate(xm_list_train_ids[:-1]):
                        xi_map = xm_train_id2map_map[xi_train_id]
                        xi_last_epoch_loaded = xi_map['up_to_epoch']
                        for j in range(i + 1, xm_n_train_ids):
                            xj_train_id = xm_list_train_ids[j]
                            xj_map = xm_train_id2map_map[xj_train_id]
                            xj_map['base_epoch'] += xi_last_epoch_loaded
                            for entry in xm_list_global_entries:
                                for k in xm_keys:
                                    v = xj_map[entry].get(k, None)
                                    if v is None:
                                        continue
                                    for ii, _ in enumerate(xj_map[entry][k]):
                                        xj_map[entry][k][ii] += xi_last_epoch_loaded

                    # Make data for finally plotting
                    xm_key2glb_x_map, xm_key2y_map, xm_key2glb_x_map_with_save_path, xm_key2y_map_with_save_path = {}, {}, {}, {}
                    for k in xm_keys:
                        xm_key2glb_x_map[k] = []
                        xm_key2y_map[k] = []
                        xm_key2glb_x_map_with_save_path[k] = []
                        xm_key2y_map_with_save_path[k] = []

                    xm_list_train_ids_show = copy.deepcopy(xm_list_train_ids)

                    for xi_train_id in xm_list_train_ids_show:
                        xi_map = xm_train_id2map_map[xi_train_id]
                        for k in xm_keys:
                            xm_key2glb_x_map[k].extend(xi_map['key2glb_x_map'][k])
                            xm_key2y_map[k].extend(xi_map['key2y_map'][k])
                            xm_key2glb_x_map_with_save_path[k].extend(xi_map['key2glb_x_map_with_save_path'][k])
                            xm_key2y_map_with_save_path[k].extend(xi_map['key2y_map_with_save_path'][k])
                    
                    xcache[xtuple] = [
                        copy.deepcopy(xm_key2glb_x_map),
                        copy.deepcopy(xm_key2y_map),
                        copy.deepcopy(xm_key2glb_x_map_with_save_path),
                        copy.deepcopy(xm_key2y_map_with_save_path),
                        copy.deepcopy(xm_list_train_ids_show),
                        copy.deepcopy(xm_train_id2map_map),
                    ]
                            
                # accumulate a group
                xm_key2glb_x_map_list.append(xm_key2glb_x_map)
                xm_key2y_map_list.append(xm_key2y_map)
                xm_key2glb_x_map_with_save_path_list.append(xm_key2glb_x_map_with_save_path)
                xm_key2y_map_with_save_path_list.append(xm_key2y_map_with_save_path)
                
                xm_list_train_ids_show_list.append(xm_list_train_ids_show)
                xm_train_id2map_map_list.append(xm_train_id2map_map)
                
            # accumulate all the groups
            xm_key2glb_x_map_list_list.append(xm_key2glb_x_map_list)
            xm_key2y_map_list_list.append(xm_key2y_map_list)
            xm_key2glb_x_map_with_save_path_list_list.append(xm_key2glb_x_map_with_save_path_list)
            xm_key2y_map_with_save_path_list_list.append(xm_key2y_map_with_save_path_list)
            
            xm_list_train_ids_show_list_list.append(xm_list_train_ids_show_list)
            xm_train_id2map_map_list_list.append(xm_train_id2map_map_list)
                

        spr = 1
        spc = xm_n_groups
        spn = 0
        fig = plt.figure(figsize=[5*xm_n_groups, 5])
        plt.subplots_adjust(wspace=0.4, top=0.8)

        for i, group in enumerate(xm_groups):
            spn += 1
            xi_ax2 = plt.subplot(spr, spc, spn)
            xi_ax1 = xi_ax2.twinx()

            xi_ax2.set_ylabel(xm_hyper)
            xi_ax1.set_title(f'{self.names[0]}@{self.host}:{self.port}\n{xm_groups_decorated[i]}')

            # plot
            hyper_set = set()
            vertical_set = set()
            for j, k in enumerate(group):
                
                # the metrics curves
                decorated = xm_groups_decorated[i][j]
                xdir = xm_groups_shift_dir[i][j]
                xlen = xm_groups_shift_len[i][j]
                x = xm_key2glb_x_map_list_list[i][j][k]
                x = np.array(x, dtype=float)
                y = xm_key2y_map_list_list[i][j][k]
                x2 = xm_key2glb_x_map_with_save_path_list_list[i][j][k]
                x2 = np.array(x2, dtype=float)
                y2 = xm_key2y_map_with_save_path_list_list[i][j][k]
                if '-' == xdir:
                    x -= xlen
                    x2 -= xlen
                elif '+' == xdir:
                    x += xlen
                    x2 += xlen
                    
                # plot hyper param
                if xm_hyper is not None:
                    xtuple = (xm_name_idxs[i][j], xm_train_ids[i][j], )
                    if xtuple not in hyper_set:
                        xi_ax2.plot(x, xm_key2y_map_list_list[i][j][xm_hyper], alpha=TvtsVisualization.plot_alpha, label=f'{xm_hyper}@{xm_name_idxs[i][j]}-{xm_train_ids[i][j]}')
                        hyper_set.add(xtuple)
                
                    
                xline, = xi_ax1.plot(x, y, label=decorated, linewidth=1, alpha=TvtsVisualization.plot_alpha)
                xcolor = xline.get_color()
                xi_ax1.scatter(
                    x, 
                    y,
                    s=2, color=xcolor, marker='o', alpha=TvtsVisualization.plot_alpha
                )
                # the dots representing epoch with saved model or weights
                xi_ax1.scatter(
                    x2,
                    y2,
                    s=10, color=xcolor, marker='x', alpha=TvtsVisualization.plot_alpha
                )

                # separator vertical line for ancestor trainings
                for xi_train_id in xm_list_train_ids_show_list_list[i][j]:
                    xtuple = (xm_name_idxs[i][j], xi_train_id, )
                    if xtuple in vertical_set:
                        continue
                    xi_map = xm_train_id2map_map_list_list[i][j][xi_train_id]
                    y_min, y_max = xi_ax1.get_ylim()
                    base = xi_map['base_epoch']
                    # base = min(xi_map['key2glb_x_map'][k])   # as 4batch, not suitable
                    xi_ax1.plot([base, base], [y_min, y_max], color=xcolor, alpha=TvtsVisualization.plot_alpha)
                    annot = f'#id@{xm_name_idxs[i][j]}-{xi_train_id}(glb_{base})'
                    xi_ax1.annotate(annot, xy=[base, y_min], color='black', alpha=TvtsVisualization.plot_alpha)
                    vertical_set.add(xtuple)

            xi_ax1.legend()
            xi_ax1.grid()
            xi_ax1.set_xlabel('Global Epoch')
            xi_ax2.legend()

        # show the plot window
        print('> Check and close the plotting window to continue ...')
        plt.show()


if '__main__' == __name__:
    import argparse
    import time
    import matplotlib.pyplot as plt
    from texttable import Texttable

    def _main():

        class ArgumentParser(argparse.ArgumentParser):
            
            def __init__(self, **kwargs):
                super(ArgumentParser, self).__init__(formatter_class=argparse.ArgumentDefaultsHelpFormatter, **kwargs)
            
            def _get_action_from_name(self, name):
                """Given a name, get the Action instance registered with this parser.
                If only it were made available in the ArgumentError object. It is
                passed as it's first arg...
                """
                container = self._actions
                if name is None:
                    return None
                for action in container:
                    if '/'.join(action.option_strings) == name:
                        return action
                    elif action.metavar == name:
                        return action
                    elif action.dest == name:
                        return action

            def error(self, message):
                exc = sys.exc_info()[1]
                if exc:
                    exc.argument = self._get_action_from_name(exc.argument_name)
                    raise exc
                super(ArgumentParser, self).error(message)

        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('names', help='name of the training', nargs='+')
        parser.add_argument('--temp', help='if only show data that is temporary or not (0/1/-1)', type=int, default=-1)

        parser.add_argument('--host', help='host of the mongodb', type=str, default=DEFAULT_HOST)
        parser.add_argument('-p', '--port', help='port of the mongodb', type=int, default=DEFAULT_PORT)
        parser.add_argument('--link', help='config name of the mongodb', type=str, default=None)
        parser.add_argument('--db', help='name of the db', type=str, default=DEFAULT_DB_NAME)
        parser.add_argument('--prefix', help='prefix of the tables', type=str, default=DEFAULT_TABLE_PREFIX)
        parser.add_argument('--service_host', help='host of the locate service', type=str, default='localhost')
        parser.add_argument('--service_port', help='port of the locate service', type=int, default=DEFAULT_LOCATE_SERVICE_PORT)
        parser.add_argument('--save_dir', help='save dir of the save_rel_path', type=str, default=None)

        parser.add_argument('--hyper', help='the default hyper param to check', type=str, default='lr')
        parser.add_argument(
            '-m',
            '--metrics',
            help='CSV list of default metrics groups, each group is vertical bar separated string',
            type=str,
            default='loss'
        )
        parser.add_argument(
            '-k',
            '--keys',
            help='CSV list of keys that will be shown in the table',
            type=str,
            default=None
        )
        parser.add_argument(
            '--batch_metrics',
            help='CSV list of default metrics groups for batch, each group is vertical bar separated string',
            type=str,
            default='cost'
        )

        args = parser.parse_args()

        name = args.names
        temp_value = args.temp
        if -1 == temp_value:
            temp_arg = None
        else:
            temp_arg = temp_value

        host = args.host
        port = args.port
        link = args.link
        if link is not None:
            client = conn(link)
            host = client.HOST
            port = client.PORT
        else:
            client = None
            link = None
        
        db = args.db
        prefix = args.prefix

        service_host = args.service_host
        service_port = args.service_port

        save_dir = args.save_dir

        hyper = args.hyper
        metrics_groups = args.metrics
        keys = args.keys
        metrics_groups_4batch = args.batch_metrics

        if client is not None:
            tvtsv = TvtsVisualization(
                name, temp_arg, host, port, db, prefix, save_dir, 
                client=client, 
                service_host=service_host, service_port=service_port,
            )
        else:
            tvtsv = TvtsVisualization(
                name, temp_arg, host, port, db, prefix, save_dir, 
                service_host=service_host, service_port=service_port,
            )

        def plot_train_id(
                xtrain_id,
                xmetrics_groups=metrics_groups,
                xhyper=None,
                is_4batch=False,
                xmetrics_groups_4batch=None,
                xepoch_range='0:0',
                xbatch_range='0:0',
        ):
            try:
                xhyper_str = ''
                if xhyper is not None:
                    xhyper_str = f'with hyper parameter {xhyper}'
                if not is_4batch:
                    print(f'> Visualizing training with id of {xtrain_id} with' \
                        f' metrics_groups {xmetrics_groups} {xhyper_str}')
                    tvtsv.plot(xtrain_id, xmetrics_groups, xhyper)
                else:
                    print(f'> Visualizing batch records of training with id of {xtrain_id} with' \
                          f' metrics_groups {xmetrics_groups_4batch} {xhyper_str}')
                    tvtsv.plot_4batch(xtrain_id, xmetrics_groups_4batch, xhyper, xepoch_range, xbatch_range)
            except Exception as ex:
                print(traceback.format_exc())

        cli_parser = ArgumentParser(prog='Input:')
        cli_parser.add_argument('train_id', help='id of the training you want to check', type=int, nargs='?', default=0)
        cli_parser.add_argument(
            '-m', '--metrics',
            help='CSV list of metrics groups, each group is vertical bar separated string',
            type=str, default=None
        )
        cli_parser.add_argument(
            '--batch_metrics',
            help='CSV list of metrics groups for batch, each group is vertical bar separated string',
            type=str, default=None
        )
        cli_parser.add_argument('--hyper', help='the hyper param to check', type=str, default=None)
        cli_parser.add_argument('-b', '--batch', help='only check batch records of this train_id', action='store_true')
        cli_parser.add_argument(
            '--epoch_range',
            help='the epoch range when check batch records',
            type=str, default='0:0'
        )
        cli_parser.add_argument(
            '--batch_range',
            help='the batch range when check batch records',
            type=str, default='0:0'
        )

        regexp4sepc_keys = re.compile(r'^\s*(m|bm|hyper|keys)=(\S+)\s*$', re.IGNORECASE)
        regexp4spec_temp = re.compile(r'^\s*temp\s*\=\s*(0|1|-1)\s*$', re.IGNORECASE)
        regexp4spec_dir = re.compile(r'^\s*dir\s*\=\s*(.*)$', re.IGNORECASE)

        def parse_input():
            nonlocal metrics_groups, metrics_groups_4batch, hyper, keys, temp_value

            try:
                the_none_return_value = [None] * 7
                the_empty_return_value = [0]
                the_empty_return_value.extend([None] * 6)

                xinput = input().strip()

                if len(xinput) == 0:
                    return the_none_return_value

                matcher = regexp4sepc_keys.search(xinput)
                if matcher:
                    var_name = matcher.group(1).lower()
                    val = matcher.group(2)
                    if 'm' == var_name:
                        metrics_groups = val
                        print(f'> m={val}')
                    elif 'bm' == var_name:
                        metrics_groups_4batch = val
                        print(f'> bm={val}')
                    elif 'hyper' == var_name:
                        hyper = val
                        print(f'> hyper={val}')
                    elif 'keys' == var_name:
                        keys = val
                        print(f'> keys={val}')
                    else:
                        return the_empty_return_value
                    return the_empty_return_value

                matcher = regexp4spec_temp.search(xinput)
                if matcher:
                    temp_value = int(matcher.group(1))
                    if temp_value == -1:
                        is_temp = None
                    else:
                        is_temp = not not temp_value
                    tvtsv.setTemp(is_temp)
                    print(f'> temp={temp_value}')
                    return the_empty_return_value

                matcher = regexp4spec_dir.search(xinput)
                if matcher:
                    save_dir_value = matcher.group(1).strip()
                    if not len(save_dir_value):
                        save_dir = None
                    else:
                        save_dir = save_dir_value
                    tvtsv.setSaveDir(save_dir)
                    print(f'> save_dir="{save_dir}"')
                    return the_empty_return_value

                xinput_lower = xinput.lower()
                if 'q' == xinput_lower or 'quit' == xinput_lower:
                    return []
                try:
                    xlist = xinput.split()
                    xargs = cli_parser.parse_args(xlist)
                    xtrain_id = xargs.train_id
                    if not xtrain_id:
                        print('train_id must be provided!')
                        raise ValueError()

                    xmetrics_groups = xargs.metrics
                    xhyper = xargs.hyper
                    is_4batch = xargs.batch
                    xmetrics_groups_4batch = xargs.batch_metrics
                    xepoch_range = xargs.epoch_range
                    xbatch_range = xargs.batch_range
                    return xtrain_id, xmetrics_groups, xhyper, is_4batch, xmetrics_groups_4batch, xepoch_range, xbatch_range
                except ValueError:
                    print(f'> Invalid input!')
                    return None
            except argparse.ArgumentError as ex:
                print(f'> Invalid input for argparse!: {ex}')
                return None
            except Exception as ex:
                print(f'Exception: {ex}')
                return None

        def input_value_util_well():
            while True:
                input_list = parse_input()
                if input_list is None:
                    continue
                else:
                    break
                time.sleep(0.2)
            return input_list

        while True:
            print('> ')
            tvtsv.show_title()
            print(f'> Help info:')
            print(f'> Directly press ENTER to show the summary table. Input q/Q/quit to quit.')
            print(f'> Or: Input "m/bm/hyper/keys=value" to change corresponding keys.')
            print(f'> Or: Input "temp=0/1/-1" to show summary table of only formal data, only temporary data, or all data.')
            print(f'> Or: Input "dir=/path/of/dir/of/saved/weights" to specify save_dir.')
            print(f'> Or: Do the plotting by CLI command like: [-m METRICS(default: {metrics_groups})]' \
                  f' [--batch_metrics BATCH_METRICS(default: {metrics_groups_4batch})]' \
                  f' [--hyper HYPER=(default: {hyper})] [-b] [--epoch_range(default: 0:0)] [--batch_range(default: 0:0)] train_id')
            input_list = input_value_util_well()
            if len(input_list) == 0:
                print('> Bye!')
                break
            xx_train_id, xx_metrics_groups, xx_hyper, xx_is_4batch, xx_metrics_groups_4batch, xx_epoch_range, xx_batch_range = input_list
            if xx_train_id == 0:
                continue
            if xx_train_id is None:
                print('> Loading ...', flush=True)
                tvtsv.summary(keys)
                continue
            if xx_metrics_groups is None:
                xx_metrics_groups = metrics_groups
            if xx_hyper is None:
                xx_hyper = hyper
            if xx_metrics_groups_4batch is None:
                xx_metrics_groups_4batch = metrics_groups_4batch
            plot_train_id(
                xx_train_id,
                xx_metrics_groups,
                xx_hyper,
                xx_is_4batch,
                xx_metrics_groups_4batch,
                xx_epoch_range,
                xx_batch_range,
            )

    _main()
