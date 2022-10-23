import re
import pymongo as pm
import os
import sys
import datetime
import matplotlib.pyplot as plt
from texttable import Texttable
import numpy as np
from typing import Optional, List, Sequence

DEFAULT_HOST = 'localhost'
DEFAULT_PORT = 27017
DEFAULT_DB_NAME = 'tvts'
DEFAULT_TABLE_PREFIX = 'train_log'
DEFAULT_SAVE_FREQ = 1
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


def long_text_to_block(xm_text: str, xm_width: int) -> str:
    assert isinstance(xm_text, str)
    assert isinstance(xm_width, int)

    xi_len_of_text = len(xm_text)
    xi_n_lines = int(np.ceil(xi_len_of_text / xm_width))
    xi_lines = [xm_text[i * xm_width:(i + 1) * xm_width] for i in range(xi_n_lines)]
    xm_text = '\n'.join(xi_lines)
    return xm_text


class Tvts(object):

    def __init__(
            self,
            name: str,
            memo: Optional[str] = '(No memo)',
            is_temp: Optional[bool] = False,
            host: Optional[str] = DEFAULT_HOST,
            port: Optional[int] = DEFAULT_PORT,
            db: Optional[str] = DEFAULT_DB_NAME,
            table_prefix: Optional[str] = DEFAULT_TABLE_PREFIX,
            save_freq: Optional[int] = DEFAULT_SAVE_FREQ,
            save_dir: Optional[str] = None,
            init_weights: Optional[str] = None,
            params: Optional[dict] = {}
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
        if save_dir is None:
            raise TvtsException(f'Please specify the save_dir at where weights are saved!')
        else:
            assert isinstance(save_dir, str)
        if init_weights is not None:
            assert isinstance(init_weights, str) and len(init_weights) > 0

        memo = '(tvts.py) ' + memo
        self.name = name
        self.memo = memo
        self.is_temp = is_temp

        self.host = host
        self.port = port
        self.db_name = db
        self.table_name = table_prefix + '_' + name
        self.table_name_4batch = table_prefix + '_' + name + '_4batch'

        self.params = params
        for k in self.params.keys():
            if k in RESERVED_KEYS:
                raise TvtsException(f'Key "{k}" is reserved and cannot be used by the user.')

        self.save_freq = save_freq
        self.save_dir = save_dir
        self.init_weights = init_weights

        # conn
        self.conn()

        # get train id
        self.train_id = self._get_next_train_id()
        print(f'> TVTS: id of this training is {self.train_id}', file=sys.stderr)

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

    def conn(self) -> None:
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
        assert isinstance(xepoch, int)
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
            xepoch = int(xepoch)
            if xepoch <= 0:
                raise TvtsException(f'Epoch value {xepoch} must be >= 1!')
            ckpt = self.table.find_one({
                'train_id': xtrain_id,
                'epoch': xepoch,
            })

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
        self.params['parent_epoch'] = int(ckpt['epoch'])

        return save_rel_path, ckpt['save_dir']

    def get_save_name(self, epoch: int) -> str:
        assert isinstance(epoch, int)
        return f'{self.name}-{self.train_id}-{epoch}'

    def save_batch(
            self,
            xepoch: int,
            xbatch: int,
            params: Optional[dict] = {},
            is_batch_global: Optional[bool] = False
    ) -> None:
        req = 'Epoch and Batch must be integers equal to or greater than 1 !'
        ex = TvtsException(req)
        try:
            xepoch = int(xepoch)
            xbatch = int(xbatch)
            if xepoch <= 0:
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
        data = self.params.copy()
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

        # insert data into db
        # IMPORTANT: db and table will be newly created at the 1st insertion if they are not there yet.
        self.table_4batch.insert_one(data)

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
            xepoch: int,
            params: Optional[dict] = {},
            save_rel_path=None,
            save_dir=None
    ) -> None:
        req = 'Epoch must be a integer equal to or greater than 1 !'
        ex = TvtsException(req)
        try:
            xepoch = int(xepoch)
            if xepoch <= 0:
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
        data = self.params.copy()
        # apply update
        for k in params.keys():
            v = params[k]
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

        # insert data into db
        # IMPORTANT: db and table will be newly created at the 1st insertion if they are not there yet.
        self.table.insert_one(data)

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
        name: str,
        is_temp: Optional[bool] = None,
        host: Optional[str] = DEFAULT_HOST,
        port: Optional[int] = DEFAULT_PORT,
        db: Optional[str] = DEFAULT_DB_NAME,
        table_prefix: Optional[str] = DEFAULT_TABLE_PREFIX,
        save_dir: Optional[str] = None
    ):
        assert isinstance(name, str)
        if is_temp is not None:
           assert isinstance(is_temp, bool)
        assert isinstance(host, str)
        assert isinstance(port, int)
        assert isinstance(db, str)
        assert isinstance(table_prefix, str)
        if save_dir is not None:
            assert isinstance(save_dir, str)

        self.name = name
        if is_temp is not None:
            self.temp_value = int(is_temp)
        else:
            self.temp_value = None
        self.host = host
        self.port = port
        self.db_name = db
        self.table_name = table_prefix + '_' + self.name
        self.table_name_4batch = table_prefix + '_' + self.name + '_4batch'
        self.save_dir = save_dir

        # conn
        self.client = pm.MongoClient(host, port)
        self.db = self.client[self.db_name]
        self.table = self.db[self.table_name]
        self.table_4batch = self.db[self.table_name_4batch]

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
        xinfo_bar = f'Name: {self.name} from {self.host}:{self.port} {self.db_name}.{self.table_name}' \
                    f' @{str(datetime.datetime.now())[:-3]} ({temp_repr})\nSpecified save_dir: "{self.save_dir}"'
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

            'epoch\ncount',
            'save\nexisted/all',
            'save\ndir',

            'from', 'to',
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
        cursor = self.table.aggregate(pipeline)
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
        cursor4batch = self.table_4batch.aggregate(pipeline4batch)

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
            cur = self.table_4batch.find({
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
            cur = self.table.find({
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
                    if os.path.exists(abs_path):
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
                f'{last_row.get("parent_id", 0)}-{last_row.get("parent_epoch", 0)}',
                long_text_to_block(str(last_row.get('init_weights', '')), LONG_TEXT_WIDTH_OF_COLUMN),

                count,
                f'{count_path_exists}/{count_path}',
                long_text_to_block(str(last_row.get('save_dir', '')), LONG_TEXT_WIDTH_OF_COLUMN),

                shorter_dt(first_row['from_datetime']),
                shorter_dt(last_row['to_datetime']),
                last_row['to_datetime'] - first_row['from_datetime'],
                train2batch_cnt_map.get(train_id, 0),

                *xvalues_list,

                long_text_to_block(str(last_row.get('memo', '')), LONG_TEXT_WIDTH_OF_COLUMN),
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
                    f'{last_row.get("parent_id", 0)}-{last_row.get("parent_epoch", 0)}',
                    long_text_to_block(str(last_row.get('init_weights', '')), LONG_TEXT_WIDTH_OF_COLUMN),

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

    def plot_4batch(
            self,
            xtrain_id=1,
            xmetrics_groups='cost',
            xhyper=None,
            xepoch_range='0:0'
    ) -> None:
        assert isinstance(xtrain_id, int)
        assert isinstance(xmetrics_groups, str)
        if xhyper is not None:
            assert isinstance(xhyper, str)
        assert isinstance(xepoch_range, str)

        xepoch_begin, xepoch_end = xepoch_range.split(':')
        xepoch_begin = int(xepoch_begin)
        xepoch_end = int(xepoch_end)

        # parse metrics groups
        xmetrics_groups = xmetrics_groups.split(',')
        groups = []
        for group in xmetrics_groups:
            groups.append(group.split('|'))
        n_groups = len(groups)
        keys = []
        for group in groups:
            keys.extend(group)
        if xhyper is not None:
            keys.append(xhyper)

        def apply_restrictions(xc_query):
            if not xepoch_begin and not xepoch_end:
                return
            xcdict = {}
            if xepoch_begin:
                xcdict['$gte'] = xepoch_begin
            if xepoch_end:
                xcdict['$lte'] = xepoch_end
            xc_query['epoch'] = xcdict

        query = {
            'train_id': xtrain_id,
            'global_batch': {
                '$ne': None
            }
        }
        apply_restrictions(query)
        cursor = self.table_4batch.find(query).sort([
            ('global_batch', pm.ASCENDING),
        ])

        # fetch the plot values
        cnt = 0
        rows_list = []
        for row in cursor:
            cnt += 1
            rows_list.append(row)

        # no global batch number available, translate local batch number
        is_global = True
        if cnt <= 0:
            is_global = False
            query = {
                'train_id': xtrain_id,
            }
            apply_restrictions(query)
            cursor = self.table_4batch.find(query).sort([
                ('epoch', pm.ASCENDING),
                ('batch', pm.ASCENDING),
            ])

            # base of each epoch is an accumulated value
            epoch_count_map, epoch_base_map = {}, {}
            epoch_list = []
            rows_list = []
            for row in cursor:
                xepoch = int(row['epoch'])
                rows_list.append(row)
                cnt = epoch_count_map.get(xepoch, None)
                if cnt is None:
                    epoch_count_map[xepoch] = 0
                    epoch_list.append(xepoch)
                    epoch_base_map[xepoch] = 0
                epoch_count_map[xepoch] += 1
            for i, xepoch in enumerate(epoch_list[:-1]):
                base = epoch_count_map[xepoch]
                for impacted_epoch in epoch_list[i + 1:]:
                    epoch_base_map[impacted_epoch] += base

        if len(rows_list) <= 0:
            print(f'> Error: find no batch record with train_id {xtrain_id}!')
            return

        # fetch the plot values
        epoch_bar_epoch2x_map = {}
        x_map, y_map = {}, {}
        for k in keys:
            y_map[k] = []
            x_map[k] = []
        xlast_epoch = 0
        for row in rows_list:
            xepoch = int(row['epoch'])
            if is_global:
                x = int(row['global_batch'])
            else:
                base = int(epoch_base_map[xepoch])
                x = int(row['batch']) + base
            if xepoch > xlast_epoch:
                epoch_bar_epoch2x_map[xepoch] = x
            xlast_epoch = xepoch
            for k in keys:
                v = row.get(k, None)
                if v is None:
                    continue
                v = float(v)
                y_map[k].append(v)
                x_map[k].append(x)

        # plot them
        spr = 1
        spc = n_groups
        spn = 0
        fig = plt.figure(figsize=[5 * n_groups, 5])
        plt.subplots_adjust(wspace=0.4)
        epoch_bars = epoch_bar_epoch2x_map.keys()
        n_epoch_bars = len(epoch_bars)
        for group in groups:
            spn += 1
            ax2 = plt.subplot(spr, spc, spn)
            ax1 = ax2.twinx()
            ax2.set_ylabel(xhyper)
            ax1.set_title(f'{self.name}@{self.host}:{self.port}\n#{xtrain_id} {group}')

            if xhyper is not None:
                ax2.plot(x_map[xhyper], y_map[xhyper], color='black')

            for k in group:
                ax1.plot(x_map[k], y_map[k], label=k, linewidth=1, alpha=0.7)

            # separator of different epochs
            if n_epoch_bars <= 5:
                y_min, y_max = ax1.get_ylim()
                for xi_epoch in epoch_bars:
                    x = epoch_bar_epoch2x_map[xi_epoch]
                    ax1.plot([x, x], (y_min, y_max), color='blue')
                    ax1.annotate(f'#e{xi_epoch}', xy=[x, y_max], color='blue')

            ax1.legend()
            ax1.grid()
        print('> Check and close the plotting window to continue ...')
        plt.show()

    def plot(
            self,
            xm_train_id: Optional[int] = 1,
            xm_metrics_groups: Optional[str] = 'cost',
            xm_since_train_id: Optional[int] = 0,
            xm_hyper: Optional[str] = None
    ) -> None:
        """
        Plot the train with xm_train_id and its ancestors showing specified metrics and hyper params
        from the train with id of xm_since_train_id
        or from the eldest ancestor if the xm_since_train_id is 0.

        :param xm_train_id: The id of the train to plot.
        :param xm_metrics_groups: The metrics groups specifier. Eg. "cost|cost_val,acc|acc_val|f1|f1_val"
        :param xm_since_train_id: If the xm_since_train_id > 0, consider this as the root ancestor's train_id.
        :param xm_hyper: The hyper param to check. Eg. "lr"
        :return: void
        """
        assert isinstance(xm_train_id, int)
        assert isinstance(xm_metrics_groups, str)
        assert isinstance(xm_since_train_id, int)
        if xm_hyper is not None:
            assert isinstance(xm_hyper, str)

        # get xm_since_train_id
        if xm_since_train_id:
            xm_since_train_id = int(xm_since_train_id)
            if xm_since_train_id < 0:
                xm_since_train_id = 0

        # get xm_metrics_groups
        xm_metrics_groups = xm_metrics_groups.split(',')
        xm_groups = []
        for group in xm_metrics_groups:
            xm_groups.append(group.split('|'))
        xm_n_groups = len(xm_groups)
        xm_keys = []
        for group in xm_groups:
            xm_keys.extend(group)
        if xm_hyper is not None:
            xm_keys.append(xm_hyper)

        def find_record(xc_train_id, xc_up_to_epoch=0):
            """
            Enclosure method.
            Find records of training with id of xc_train_id up to epoch of xc_up_to_epoch.

            :param xc_train_id: The training's id.
            :param xc_up_to_epoch: Up to this epoch.
            :return: dict or None
            """

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
            cursor = self.table.find(query).sort('epoch', pm.ASCENDING)

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
            xc_epoch2row_map = {}
            for i, row in enumerate(cursor):
                count += 1
                xi_epoch = int(row['epoch'])
                xc_epoch2row_map[xi_epoch] = row
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
                'up_to_epoch': int(xc_last_row['epoch']),
                'parent_id': int(xc_last_row.get('parent_id', 0)),
                'parent_epoch': int(xc_last_row.get('parent_epoch', 0)),
                'base_epoch': 0,
                'epoch2row_map': xc_epoch2row_map,
            }
            return xc_map

        # recursively get data of ancestor trainings
        xm_train_id2map_map = {}
        xm_set_train_ids, xm_list_train_ids = set(), []
        xi_train_id = xm_train_id
        xi_up_to_epoch = 0
        while True:
            if xi_train_id in xm_set_train_ids:
                raise TvtsException(f'Error: loop back! {xm_list_train_ids} meet {xi_train_id}')
            xm_set_train_ids.add(xi_train_id)
            xm_list_train_ids.append(xi_train_id)
            xi_map = find_record(xi_train_id, xi_up_to_epoch)
            if xi_map is None:
                print(f'> Error: find no record with train_id {xi_train_id} !' \
                      f' If there are only batch records, please add -b to arguments.')
                return
            xm_train_id2map_map[xi_train_id] = xi_map
            xi_train_id = int(xi_map.get('parent_id', 0))
            xi_up_to_epoch = int(xi_map.get('parent_epoch', 0))
            if not xi_train_id:  # if it is the eldest ancestor
                break
        # order by datetime
        xm_list_train_ids = xm_list_train_ids[::-1]

        # validate ancestors
        if xm_since_train_id and not xm_since_train_id in xm_set_train_ids:
            raise TvtsException(f'The training id ({xm_since_train_id}) you specified as "since"' \
                                f' is not an ancestor of this training ({xm_train_id})!')

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

        # Make data to finally plot
        # and apply the "since" training id that user specified
        xm_key2glb_x_map, xm_key2y_map, xm_key2glb_x_map_with_save_path, xm_key2y_map_with_save_path = {}, {}, {}, {}
        for k in xm_keys:
            xm_key2glb_x_map[k] = []
            xm_key2y_map[k] = []
            xm_key2glb_x_map_with_save_path[k] = []
            xm_key2y_map_with_save_path[k] = []
        if not xm_since_train_id:
            xm_list_train_ids_show = xm_list_train_ids.copy()
        else:
            xm_list_train_ids_show = []
            start = False
            for xi_train_id in xm_list_train_ids:
                if xi_train_id == xm_since_train_id:
                    start = True
                if start:
                    xm_list_train_ids_show.append(xi_train_id)
        for xi_train_id in xm_list_train_ids_show:
            xi_map = xm_train_id2map_map[xi_train_id]
            for k in xm_keys:
                xm_key2glb_x_map[k].extend(xi_map['key2glb_x_map'][k])
                xm_key2y_map[k].extend(xi_map['key2y_map'][k])
                xm_key2glb_x_map_with_save_path[k].extend(xi_map['key2glb_x_map_with_save_path'][k])
                xm_key2y_map_with_save_path[k].extend(xi_map['key2y_map_with_save_path'][k])
        n_xm_list_train_ids_show = len(xm_list_train_ids_show)

        # plot
        if n_xm_list_train_ids_show < 1:
            raise TvtsException(f'Error: n_xm_list_train_ids_show < 1 !')
        if n_xm_list_train_ids_show == 1:
            title_epoch = f'#{xm_list_train_ids_show[0]}'
        else:
            title_epoch = f'#{xm_list_train_ids_show[0]}~#{xm_list_train_ids_show[-1]}'
            pass
        xm_hover_box_width = 20
        xm_hover_box_xoffset_abs, xm_hover_box_yoffset_abs = 20, 20
        spr = 1
        spc = xm_n_groups
        spn = 0
        fig = plt.figure(figsize=[5*xm_n_groups, 5])
        plt.subplots_adjust(wspace=0.4, top=0.8)
        xm_list_of_tuple_ax_dots_key = []
        xm_ax2annot_map = {}
        for group in xm_groups:
            spn += 1
            xi_ax2 = plt.subplot(spr, spc, spn)
            xi_ax1 = xi_ax2.twinx()

            xi_ax2.set_ylabel(xm_hyper)
            xi_ax1.set_title(f'{self.name}@{self.host}:{self.port}\n{title_epoch} {group}')

            # plot hyper param
            if xm_hyper is not None:
                xxx, yyy = [], []
                for xi_train_id in xm_list_train_ids_show:
                    xi_map = xm_train_id2map_map[xi_train_id]
                    xxx.extend(xi_map['key2glb_x_map'][xm_hyper])
                    yyy.extend(xi_map['key2y_map'][xm_hyper])
                xi_ax2.plot(xxx, yyy, color='black', alpha=0.7, label=xm_hyper)

            # plot metrics
            for k in group:
                # the curve
                dots = xi_ax1.plot(xm_key2glb_x_map[k], xm_key2y_map[k], label=k, linewidth=1, alpha=0.7)
                # the dots representing epoch with saved model or weights
                xi_ax1.scatter(
                    xm_key2glb_x_map_with_save_path[k],
                    xm_key2y_map_with_save_path[k],
                    s=4, color='black', marker='s', alpha=0.7
                )
                # collect data for drawing when hovering over the curve
                dots = dots[0]
                xm_list_of_tuple_ax_dots_key.append((xi_ax1, dots, k))

            # separator vertical line for ancestor trainings
            for xi_train_id in xm_list_train_ids_show:
                xi_map = xm_train_id2map_map[xi_train_id]
                y_min, y_max = xi_ax1.get_ylim()
                base = xi_map['base_epoch'] + 1
                xi_ax1.plot([base, base], [y_min, y_max], color='black', alpha=0.7)
                xi_ax1.annotate(f'#id_{xi_train_id}(glb_{base})', xy=[base, y_max], color='black', alpha=0.7)

            xi_ax1.legend()
            xi_ax1.grid()
            xi_ax1.set_xlabel('Global Epoch')
            xi_ax2.legend()

            xi_last_base = xm_train_id2map_map[xm_list_train_ids_show[-1]]['base_epoch']
            xi_2nd_xaxis = xi_ax1.secondary_xaxis(
                'top',
                functions=(lambda x: x - xi_last_base, lambda x: x + xi_last_base)
            )
            # xi_2nd_xaxis.set_xlabel('Epoch in this turn')

            xi_annot = xi_ax1.annotate(
                "",
                xy=(0, 0),
                xytext=(xm_hover_box_xoffset_abs, xm_hover_box_yoffset_abs),
                textcoords="offset points",
                bbox=dict(boxstyle="round", fc="w"),
                arrowprops=dict(arrowstyle="->"),
            )
            xi_annot.set_visible(False)
            xm_ax2annot_map[id(xi_ax1)] = xi_annot

        def update_annot(xc_index, xc_ax, xc_dots, xc_key):
            """
            Enclosure method.
            Update the hover info box

            :param xc_index: x-index
            :param xc_ax: hover event from which ax
            :param xc_dots: hover event from dots of which plot
            :param xc_key: hover event from plot of which key
            :return: void
            """

            # get position
            xc_index = xc_index["ind"][0]
            xc_global_epoch = xc_index + 1
            xc_position = xc_dots.get_xydata()[xc_index]
            xc_y_val = xc_position[1]
            xc_y_repr = get_better_repr(xc_y_val)

            # get data from position
            xc_text = None
            for xi_train_id in xm_list_train_ids_show[::-1]:
                xi_map = xm_train_id2map_map[xi_train_id]
                xi_epoch2row_map = xi_map['epoch2row_map']
                xi_base_epoch = xi_map['base_epoch']
                if xc_global_epoch <= xi_base_epoch:
                    continue
                xi_local_epoch = xc_global_epoch - xi_base_epoch

                row = xi_epoch2row_map[xi_local_epoch]
                xi_hyper_val = row[xm_hyper]
                xi_hyper_repr = get_better_repr(xi_hyper_val)
                xc_text = f'global epoch={xc_global_epoch}, train_id={xi_train_id}, epoch={xi_local_epoch},' \
                          f' {xc_key}={xc_y_repr}, {xm_hyper}={xi_hyper_repr}'
                xc_text = long_text_to_block(xc_text, xm_hover_box_width)
                break  # just show the first match
            if xc_text is None:  # no matched data
                return

            # decide where to put the hover info box
            xmin, xmax = xc_ax.get_xlim()
            ymin, ymax = xc_ax.get_ylim()
            xc_x_mid = (xmin + xmax) / 2
            xc_y_mid = (ymin + ymax) / 2
            if xc_global_epoch > xc_x_mid:
                # right: show at left
                # print('R')
                xc_hover_box_xoffset = - xm_hover_box_xoffset_abs
            else:
                # left: show at right
                # print('L')
                xc_hover_box_xoffset = xm_hover_box_xoffset_abs
            if xc_y_val > xc_y_mid:
                # upper: show at lower
                # print('U')
                xc_hover_box_yoffset = - xm_hover_box_yoffset_abs
            else:
                # lower: show at upper
                # print('L')
                xc_hover_box_yoffset = xm_hover_box_yoffset_abs

            # draw the hover info box and show it
            xc_annot = xm_ax2annot_map[id(xc_ax)]
            xc_annot.xy = xc_position
            xc_annot.set_text(xc_text)
            xc_bbox = xc_annot.get_bbox_patch()
            xc_bbox.set_facecolor('blue')
            xc_bbox.set_alpha(0.85)
            xc_bbox_h, xc_bbox_w = xc_bbox.get_height(), xc_bbox.get_width()
            if xc_global_epoch > xc_x_mid:
                if xc_y_val > xc_y_mid:
                    # right, upper
                    # print('RU')
                    xc_hover_box_xoffset -= xc_bbox_w
                    xc_hover_box_yoffset -= xc_bbox_h
                else:
                    # right, lower
                    # print('RL')
                    xc_hover_box_xoffset -= xc_bbox_w
            else:
                if xc_y_val > xc_y_mid:
                    # left, upper
                    # print('LU')
                    xc_hover_box_yoffset -= xc_bbox_h
                else:
                    # left, lower
                    # print('LL')
                    pass
            xc_annot.xyann = (xc_hover_box_xoffset, xc_hover_box_yoffset)
            xc_annot.set_visible(True)

        def hover(xc_event):
            """
            Enclosure callback method.
            Triggered by mouse hover event.

            :param xc_event: The event from matplotlib UI.
            :return: void
            """
            for ax, sc, key in xm_list_of_tuple_ax_dots_key:
                if xc_event.inaxes == ax:
                    cont, ind = sc.contains(xc_event)
                    if cont:
                        update_annot(ind, ax, sc, key)
                        fig.canvas.draw_idle()
                        break
                    else:
                        for annot in xm_ax2annot_map.values():
                            annot.set_visible(False)
                        fig.canvas.draw_idle()

        # Bind a callback to mouse hover events
        fig.canvas.mpl_connect('motion_notify_event', hover)
        # show the plot window
        print('> Check and close the plotting window to continue ...')
        plt.show()


if '__main__' == __name__:
    import argparse
    import time

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
        parser.add_argument('name', help='name of the training')
        parser.add_argument('--temp', help='if only show data that is temporary or not (0/1/-1)', type=int, default=-1)

        parser.add_argument('--host', help='host of the mongodb', type=str, default=DEFAULT_HOST)
        parser.add_argument('-p', '--port', help='port of the mongodb', type=int, default=DEFAULT_PORT)
        parser.add_argument('--db', help='name of the db', type=str, default=DEFAULT_DB_NAME)
        parser.add_argument('--prefix', help='prefix of the tables', type=str, default=DEFAULT_TABLE_PREFIX)

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

        name = args.name
        temp_value = args.temp
        if -1 == temp_value:
            temp_arg = None
        else:
            temp_arg = temp_value

        host = args.host
        port = args.port
        db = args.db
        prefix = args.prefix

        save_dir = args.save_dir

        hyper = args.hyper
        metrics_groups = args.metrics
        keys = args.keys
        metrics_groups_4batch = args.batch_metrics

        tvtsv = TvtsVisualization(name, temp_arg, host, port, db, prefix, save_dir)

        def plot_train_id(
                xtrain_id,
                xmetrics_groups=metrics_groups,
                xsince=0,
                xhyper=None,
                is_4batch=False,
                xmetrics_groups_4batch=None,
                xepoch_range='0:0'
        ):
            try:
                xhyper_str = ''
                if xhyper is not None:
                    xhyper_str = f'with hyper parameter {xhyper}'
                if not is_4batch:
                    if xsince:
                        print(f'> Visualizing training with id of {xtrain_id} from id of {xsince}' \
                              f' with metrics_groups {xmetrics_groups} {xhyper_str}')
                    else:
                        print(f'> Visualizing training with id of {xtrain_id} with' \
                              f' metrics_groups {xmetrics_groups} {xhyper_str}')
                    tvtsv.plot(xtrain_id, xmetrics_groups, xsince, xhyper)
                else:
                    print(f'> Visualizing batch records ONLY of training with id of {xtrain_id} with' \
                          f' metrics_groups {xmetrics_groups_4batch} {xhyper_str}')
                    tvtsv.plot_4batch(xtrain_id, xmetrics_groups_4batch, xhyper, xepoch_range)
            except Exception as ex:
                print(ex)

        cli_parser = ArgumentParser(prog='Input:')
        cli_parser.add_argument('train_id', help='id of the training you want to check', type=int, nargs='?', default=0)
        cli_parser.add_argument('-s', '--since', help='since which ancestor id you want to check', type=int, default=0)
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
                    xsince_train_id = xargs.since
                    xmetrics_groups = xargs.metrics
                    xhyper = xargs.hyper
                    is_4batch = xargs.batch
                    xmetrics_groups_4batch = xargs.batch_metrics
                    xepoch_range = xargs.epoch_range
                    return xtrain_id, xmetrics_groups, xsince_train_id, xhyper, is_4batch, xmetrics_groups_4batch, xepoch_range
                except ValueError:
                    print(f'> Invalid input!')
                    return None
            except argparse.ArgumentError as ex:
                print(f'> Invalid input for argparse!: {ex}')
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
            print(f'> Or: Input "temp=0/1/-1" to show summary table of only temporary data, only formal data, or all data.')
            print(f'> Or: Input "dir=/path/of/dir/of/saved/weights" to specify save_dir.')
            print(f'> Or: Do the plotting by CLI command like: [-s SINCE] [-m METRICS(default: {metrics_groups})]' \
                  f' [--batch_metrics BATCH_METRICS(default: {metrics_groups_4batch})]' \
                  f' [--hyper HYPER=(default: {hyper})] [-b] [--epoch_range(default: 0:0)] train_id')
            input_list = input_value_util_well()
            if len(input_list) == 0:
                print('> Bye!')
                break
            xx_train_id, xx_metrics_groups, xx_since, xx_hyper, xx_is_4batch, xx_metrics_groups_4batch, xx_epoch_range = input_list
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
                xx_since,
                xx_hyper,
                xx_is_4batch,
                xx_metrics_groups_4batch,
                xx_epoch_range
            )

    _main()
