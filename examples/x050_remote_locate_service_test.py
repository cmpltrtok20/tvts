if '__main__' == __name__:
    from tvts.tvts import is_exist, DEFAULT_LOCATE_SERVICE_PORT
    
    host = '192.168.31.227'
    paths = [
        '/home/yunpeng/checkpoints/distilbert-base-uncased.on.imdb/',
        '/home/yunpeng/checkpoints/distilbert-base-uncased.on.imdb/empty',
        '/home/yunpeng/checkpoints/distilbert-base-uncased.on.imdb/distilbert-base-uncased.on.imdb_2024_03_29_18_03_02_796852_temp1_len64_freeze0_batch8_drop0.1/checkpoint-4',
        '/home/yunpeng/checkpoints/distilbert-base-uncased.on.imdb/distilbert-base-uncased.on.imdb_2024_03_29_18_03_02_796852_temp1_len64_freeze0_batch8_drop0.1/checkpoint-4/aaa',
        '/home/yunpeng/checkpoints/distilbert-base-uncased.on.imdb/distilbert-base-uncased.on.imdb_2024_03_29_18_03_02_796852_temp1_len64_freeze0_batch8_drop0.1/checkpoint-4/config.json',
    ]
    for p in paths:
        # result = is_exist(p, host, port=DEFAULT_LOCATE_SERVICE_PORT)
        result = is_exist(p, host)
        print(p, result)
        