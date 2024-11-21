if '__main__' == __name__:
    import argparse
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    import uvicorn
    import os
    from tvts import DEFAULT_LOCATE_SERVICE_PORT, LOCAT_SERVICE_PWD
    from PyCmpltrtok.common import md5, has_content
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--port', help='port of this service', type=str, default=DEFAULT_LOCATE_SERVICE_PORT)
    args = parser.parse_args()
    port = args.port
    
    app = FastAPI()
    
    @app.post("/api", response_class=JSONResponse)
    async def do_infer(request: Request):
        # 接收输入
        req_json = await request.json()  # 请求json

        path = req_json['path']
        xcheck = req_json['check']
        xmy_check = md5(path + LOCAT_SERVICE_PWD)
        if xcheck.lower() != xmy_check.lower():
            print(f'Checking not passed! path=|{path}|')
            return None

        isdir, result = has_content(path)
        print('path:', path, 'isdir', isdir, 'result:', result)

        # 返回输出
        res_dict = dict()
        res_dict['path'] = path
        res_dict['result'] = result

        # https://stackoverflow.com/questions/71794990/fast-api-how-to-return-a-str-as-json
        return JSONResponse(content=res_dict)
    
    uvicorn.run(app, host='0.0.0.0', port=port)
    