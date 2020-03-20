import boto3
import json
from threading import Thread

def thread_call():
    faas = boto3.client('lambda')
    for num in [100, 1000]:
        parameter={"multiply": str(num)}
        res = faas.invoke(FunctionName="testLambda", InvocationType="Event", Payload=json.dumps(parameter))
    del faas

def memcached_lambda_threadcall_test():
    faas = boto3.client('lambda')
    th1 = Thread(target=thread_call)
    th1.start()
    th1.join()
    del faas

def thread_call_noloop():
    faas = boto3.client('lambda')
    for num in range(0, 2000):
        param = {'multiply' : str(num)}
        res = faas.invoke(FunctionName="testLambda", InvocationType="Event", Payload=json.dumps(param))

    del faas

def memcached_lambda_threadcall_noloop_test():
    faas = boto3.client('lambda')
    th1 = Thread(target=thread_call_noloop)
    th1.start()
    th1.join()
    del faas
